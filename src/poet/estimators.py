"""Batch policy gradient estimators for a tabular softmax.

All gradients are sparse dicts: state -> gradient row. Episodes have variable
length; for the finite-horizon formulas they are treated as length-T episodes
that stay in an absorbing state with zero reward (and no action, hence zero
score) after termination.
"""
import numpy as np

from .policy import add_score

METHODS = ["reinforce", "reinforce_b", "gpomdp", "gpomdp_b"]
LABELS = {"reinforce": "REINFORCE", "reinforce_b": "REINFORCE + baseline",
          "gpomdp": "GPOMDP", "gpomdp_b": "GPOMDP + baseline b_t", "ac": "Actor-critic"}


def leave_one_out_mean(x: np.ndarray) -> np.ndarray:
    """b_i = (1/(n-1)) sum_{j != i} x_j, along axis 0.

    The average baseline computed on the same batch is slightly
    biased, because b then depends on tau_i itself. Leaving tau_i out makes b_i
    independent of tau_i, so E[score(tau_i) b_i] = E[score(tau_i)] E[b_i] = 0
    and the estimator stays exactly unbiased.
    """
    n = x.shape[0]
    assert n >= 2, "a leave-one-out baseline needs a batch of at least 2 episodes"
    return (x.sum(axis=0, keepdims=True) - x) / (n - 1)


def reinforce(trajs, tau: float, baseline: bool = False) -> dict:
    """REINFORCE, optionally with average baseline:

        g = (1/n) sum_i [ sum_t grad log pi(A_t^i | S_t^i) ] (R(tau_i) - b_i)
    """
    n = len(trajs)
    R = np.array([tr.ret for tr in trajs])
    b = leave_one_out_mean(R) if baseline else np.zeros(n)
    grad = {}
    for tr, coef in zip(trajs, R - b):
        for s, a, p in zip(tr.states, tr.actions, tr.probs):
            add_score(grad, s, p, a, coef / n, tau)
    return grad


def padded_rewards(trajs, T: int) -> np.ndarray:
    """n x T matrix r[i, t], zero after the end of episode i."""
    r = np.zeros((len(trajs), T))
    for i, tr in enumerate(trajs):
        r[i, :len(tr)] = tr.rewards
    return r


def gpomdp(trajs, tau: float, T: int, baseline: bool = False) -> dict:
    """GPOMDP, optionally with time-dependent baseline b_t:

        g = (1/n) sum_i sum_{t=0}^{T-1} (r_t^i - b_t) sum_{h=0}^{t} grad log pi(A_h^i | S_h^i)
          = (1/n) sum_i sum_h grad log pi(A_h^i | S_h^i) sum_{t=h}^{T-1} (r_t^i - b_t)

    (second form -> first form by swapping the sums; we compute the first form).
    b_t is the leave-one-out average of the per-step rewards r_t^j over the batch.

    The sum over t runs to the common horizon T, not to the length of episode i.
    So for a short episode the terms -b_t with t >= len(tau_i) still multiply its
    whole score. Dropping them would give a *biased* estimator, because whether
    step t exists depends on the earlier actions. Without a baseline these terms
    are zero (r = 0 after termination) and the two versions coincide.
    """
    n = len(trajs)
    r = padded_rewards(trajs, T)
    if baseline:
        r = r - leave_one_out_mean(r)
    to_go = np.cumsum(r[:, ::-1], axis=1)[:, ::-1]      # to_go[i, h] = sum_{t >= h} (r - b)
    grad = {}
    for i, tr in enumerate(trajs):
        for h, (s, a, p) in enumerate(zip(tr.states, tr.actions, tr.probs)):
            add_score(grad, s, p, a, to_go[i, h] / n, tau)
    return grad


def td_advantage(trajs, tau: float, V: dict) -> dict:
    """Batch version of the actor step of the actor-critic with a *fixed* critic V:

        g = (1/n) sum_i sum_h grad log pi(A_h^i | S_h^i) delta_h^i,
        delta_h = r_h + V(S_{h+1}) - V(S_h)          (V(terminal) = 0, unknown states -> 0)

    With V = V^pi, E[delta_h | S_h, A_h] = Q^pi(S_h, A_h) - V^pi(S_h) and g is unbiased.
    With any other V it is biased in general, because V(S_{h+1}) depends on A_h.
    Used only to measure variance and bias at a fixed policy (notebook 06).
    """
    n = len(trajs)
    grad = {}
    for tr in trajs:
        for h, (s, a, p, r) in enumerate(zip(tr.states, tr.actions, tr.probs, tr.rewards)):
            v_next = V.get(tr.states[h + 1], 0.0) if h + 1 < len(tr) else 0.0
            add_score(grad, s, p, a, (r + v_next - V.get(s, 0.0)) / n, tau)
    return grad


def estimate(method: str, trajs, tau: float, T: int) -> dict:
    """Dispatch by name: 'reinforce', 'reinforce_b', 'gpomdp', 'gpomdp_b'."""
    if method == "reinforce":
        return reinforce(trajs, tau, baseline=False)
    if method == "reinforce_b":
        return reinforce(trajs, tau, baseline=True)
    if method == "gpomdp":
        return gpomdp(trajs, tau, T, baseline=False)
    if method == "gpomdp_b":
        return gpomdp(trajs, tau, T, baseline=True)
    raise ValueError(method)
