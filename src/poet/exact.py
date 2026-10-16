"""Exact computations on a tiny poem MDP, by enumerating every trajectory (notebook 06).

With a handful of words and short lines, the poem MDP has a few hundred possible
poems. We can then compute J(theta), its gradient, the value function V^pi, and the
exact mean and variance of a batch estimator over *all* batches of 2 poems, with no
Monte Carlo error. This checks the unbiasedness of REINFORCE, GPOMDP and their baselines
(including b_t) on the code that is actually used for training.
"""
import numpy as np

from .bigram import Bigram
from .policy import Trajectory, add_score, env_kl_coef, kl_penalty


def restrict_bigram(lm: Bigram, words: list[str]) -> Bigram:
    """The bigram restricted to `words` (+ <s> and </s>), each row renormalised."""
    ids = [lm.vocab.index(w) for w in words]
    rows, cols = ids + [lm.BOS], ids + [lm.EOS]
    P = lm.P[np.ix_(rows, cols)]
    return Bigram(list(words), lm.syllables[ids], P / P.sum(axis=1, keepdims=True), lm.k)


def enumerate_trajectories(env, policy):
    """Every possible episode and its probability under the policy.

    Rewards include the KL penalty at the current policy, exactly as in policy.rollout.
    Returns (list of Trajectory, np.array of probabilities summing to 1).
    """
    kl_coef = env_kl_coef(env)
    trajs, probs = [], []

    def extend(s, tr, prob):
        p = policy.probs(s)
        for a in range(env.n_actions):
            s_next, r, done = env.step(s, a)
            new = Trajectory(tr.states + [s], tr.actions + [a], tr.rewards + [r + kl_penalty(env, s, a, p, kl_coef)],
                             tr.task_rewards + [r], tr.probs + [p])
            if done:
                trajs.append(new)
                probs.append(prob * p[a])
            else:
                extend(s_next, new, prob * p[a])

    extend(env.reset(), Trajectory(), 1.0)
    return trajs, np.array(probs)


def state_index(trajs, key=None) -> dict:
    """Policy table row (key(state), default the state itself) -> row number of the
    flattened parameter vector, for every reachable state."""
    key = key or (lambda s: s)
    index = {}
    for tr in trajs:
        for s in tr.states:
            index.setdefault(key(s), len(index))
    return index


def flatten(grad: dict, index: dict, n_actions: int, key=None) -> np.ndarray:
    """Sparse gradient (state -> row) -> vector of length |rows| * |A|.
    States that share a table row (same key) add up, as in TabularSoftmax.apply."""
    key = key or (lambda s: s)
    v = np.zeros((len(index), n_actions))
    for s, row in grad.items():
        v[index[key(s)]] += row
    return v.ravel()


def objective(env, policy) -> float:
    """J(theta) = E[sum_t r_t], with the KL penalty included (it depends on theta)."""
    trajs, probs = enumerate_trajectories(env, policy)
    return float(probs @ np.array([tr.ret for tr in trajs]))


def finite_difference_gradient(env, policy, index: dict, eps: float = 1e-5) -> np.ndarray:
    """dJ/dtheta(s, a) by central differences, entry by entry.

    It differentiates everything, including log pi inside the KL penalty, and uses
    no policy gradient theorem, so it is an independent reference for the estimators."""
    grad = np.zeros((len(index), env.n_actions))
    for row_key, k in index.items():
        row = policy.theta[row_key]           # the table row itself (shared by aliased states)
        for a in range(env.n_actions):
            old = row[a]
            row[a] = old + eps
            up = objective(env, policy)
            row[a] = old - eps
            down = objective(env, policy)
            row[a] = old
            grad[k, a] = (up - down) / (2 * eps)
    return grad.ravel()


def value_function(env, policy) -> dict:
    """V^pi(s) = sum_a pi(a|s) (r(s, a) + V^pi(s')) by recursion; the poem MDP is acyclic.
    r includes the KL penalty, as in the rewards the estimators see."""
    kl_coef = env_kl_coef(env)
    V = {}

    def value(s):
        if s not in V:
            p = policy.probs(s)
            total = 0.0
            for a in range(env.n_actions):
                s_next, r, done = env.step(s, a)
                total += p[a] * (r + kl_penalty(env, s, a, p, kl_coef) + (0.0 if done else value(s_next)))
            V[s] = total
        return V[s]

    value(env.reset())
    return V


def pair_moments(estimator, trajs, probs, index: dict, n_actions: int, key=None):
    """Exact mean and total variance of a batch estimator with n = 2 episodes.

    Loops over all N^2 ordered pairs (tau_i, tau_j), weighted by p_i p_j:
        mean       = sum_ij p_i p_j g(tau_i, tau_j)
        total var  = sum_ij p_i p_j ||g(tau_i, tau_j)||^2 - ||mean||^2
    `estimator(batch)` returns a sparse gradient dict.
    """
    d = len(index) * n_actions
    mean, second = np.zeros(d), 0.0
    for i, (ti, pi) in enumerate(zip(trajs, probs)):
        for tj, pj in zip(trajs, probs):
            g = flatten(estimator([ti, tj]), index, n_actions, key)
            mean += pi * pj * g
            second += pi * pj * float(g @ g)
    return mean, second - float(mean @ mean)


def aliased_value(trajs, probs, key) -> dict:
    """The best critic that only sees key(state): V(o) = E[return-to-go | key(S_h) = o], averaged
    over all visits under the policy (the limit of a Monte Carlo fit on the naive state).
    Returned as a dict state -> V(key(state)), usable by estimators.td_advantage."""
    total, weight = {}, {}
    for tr, p in zip(trajs, probs):
        to_go = np.cumsum(tr.rewards[::-1])[::-1]
        for s, g in zip(tr.states, to_go):
            total[key(s)] = total.get(key(s), 0.0) + p * g
            weight[key(s)] = weight.get(key(s), 0.0) + p
    return {s: total[key(s)] / weight[key(s)] for tr in trajs for s in tr.states}


# ---- biased variants, for the demonstration in notebook 06 only -----------------
def reinforce_same_batch_baseline(trajs, tau: float) -> dict:
    """REINFORCE with b = average return of the *whole* batch, tau_i included.
    b depends on tau_i, so E[g] = (1 - 1/n) grad J (this baseline is biased)."""
    n = len(trajs)
    R = np.array([tr.ret for tr in trajs])
    grad = {}
    for tr, coef in zip(trajs, R - R.mean()):
        for s, a, p in zip(tr.states, tr.actions, tr.probs):
            add_score(grad, s, p, a, coef / n, tau)
    return grad


def gpomdp_truncated_baseline(trajs, tau: float) -> dict:
    """GPOMDP with leave-one-out b_t, but the sum over t stops at the end of *each* episode:
        g = (1/n) sum_i sum_{t < L_i} (r_t^i - b_t) sum_{h <= t} grad log pi(A_h^i | S_h^i)
    (b_t averaged over the batch padded to the longest episode). Whether step t exists
    depends on the actions before it, so the dropped terms -b_t (t >= L_i) do not
    average to zero and the estimator is biased."""
    n = len(trajs)
    T = max(len(tr) for tr in trajs)
    r = np.zeros((n, T))
    for i, tr in enumerate(trajs):
        r[i, :len(tr)] = tr.rewards
    b = (r.sum(axis=0, keepdims=True) - r) / (n - 1)
    grad = {}
    for i, tr in enumerate(trajs):
        L = len(tr)
        to_go = np.cumsum((r[i, :L] - b[i, :L])[::-1])[::-1]
        for h, (s, a, p) in enumerate(zip(tr.states, tr.actions, tr.probs)):
            add_score(grad, s, p, a, to_go[h] / n, tau)
    return grad
