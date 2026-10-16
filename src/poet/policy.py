"""Tabular softmax policy and episode rollouts."""
from dataclasses import dataclass, field

import numpy as np


class TabularSoftmax:
    """pi_theta(a|s) = exp(theta(s,a)/tau) / sum_b exp(theta(s,b)/tau).

    Rows theta(s, .) are created lazily the first time state s is visited, with
    value init_logits(s). The poem MDP has ~30k states x 501 actions, but only the
    visited rows are ever stored or updated.

    key(s) maps an MDP state to the table row the policy uses (default: the state
    itself). The naive policy of notebook 07 uses key = previous word only.
    """

    def __init__(self, init_logits, tau: float, key=None):
        self.init_logits = init_logits   # function: state -> np.ndarray of logits
        self.tau = tau
        self.key = key
        self.theta = {}                  # key(state) -> np.ndarray (one row of the table)

    def row(self, s) -> np.ndarray:
        k = s if self.key is None else self.key(s)
        if k not in self.theta:
            self.theta[k] = np.array(self.init_logits(s), dtype=float)
        return self.theta[k]

    def probs(self, s) -> np.ndarray:
        """Softmax of theta(s, .)/tau (max subtracted for numerical stability)."""
        z = self.row(s) / self.tau
        e = np.exp(z - z.max())
        return e / e.sum()

    def sample(self, s, rng: np.random.Generator):
        """Draw A ~ pi(.|s). Returns (action, probability vector)."""
        p = self.probs(s)
        a = int(np.searchsorted(np.cumsum(p), rng.random() * p.sum(), side="right"))
        return min(a, len(p) - 1), p

    def apply(self, grad: dict, step: float):
        """Gradient ascent step  theta <- theta + step * grad  (grad is sparse: state -> row)."""
        for s, g in grad.items():
            self.row(s)[:] += step * g

    def copy(self) -> "TabularSoftmax":
        new = TabularSoftmax(self.init_logits, self.tau, self.key)
        new.theta = {s: r.copy() for s, r in self.theta.items()}
        return new


def add_score(grad: dict, s, p: np.ndarray, a: int, coef: float, tau: float):
    """grad[s] += coef * grad_theta log pi(a|s).

    For the tabular softmax only row s is affected:
        d log pi(a|s) / d theta(s, b) = (1{b = a} - pi(b|s)) / tau.
    """
    g = grad.get(s)
    if g is None:
        g = grad[s] = np.zeros_like(p)
    g -= (coef / tau) * p
    g[a] += coef / tau


@dataclass
class Trajectory:
    states: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    rewards: list = field(default_factory=list)       # reward used for learning (incl. KL penalty)
    task_rewards: list = field(default_factory=list)  # environment reward only (for evaluation)
    probs: list = field(default_factory=list)    # pi(.|S_t) at sampling time (for the score)

    def __len__(self):
        return len(self.actions)

    @property
    def ret(self) -> float:
        """Return R(tau) = sum_t r_t (undiscounted, gamma = 1)."""
        return float(sum(self.rewards))

    @property
    def task_ret(self) -> float:
        return float(sum(self.task_rewards))


def kl_penalty(env, s, a, p: np.ndarray, kl_coef: float) -> float:
    """Per-step KL penalty to the bigram:  -eta (log pi_theta(a|s) - log P_bigram(a|s)).

    Its expectation over a ~ pi is -eta KL(pi(.|s) || P_bigram(.|prev)). It is added
    to the reward and treated as a constant when computing the gradient (we do not
    differentiate through log pi inside the reward).
    """
    if kl_coef == 0.0:
        return 0.0
    return -kl_coef * (float(np.log(p[a])) - env.log_prior(s, a))


def env_kl_coef(env) -> float:
    """The KL coefficient eta from the env's reward config (0 if it has none)."""
    cfg = getattr(env, "cfg", None)
    return 0.0 if cfg is None or cfg.sparse else cfg.kl_coef


def rollout(env, policy: TabularSoftmax, rng: np.random.Generator) -> Trajectory:
    """Play one episode S_0, A_0, r_0, S_1, ... until termination.

    env must provide reset(rng) -> s, step(s, a, rng) -> (s', r, done) and horizon
    (the maximum episode length T; reaching it would be a modelling bug).
    """
    kl_coef = env_kl_coef(env)
    traj = Trajectory()
    s, done = env.reset(rng), False
    while not done:
        assert len(traj) < env.horizon, "episode longer than the horizon T"
        a, p = policy.sample(s, rng)
        s_next, r, done = env.step(s, a, rng)
        traj.states.append(s); traj.actions.append(a)
        traj.rewards.append(r + kl_penalty(env, s, a, p, kl_coef))
        traj.task_rewards.append(r); traj.probs.append(p)
        s = s_next
    return traj
