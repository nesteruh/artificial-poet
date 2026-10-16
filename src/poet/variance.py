"""Empirical variance of the batch gradient estimators at a fixed policy (notebooks 05 and 06)."""
import copy

import numpy as np

from . import estimators
from .policy import rollout


def _add(acc: dict, g: dict, scale: float = 1.0):
    for s, row in g.items():
        if s in acc:
            acc[s] += scale * row
        else:
            acc[s] = scale * row.copy()


def sq_norm(g: dict) -> float:
    return float(sum(np.dot(row, row) for row in g.values()))


def sq_dist(g: dict, mean: dict) -> float:
    """||g - mean||^2 for sparse gradients (rows missing from g are zero)."""
    total = sq_norm(mean)
    for s, row in g.items():
        m = mean.get(s)
        total += np.dot(row, row) if m is None else np.dot(row - m, row - m) - np.dot(m, m)
    return float(total)


def fit_critic_mc(env, policy, n_episodes: int, rng: np.random.Generator) -> dict:
    """Monte Carlo estimate of V^pi: average return-to-go (learning rewards, KL included) over
    all visits to each state in n_episodes fresh episodes. Unvisited states are absent (-> 0).
    A best-case stand-in for the TD(0) critic of the actor-critic, at a fixed policy."""
    total, count = {}, {}
    for _ in range(n_episodes):
        tr = rollout(env, policy, rng)
        to_go = np.cumsum(tr.rewards[::-1])[::-1]
        for s, g in zip(tr.states, to_go):
            total[s] = total.get(s, 0.0) + g
            count[s] = count.get(s, 0) + 1
    return {s: total[s] / count[s] for s in total}


def estimator_variance(env, policy, methods, n: int, M: int, rng: np.random.Generator,
                       extra: dict = None) -> dict:
    """Draw M independent batches of n episodes at the fixed policy and, for each estimator,
    estimate on the *same* batches:

        mean gradient   g_bar = (1/M) sum_m g_m
        total variance  tr Cov[g] = (1/(M-1)) sum_m ||g_m - g_bar||^2

    The trace of the covariance (summed over all table entries) is the quantity that
    controls the noise of a gradient step: E||g - grad J||^2 = tr Cov[g] for an unbiased g.

    `methods` are names for estimators.estimate; `extra` maps further names to functions
    batch -> sparse gradient (e.g. estimators.td_advantage with a fixed critic).
    A second pass replays the same batches (a copy of rng) to get ||g_m - g_bar||^2 per batch,
    hence a standard error for tr Cov.

    Returns {name: {"mean": g_bar, "total_var", "total_var_se", "grad_sq"}} where
    grad_sq = ||g_bar||^2 - tr Cov / M is an unbiased estimate of ||E g||^2.
    """
    fns = {m: (lambda batch, m=m: estimators.estimate(m, batch, policy.tau, env.horizon)) for m in methods}
    fns.update(extra or {})
    replay = copy.deepcopy(rng)
    sums = {m: {} for m in fns}
    for _ in range(M):
        batch = [rollout(env, policy, rng) for _ in range(n)]
        for m, f in fns.items():
            _add(sums[m], f(batch))
    means = {m: {s: row / M for s, row in sums[m].items()} for m in fns}
    dev = {m: np.empty(M) for m in fns}
    for k in range(M):
        batch = [rollout(env, policy, replay) for _ in range(n)]
        for m, f in fns.items():
            dev[m][k] = sq_dist(f(batch), means[m])
    out = {}
    for m in fns:
        var = dev[m].sum() / (M - 1)
        out[m] = {"mean": means[m], "total_var": var,
                  "total_var_se": float(dev[m].std(ddof=1) / np.sqrt(M)),
                  "grad_sq": sq_norm(means[m]) - var / M}
    return out
