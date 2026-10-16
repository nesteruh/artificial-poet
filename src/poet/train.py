"""Training loops: batch policy gradient (REINFORCE / GPOMDP family) and actor-critic.

Both loops count *training episodes* (the x-axis of every learning curve) and call
evaluate(policy) every `eval_every` episodes. Evaluation episodes are sampled by
the evaluate function with its own generator and do not count towards the budget.
"""
import numpy as np

from . import estimators
from .policy import TabularSoftmax, add_score, env_kl_coef, kl_penalty, rollout


def _record(history: dict, episodes: int, metrics: dict):
    history.setdefault("episodes", []).append(episodes)
    for k, v in metrics.items():
        history.setdefault(k, []).append(v)


def train_batch_pg(env, policy: TabularSoftmax, method: str, step_size: float, batch_size: int,
                   n_episodes: int, rng: np.random.Generator, evaluate=None, eval_every: int = 1000):
    """Plain gradient ascent  theta <- theta + step_size * g_hat,  g_hat from a batch of n episodes.

    Returns a history dict with lists 'episodes', 'train_return' (mean return of each
    batch, at the episode count after that batch) and one list per evaluation metric
    (at 'eval_episodes').
    """
    history = {"train_episodes": [], "train_return": []}
    evals = {}
    if evaluate is not None:
        _record(evals, 0, evaluate(policy))
    episodes = 0
    while episodes < n_episodes:
        trajs = [rollout(env, policy, rng) for _ in range(batch_size)]
        episodes += batch_size
        grad = estimators.estimate(method, trajs, policy.tau, env.horizon)
        policy.apply(grad, step_size)
        history["train_episodes"].append(episodes)
        history["train_return"].append(float(np.mean([tr.task_ret for tr in trajs])))
        if evaluate is not None and episodes % eval_every == 0:
            _record(evals, episodes, evaluate(policy))
    history.update({("eval_" + k if k != "episodes" else "eval_episodes"): v for k, v in evals.items()})
    return history


def train_actor_critic(env, policy: TabularSoftmax, alpha: float, beta: float, n_episodes: int,
                       rng: np.random.Generator, evaluate=None, eval_every: int = 1000):
    """Simplified Advantage Actor-Critic, episodic and undiscounted.

    At every step t:
        delta_t = r_t + V_w(S_{t+1}) - V_w(S_t)              (V_w(terminal) = 0)
        theta  <- theta + alpha * delta_t * grad log pi(A_t | S_t)
        w      <- w + beta * delta_t * grad_w V_w(S_t)       (tabular: V(S_t) += beta * delta_t)

    Adaptation of the discounted algorithm: gamma = 1 and the 1/(1-gamma) factor of the
    actor step is dropped. That factor comes from normalising the discounted
    state-occupancy measure and is undefined at gamma = 1. Here the episode itself
    plays the role of the occupancy, as in the finite-horizon PG theorem.
    The critic is tabular on the same representation as the policy (policy.key), so a
    naive policy (notebook 07) also gets a naive critic V(previous word).
    Returns (history, V) where V is the learned critic (dict key(state) -> value).
    """
    kl_coef = env_kl_coef(env)
    key = policy.key or (lambda s: s)
    V = {}
    history = {"train_episodes": [], "train_return": []}
    evals = {}
    if evaluate is not None:
        _record(evals, 0, evaluate(policy))
    for episode in range(1, n_episodes + 1):
        s, done, steps, ret = env.reset(rng), False, 0, 0.0
        while not done:
            assert steps < env.horizon, "episode longer than the horizon T"
            a, p = policy.sample(s, rng)
            s_next, r, done = env.step(s, a, rng)
            r_learn = r + kl_penalty(env, s, a, p, kl_coef)
            delta = r_learn + (0.0 if done else V.get(key(s_next), 0.0)) - V.get(key(s), 0.0)
            grad = {}
            add_score(grad, s, p, a, delta, policy.tau)
            policy.apply(grad, alpha)
            V[key(s)] = V.get(key(s), 0.0) + beta * delta
            s, steps, ret = s_next, steps + 1, ret + r
        history["train_episodes"].append(episode)
        history["train_return"].append(ret)
        if evaluate is not None and episode % eval_every == 0:
            _record(evals, episode, evaluate(policy))
    history.update({("eval_" + k if k != "episodes" else "eval_episodes"): v for k, v in evals.items()})
    return history, V
