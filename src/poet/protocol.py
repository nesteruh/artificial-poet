"""The comparison protocol of notebook 04 (shared with notebooks 05-07).

Budget, step-size grid, seeds, the tuning criterion and the sample-complexity
measure live here, so that every notebook builds exactly the same run specs
(and hence hits the same cache entries in results/).
"""
import numpy as np

from . import stats

METHODS = ["reinforce", "reinforce_b", "gpomdp", "gpomdp_b", "ac"]
ALPHA_GRID = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]   # extended for all methods if a best alpha is at an edge
N_EPISODES = 100_000                             # training episodes per run (the budget)
EVAL_EVERY = 1000
SMOOTH = 5                                       # smoothing window, in evaluations (= 5000 episodes)
PILOT_SEED = 1000
TUNE_SEEDS = [0, 1, 2]
FINAL_SEEDS = list(range(100, 110))              # fresh seeds: the final runs are not the tuning runs


# Reward designs of notebook 05 (overrides of haiku.RewardConfig; {} = the shaped default).
REWARD_DESIGNS = {
    "shaped": {},
    "lines": dict(syllable_weight=0.0, exact_line_bonus=1.0, poem_bonus=1.0, fluency_weight=0.0,
                  repeat_penalty=0.0, overflow_penalty=0.0, progress=False, kl_coef=0.0),
    "sparse": {"sparse": True},
}


def best_alphas(scores: dict, grid: list, methods, reached: dict) -> tuple[dict, dict]:
    """Best alpha per method = highest mean tuning score (ties -> the smaller alpha).

    Returns (best, informative): `informative` only holds the methods that reached the
    target in at least one tuning run. A method that never learns at any alpha has no
    meaningful best value, so it must not trigger a grid extension (see extend_grid).
    """
    best = {m: max(grid, key=lambda a: (np.mean(scores[(m, a)]), -a)) for m in methods}
    informative = {m: a for m, a in best.items() if reached[m]}
    return best, informative


def spec(method: str, alpha: float, seed: int, **extra) -> dict:
    """One run of the comparison. Batch methods use batch size 10; the actor-critic
    uses critic step beta = min(10 alpha, 1) (see experiments.DEFAULTS)."""
    return {"method": method, "alpha": alpha, "seed": seed, "n_episodes": N_EPISODES,
            "eval_every": EVAL_EVERY, **extra}


def smoothed(history: dict, key: str = "return") -> np.ndarray:
    """Trailing moving average of an evaluation curve over SMOOTH evaluations."""
    return stats.moving_average(history["eval_" + key], SMOOTH)


def tuning_score(result: dict) -> float:
    """Tuning criterion: mean evaluation return over the whole budget (area under the
    learning curve / budget). It rewards reaching a high return *early*, and it does not
    depend on the target, which is fixed separately."""
    return float(np.mean(result["history"]["eval_return"]))


def extend_grid(grid: list, best_alphas: dict) -> list:
    """If some method's best alpha is the smallest (largest) grid value, add one more
    value below (above) for all methods, keeping the 1-3-10 spacing."""
    def leading_digit(x):
        return f"{x:.0e}"[0]                         # 0.01 -> '1', 3.0 -> '3'

    grid = sorted(grid)
    new = list(grid)
    if any(a == grid[0] for a in best_alphas.values()):
        lo = grid[0]
        new.insert(0, float(f"{lo * 0.3 if leading_digit(lo) == '1' else lo / 3:.1g}"))   # 0.01 -> 0.003 -> 0.001
    if any(a == grid[-1] for a in best_alphas.values()):
        hi = grid[-1]
        new.append(float(f"{hi * 10 / 3 if leading_digit(hi) == '3' else hi * 3:.1g}"))   # 3 -> 10 -> 30
    return new


def sample_complexity(history: dict, target: float, key: str = "return"):
    """First training episode at which the smoothed evaluation curve reaches `target`,
    or None if it never does within the budget."""
    curve = smoothed(history, key)
    hit = np.nonzero(curve >= target)[0]
    return int(history["eval_episodes"][hit[0]]) if len(hit) else None
