"""Running (many) training runs reproducibly, in parallel, with an on-disk cache.

A run is described by a plain dict `spec` (method, step size, seed, reward config, ...).
run_experiment(spec) is a pure function of spec, so results can be cached by spec:
run_many(specs, cache_path) only computes the specs that are not cached yet.
Delete the cache file in results/ to regenerate everything from scratch.
Cache files are gzip-compressed pickles (a .pkl name is kept; plain pickles are still read).
"""
import gzip
import json
import os
import pickle
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from . import bigram, evaluate, haiku, train
from .policy import TabularSoftmax

ROOT = Path(__file__).resolve().parents[2]
BOOK_DIR = ROOT / "data" / "books"
RESULTS_DIR = ROOT / "results"

DEFAULTS = {
    "method": "gpomdp_b",     # reinforce | reinforce_b | gpomdp | gpomdp_b | ac
    "alpha": 1.0,             # actor / policy step size
    "beta_factor": 10.0,      # actor-critic: critic step = min(beta_factor * alpha, 1)
    "batch_size": 10,
    "n_episodes": 50_000,
    "seed": 0,
    "tau": 1.0,
    "init": "bigram",         # bigram | uniform
    "reward": {},             # overrides of haiku.RewardConfig
    "eval_every": 1000,
    "eval_n": 200,
    "n_final_poems": 200,
}
# Optional key, deliberately not in DEFAULTS (so the cache keys of older runs stay valid):
#   "state": "naive"   the policy (and the actor-critic's critic) sees only the previous word

_LM = None


def get_lm():
    """The bigram, built once per process (deterministic, about 1 s)."""
    global _LM
    if _LM is None:
        _LM = bigram.build_bigram(BOOK_DIR)
    return _LM


def full_spec(spec: dict) -> dict:
    return {**DEFAULTS, **spec}


def spec_key(spec: dict) -> str:
    return json.dumps(full_spec(spec), sort_keys=True)


def make_env(spec: dict) -> haiku.HaikuMDP:
    return haiku.HaikuMDP(get_lm(), haiku.RewardConfig(**full_spec(spec)["reward"]))


def previous_word(s):
    """Naive state of notebook 07: only the previous token of the enriched state (prev, line, syl, exact)."""
    return s[0]


def make_policy(env, spec: dict) -> TabularSoftmax:
    spec = full_spec(spec)
    logits = env.bigram_logits(spec["tau"]) if spec["init"] == "bigram" else env.uniform_logits()
    key = previous_word if spec.get("state", "enriched") == "naive" else None
    return TabularSoftmax(logits, spec["tau"], key)


def run_experiment(spec: dict) -> dict:
    """Train one policy and return its learning curves and final sample poems.

    Seeds: the training generator uses `seed`; evaluation uses its own generator
    seeded with (seed, 1), so evaluation poems never come from the training stream.
    """
    spec = full_spec(spec)
    env = make_env(spec)
    policy = make_policy(env, spec)
    rng = np.random.default_rng([spec["seed"], 0])
    evaluator = evaluate.make_evaluator(env, spec["eval_n"], [spec["seed"], 1])
    if spec["method"] == "ac":
        beta = min(spec["beta_factor"] * spec["alpha"], 1.0)
        history, _ = train.train_actor_critic(env, policy, spec["alpha"], beta, spec["n_episodes"],
                                              rng, evaluator, spec["eval_every"])
    else:
        history = train.train_batch_pg(env, policy, spec["method"], spec["alpha"], spec["batch_size"],
                                       spec["n_episodes"], rng, evaluator, spec["eval_every"])
    final = evaluate.sample_poems(env, policy, spec["n_final_poems"], np.random.default_rng([spec["seed"], 2]))
    return {"spec": spec, "history": history,
            "final_metrics": evaluate.poem_metrics(env, final),
            "final_poems": Counter(env.format_poem(tr.actions) for tr in final).most_common(),
            "n_states": len(policy.theta)}


def load_cache(path) -> dict:
    path = Path(path)
    if not path.exists():
        return {}
    data = path.read_bytes()
    if data[:2] == bytes([0x1F, 0x8B]):                 # gzip magic number
        data = gzip.decompress(data)
    return pickle.loads(data)


def run_many(specs: list, cache_path, processes: int = None, verbose: bool = True) -> list:
    """Run every spec (in parallel), reusing results already in `cache_path`.
    Returns the results in the order of `specs`.

    processes=None uses the environment variable POET_PROCESSES if it is set (e.g. to save
    memory), else one worker per CPU. Results do not depend on it: every run is seeded by its spec."""
    if processes is None and os.environ.get("POET_PROCESSES"):
        processes = int(os.environ["POET_PROCESSES"])
    cache_path = Path(cache_path)
    cache = load_cache(cache_path)
    todo = [s for s in {spec_key(s): s for s in specs}.items() if s[0] not in cache]
    if todo:
        if verbose:
            print(f"running {len(todo)} new runs ({len(specs) - len(todo)} cached) ...", flush=True)
        if processes == 1 or len(todo) == 1:
            outs = [run_experiment(s) for _, s in todo]
        else:
            with Pool(processes) as pool:
                outs = pool.map(run_experiment, [s for _, s in todo], chunksize=1)
        for (k, _), out in zip(todo, outs):
            cache[k] = out
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(gzip.compress(pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL), compresslevel=6))
    elif verbose:
        print(f"all {len(specs)} runs loaded from {cache_path.name}")
    return [cache[spec_key(s)] for s in specs]
