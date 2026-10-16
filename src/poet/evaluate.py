"""Evaluation on fresh poems (sampled with a separate generator; not part of the training budget)."""
import numpy as np

from .policy import rollout


def sample_poems(env, policy, n: int, rng: np.random.Generator):
    return [rollout(env, policy, rng) for _ in range(n)]


def poem_metrics(env, trajs) -> dict:
    """Summary statistics of a set of sampled poems.

    return          mean task return (shaped reward, without any KL penalty)
    valid           share of poems that are exactly 5-7-5
    distinct_poems  share of distinct poems among the n samples
    distinct_words  share of the vocabulary used anywhere in the n poems (|words used| / V);
                    compare values computed with the same n
    unique_in_poem  mean over poems of (distinct words in the poem / words in the poem)
    fluency         mean log P_bigram(a | prev) per action (higher = more bigram-like)
    syl_error       mean |syllables - target| per line
    overflow        share of lines that were force-ended
    """
    poems, words, logp, errors, forced, valid, uniq = set(), [], [], [], [], [], []
    for tr in trajs:
        poems.add(tuple(tr.actions))
        w = [a for a in tr.actions if a != env.EOL]
        words += w
        uniq.append(len(set(w)) / max(len(w), 1))
        logp += [env.log_prior(s, a) for s, a in zip(tr.states, tr.actions)]
        s = env.reset()
        for a in tr.actions:
            line = s[1]
            s, info = env.transition(s, a)
            if info["line_end"]:
                errors.append(abs(info["line_syllables"] - env.targets[line]))
                forced.append(info["forced"])
                if "poem_valid" in info:
                    valid.append(info["poem_valid"])
    return {"return": float(np.mean([tr.task_ret for tr in trajs])),
            "valid": float(np.mean(valid)),
            "distinct_poems": len(poems) / len(trajs),
            "distinct_words": len(set(words)) / env.V,
            "unique_in_poem": float(np.mean(uniq)),
            "fluency": float(np.mean(logp)),
            "syl_error": float(np.mean(errors)),
            "overflow": float(np.mean(forced))}


def make_evaluator(env, n: int, seed: int):
    """evaluate(policy) -> metrics on n fresh poems. The evaluator owns its own generator,
    so evaluation never consumes the training generator's random numbers."""
    rng = np.random.default_rng(seed)

    def evaluate(policy):
        return poem_metrics(env, sample_poems(env, policy, n, rng))
    return evaluate
