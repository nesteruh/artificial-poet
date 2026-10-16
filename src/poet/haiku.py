"""The haiku MDP: writing a 5-7-5 poem word by word.

State  S = (prev, line, syl, exact_so_far)
    prev          previous token (word index, or <s> = V at the start of a line)
    line          index of the current line, 0, 1 or 2
    syl           syllables written so far in the current line
    exact_so_far  1 if every *previous* line had exactly its target number of syllables
Action A = a word index 0..V-1, or V = </s> = end-of-line.

A line ends when the agent plays </s>, or is force-ended when it overflows its
target by more than TOLERANCE syllables. The 3rd line end terminates the poem.
Every word has >= 1 syllable, so a line lasts at most target + TOLERANCE + 1 steps
and the horizon is T = sum_lines (target + TOLERANCE + 1) = 8 + 10 + 8 = 26.

Every reward term below is a function of (S, A) only, so the process is Markov:
    syllable score / exact-line bonus / overflow penalty  -> syl, line, A
    poem completion bonus                                  -> exact_so_far, syl, line, A
    fluency  lambda log P_bigram(A | prev)                 -> prev, A
    repeated-word penalty  1{A = prev}                     -> prev, A
"""
from dataclasses import dataclass, asdict

import numpy as np

TARGETS = (5, 7, 5)
TOLERANCE = 2


@dataclass
class RewardConfig:
    """All reward weights in one place. The defaults are the final design chosen in
    notebook 03 (the initial design was syllable score at line end, fluency 0.1,
    bonuses 2 / 5, overflow 2, no KL). sparse=True replaces everything with
    1{valid 5-7-5} at the last step."""
    syllable_weight: float = 1.0     # -w * |syl - target| when a line ends (or per-word progress)
    exact_line_bonus: float = 3.0    # + when a line ends with exactly `target` syllables
    poem_bonus: float = 10.0         # + when the poem ends and all three lines were exact
    fluency_weight: float = 0.03     # lambda in  lambda * log P_bigram(A | prev), every step
    repeat_penalty: float = 2.0      # - when A == prev (same word twice in a row)
    overflow_penalty: float = 1.0    # - when a line is force-ended for overflowing
    progress: bool = True            # give the syllable score per word (see reward_terms)
    kl_coef: float = 0.5             # eta of the KL penalty to the bigram (added in policy.rollout);
                                     # not called beta: beta is the critic step of the actor-critic
    sparse: bool = False

    def as_dict(self):
        return asdict(self)


class HaikuMDP:
    def __init__(self, lm, cfg: RewardConfig = None, targets=TARGETS, tolerance: int = TOLERANCE):
        """targets / tolerance can be changed for the tiny poem MDP of notebook 06."""
        self.lm = lm
        self.cfg = cfg or RewardConfig()
        self.targets = tuple(targets)
        self.tolerance = tolerance
        self.V = lm.V
        self.EOL = lm.EOS          # </s> doubles as the end-of-line action
        self.BOS = lm.BOS
        self.n_actions = lm.V + 1
        self.horizon = sum(t + self.tolerance + 1 for t in self.targets)
        self.word_syllables = np.append(lm.syllables, 0)   # syllables of each action (</s> = 0)

    # ---- dynamics -------------------------------------------------------------
    def reset(self, rng=None):
        return (self.BOS, 0, 0, 1)

    def overflows(self, line: int, syl: int) -> bool:
        return syl > self.targets[line] + self.tolerance

    def transition(self, s, a):
        """Deterministic dynamics. Returns (next_state or None if terminal, info dict)."""
        prev, line, syl, exact = s
        if a == self.EOL:
            line_end, forced, syl_end = True, False, syl
        else:
            syl_end = syl + int(self.word_syllables[a])
            forced = self.overflows(line, syl_end)
            line_end = forced
        if not line_end:
            return (a, line, syl_end, exact), {"line_end": False}
        line_exact = syl_end == self.targets[line]
        exact_next = int(exact and line_exact)
        info = {"line_end": True, "forced": forced, "line_syllables": syl_end, "line_exact": line_exact}
        if line == len(self.targets) - 1:
            info["poem_valid"] = bool(exact_next)
            return None, info
        return (self.BOS, line + 1, 0, exact_next), info

    def reward_terms(self, s, a) -> dict:
        """Every reward component for taking action a in state s (they sum to r(s, a))."""
        prev, line, syl, exact = s
        _, info = self.transition(s, a)
        cfg = self.cfg
        if cfg.sparse:
            return {"valid_poem": float(info.get("poem_valid", False))}
        terms = {"fluency": cfg.fluency_weight * float(self.lm.logP[prev, a]),
                 "repeat": -cfg.repeat_penalty * float(a == prev and a != self.EOL)}
        if cfg.progress and a != self.EOL:
            # per-word progress  w (|target - syl| - |target - syl'|): summed over a line this is
            # w (target - |target - syl_end|), i.e. the line-end score below plus a constant w * target
            syl_next = syl + int(self.word_syllables[a])
            terms["syllables"] = cfg.syllable_weight * (abs(self.targets[line] - syl) - abs(self.targets[line] - syl_next))
        if info["line_end"] and not cfg.progress:
            terms["syllables"] = -cfg.syllable_weight * abs(info["line_syllables"] - self.targets[line])
        if info["line_end"]:
            terms["exact_line"] = cfg.exact_line_bonus * info["line_exact"]
            terms["overflow"] = -cfg.overflow_penalty * info["forced"]
            terms["poem_bonus"] = cfg.poem_bonus * info.get("poem_valid", False)
        return terms

    def step(self, s, a, rng=None):
        s_next, _ = self.transition(s, a)
        r = sum(self.reward_terms(s, a).values())
        return s_next, r, s_next is None

    def log_prior(self, s, a) -> float:
        """log P_bigram(a | prev): the reference policy for the KL penalty."""
        return float(self.lm.logP[s[0], a])

    # ---- policy initialisation and helpers ------------------------------------
    def bigram_logits(self, tau: float):
        """theta(s, .) = tau * log P_bigram(. | prev), so that pi_theta0 is exactly the bigram."""
        logP = self.lm.logP
        return lambda s: tau * logP[s[0]]

    def uniform_logits(self):
        n = self.n_actions
        return lambda s: np.zeros(n)

    def lines(self, actions) -> list[list[int]]:
        """Split an action sequence into lines of word indices (replaying the dynamics)."""
        out, cur, s = [], [], self.reset()
        for a in actions:
            if a != self.EOL:
                cur.append(a)
            s, info = self.transition(s, a)
            if info["line_end"]:
                out.append(cur)
                cur = []
        return out

    def format_poem(self, actions) -> str:
        """'word word / word word word / word' with syllable counts per line."""
        parts = []
        for words in self.lines(actions):
            n = int(self.word_syllables[words].sum()) if words else 0
            parts.append(f"{' '.join(self.lm.vocab[w] for w in words)} ({n})")
        return " / ".join(parts)

    def encode(self, text: str) -> list[int]:
        """'an old pond / ... / ...' -> action list. Each line ends with </s>, unless the
        dynamics already force-ended it (then the remaining words of that line are dropped)."""
        index = {w: i for i, w in enumerate(self.lm.vocab)}
        actions, s = [], self.reset()
        for line in text.split("/"):
            for a in [index[w] for w in line.split()] + [self.EOL]:
                actions.append(a)
                s, info = self.transition(s, a)
                if info["line_end"]:
                    break
        return actions
