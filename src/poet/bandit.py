"""Repeated 2-armed bandit: a tiny episodic problem with a closed-form policy gradient.

Each episode pulls an arm up to H times. Pulling arm a gives reward mu[a] + noise,
with noise ~ N(0, sigma^2). After every pull the episode stops with probability q
(independently of the action), so episodes have variable length 1..H, like poems.
The policy is a stateless softmax (a single state, 0).
"""
import numpy as np


class RepeatedBandit:
    def __init__(self, mu, sigma: float = 0.5, H: int = 10, q: float = 0.2):
        self.mu = np.asarray(mu, dtype=float)
        self.sigma, self.H, self.q = sigma, H, q
        self.horizon = H
        self.n_actions = len(self.mu)

    def reset(self, rng):
        self.t = 0      # the step counter is kept inside the env: the policy sees one state, 0
        return 0

    def step(self, s, a, rng):
        r = self.mu[a] + self.sigma * rng.standard_normal()
        self.t += 1
        done = (self.t == self.H) or (rng.random() < self.q)
        return 0, float(r), done

    def expected_length(self) -> float:
        """E[number of pulls] = sum_{t=0}^{H-1} (1-q)^t."""
        return float(sum((1 - self.q) ** t for t in range(self.H)))

    def J(self, pi: np.ndarray) -> float:
        """Expected return J = E[length] * sum_a pi(a) mu[a]."""
        return self.expected_length() * float(pi @ self.mu)

    def exact_gradient(self, pi: np.ndarray, tau: float) -> np.ndarray:
        """grad_theta J for pi = softmax(theta / tau):

            dJ/dtheta(a) = E[length] * pi(a) (mu[a] - pi . mu) / tau
        """
        return self.expected_length() * pi * (self.mu - pi @ self.mu) / tau
