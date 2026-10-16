# Artificial Poet

Course project 9 (PG-3) of the Reinforcement Learning course. A word bigram language model is fitted on three Project Gutenberg books: Homer's Iliad, the Odyssey and Beowulf. The bigram, restricted to the 500 most frequent words
in the CMU Pronouncing Dictionary, is the initial policy of an episodic MDP that writes one poem per episode. The state is the previous
word, the line, the syllables written so far, and whether all previous lines were exact. The policy is a tabular softmax initialised at
θ = τ·log P_bigram and fine-tuned by policy gradient to write English haiku (5-7-5 syllables). We compare the sample complexity of five
methods:
- REINFORCE;
- REINFORCE with a leave-one-out average baseline;
- GPOMDP;
- GPOMDP with a time-dependent baseline b_t;
- the simplified advantage actor-critic, adapted to γ = 1.

We also study sparse and shaped rewards, estimator variance (with an exact unbiasedness check by enumerating every trajectory of a tiny poem
MDP), and a naive previous-word state.

## Install and run

Python 3.11, from the project root:

```bash
python -m venv .venv
.venv/Scripts/activate          
pip install -r requirements.txt
```

The notebooks are saved with their outputs, in this order:

| notebook | content |
|---|---|
| `01_data_and_bigram` | corpus, vocabulary, bigram |
| `02_bandit_warmup` | estimator check on a bandit with a known gradient |
| `03_poem_mdp_and_reward` | MDP, reward design, the reward hacks  |
| `04_training_comparison` | the main comparison |
| `05_sparse_vs_shaped_reward` | sparse and shaped rewards |
| `06_variance_analysis` | estimator variance and the exact unbiasedness check |
| `07_naive_vs_enriched_state` | naive and enriched state |

The code is in `src/poet/` and the notebooks import it from there. Every run is seeded and uses `np.random.Generator` objects, so the
notebooks reproduce exactly. Training runs are cached in `results/*.pkl`, keyed by their full specification. 
                                                   
## Results

Budget: 10^5 training episodes per run. The step size is tuned per method on a grid with 3 seeds, then 10 fresh seeds are used for the
final runs (mean ± 95% CI). The targets were fixed after a pilot run, before the final comparison: evaluation return ≥ 30, valid 5-7-5
rate ≥ 0.8. The two reference policies: the untrained bigram has return 4.4 and a valid rate of about 10⁻⁴; a uniform random policy has
return −0.4.

**Main result: two regimes.**
- **With the shaped reward, all five methods tie at their tuned step size.** They reach the targets after 6 000–7 000 episodes (final return about 34, valid rate about 1.0), with
  overlapping confidence intervals.
- **With line-only rewards, variance reduction decides whether a run succeeds at all.** These rewards pay +1 per exact line and +1 for a
  valid poem. REINFORCE reaches the valid-rate target in 4/10 runs, after 30 750 episodes on average over those 4. GPOMDP + b_t reaches
  it in 9/10, after 4 900 ± 260 episodes, about 6× fewer. Every failed run is locked into a near-miss poem.

**Sample complexity: training episodes until the smoothed evaluation curve reaches the target (runs that reached it / 10)**

| method | shaped reward, return ≥ 30 | shaped, valid ≥ 0.8 | line rewards only, valid ≥ 0.8 | sparse reward (1 if valid) | naive state, final return / valid rate |
|---|---|---|---|---|---|
| REINFORCE | 10/10, 6 500 ± 700 (α = 3) | 6 500 ± 700 | 4/10, 30 750 | 0/10 | 19.1 / 0.008 |
| REINFORCE + baseline | 10/10, 7 000 ± 830 (α = 3) | 7 000 ± 950 | 10/10, 9 100 ± 7 700 | 0/10 | 19.7 / 0.000 |
| GPOMDP | 10/10, 6 200 ± 660 (α = 3) | 6 100 ± 860 | 7/10, 13 100 | 0/10 | 19.7 / 0.000 |
| GPOMDP + baseline b_t | 10/10, 6 700 ± 480 (α = 3) | 6 500 ± 770 | 9/10, 4 900 ± 260 | 0/10 | 19.7 / 0.000 |
| Actor-critic | 10/10, 6 000 ± 480 (α = 1) | 6 100 ± 410 | 0/10 | 0/10 | 15.7 / 0.001 |

- **Variance (notebook 06).** It explains where each method's sample-complexity-vs-α curve turns back up, not the minimum of the curve.
  The lower-variance GPOMDP estimators still learn at α = 10, where REINFORCE collapses. The four training estimators are verified to be exactly
  unbiased by enumerating a tiny poem MDP.
- **Sparse reward (notebook 05).** It is learnable only through a lucky first success followed by memorisation (off-protocol α = 100). With
  a reward paid only at the end, GPOMDP and REINFORCE are exactly the same estimator.
- **Naive state (notebook 07).** With the previous word only, the valid rate is provably at most 4/27, and every method converges to one
  5-syllable line written three times.
- **Diversity.** Every method collapses onto one or a few memorised poems. This is expected: maximising expected return admits a
  deterministic optimal policy, and the KL-regularised optimum π ∝ P_bigram·exp(Q/η) is stochastic only where value gaps are comparable to
  η. The KL penalty (η = 0.5) keeps the shaped runs from locking into near-miss poems, but not from memorising.
