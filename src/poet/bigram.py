"""Word bigram language model with add-k smoothing."""
import numpy as np


def sentence_to_pairs(sentence: list[str], word_index: dict[str, int]) -> list[tuple[int, int]]:
    """(prev, next) index pairs of one sentence, including <s> -> first and last -> </s>."""
    V = len(word_index)
    ids = [word_index.get(w) for w in sentence]
    padded = [V] + ids + [V]             # <s>, </s>
    return [(a, b) for a, b in zip(padded[:-1], padded[1:]) if a is not None and b is not None]


def count_bigrams(sentences: list[list[str]], vocab: list[str]) -> np.ndarray:
    """Count matrix C[prev, next] of shape (V+1, V+1)."""
    word_index = {w: i for i, w in enumerate(vocab)}
    V = len(vocab)
    C = np.zeros((V + 1, V + 1))
    for sentence in sentences:
        for a, b in sentence_to_pairs(sentence, word_index):
            C[a, b] += 1
    return C


def add_k_probabilities(C: np.ndarray, k: float) -> np.ndarray:
    """Add-k smoothing:  P(w | u) = (C(u, w) + k) / (sum_w' C(u, w') + k (V+1)).

    Every entry is strictly positive, so log P_bigram is finite everywhere
    (needed for the fluency reward and for the policy initialisation).
    """
    smoothed = C + k
    return smoothed / smoothed.sum(axis=1, keepdims=True)


def heldout_log_likelihood(P: np.ndarray, pairs: np.ndarray) -> float:
    """Average log-probability per bigram  (1/N) sum_n log P[u_n, w_n]  on held-out pairs."""
    return float(np.mean(np.log(P[pairs[:, 0], pairs[:, 1]])))


def perplexity(P: np.ndarray, pairs: np.ndarray) -> float:
    """Perplexity = exp(-average log-likelihood); lower is better."""
    return float(np.exp(-heldout_log_likelihood(P, pairs)))


def split_sentences(sentences: list, heldout_fraction: float, rng: np.random.Generator):
    """Random train / held-out split of a list of sentences."""
    perm = rng.permutation(len(sentences))
    n_held = int(round(heldout_fraction * len(sentences)))
    held = [sentences[i] for i in perm[:n_held]]
    train = [sentences[i] for i in perm[n_held:]]
    return train, held


def all_pairs(sentences: list[list[str]], vocab: list[str]) -> np.ndarray:
    """All in-vocabulary (prev, next) pairs of a list of sentences, as an (N, 2) int array."""
    word_index = {w: i for i, w in enumerate(vocab)}
    pairs = [p for s in sentences for p in sentence_to_pairs(s, word_index)]
    return np.array(pairs, dtype=np.int64).reshape(-1, 2)


def select_k(sentences, vocab, k_grid, rng: np.random.Generator, heldout_fraction=0.1):
    """Choose the smoothing constant k that minimises held-out perplexity.

    Returns (best_k, list of perplexities in the order of k_grid).
    """
    train, held = split_sentences(sentences, heldout_fraction, rng)
    C = count_bigrams(train, vocab)
    pairs = all_pairs(held, vocab)
    ppl = [perplexity(add_k_probabilities(C, k), pairs) for k in k_grid]
    return k_grid[int(np.argmin(ppl))], ppl


def sample_sentence(P: np.ndarray, rng: np.random.Generator, max_len: int = 30) -> list[int]:
    """Ancestral sampling: start from <s>, draw w_{t+1} ~ P(. | w_t) until </s> or max_len.

    Returns the list of word indices (without <s> and </s>).
    """
    V = P.shape[0] - 1
    prev, out = V, []
    while len(out) < max_len:
        nxt = int(rng.choice(V + 1, p=P[prev]))
        if nxt == V:
            break
        out.append(nxt)
        prev = nxt
    return out


class Bigram:
    """A fitted bigram: vocabulary, syllable counts and the smoothed matrix P."""

    def __init__(self, vocab: list[str], syllables: list[int], P: np.ndarray, k: float):
        self.vocab = vocab
        self.syllables = np.asarray(syllables)   # syllables[i] for word i
        self.P = P
        self.logP = np.log(P)
        self.k = k
        self.V = len(vocab)
        self.EOS = self.V    # </s> as an action (column)
        self.BOS = self.V    # <s>  as a context (row)

    def words(self, ids) -> list[str]:
        """Indices -> words (</s> printed as '/')."""
        return [self.vocab[i] if i < self.V else "/" for i in ids]


K_GRID = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def build_bigram(book_dir, vocab_size: int = 500, k_grid=K_GRID, seed: int = 0) -> Bigram:
    """Full pipeline used by every notebook: corpus -> vocabulary -> k by held-out
    perplexity -> add-k bigram fitted on the whole corpus. Deterministic given `seed`."""
    from . import corpus
    sentences = corpus.load_corpus(book_dir)
    cmu = corpus.load_cmu_dict()
    vocab = corpus.build_vocabulary(corpus.count_words(sentences), cmu, vocab_size)
    k, _ = select_k(sentences, vocab, k_grid, np.random.default_rng(seed))
    P = add_k_probabilities(count_bigrams(sentences, vocab), k)
    return Bigram(vocab, [corpus.syllables(w, cmu) for w in vocab], P, k)
