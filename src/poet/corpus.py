"""Corpus loading, tokenisation, syllable counting and vocabulary selection.

Pipeline (used by notebook 01 and by everything downstream):

    text      = load_book(path)                    # Gutenberg body -> the poem only, cleaned
    sentences = tokenize_sentences(text)           # list of lists of lowercase words
    counts    = count_words(sentences)
    vocab     = build_vocabulary(counts, cmu, 500) # most frequent words with a pronunciation
"""
import re
import unicodedata
from collections import Counter
from pathlib import Path

import cmudict

# Single-letter tokens are almost always debris from elisions ("o'er" -> "o", "er")
# or footnote letters, except for these three real words.
SINGLE_LETTER_WORDS = {"a", "i", "o"}

# Where the poem itself starts and ends in each of our books (matched on the Gutenberg
# "Title:" line). Everything outside the span is editorial apparatus: introductions,
# prefaces, glossaries, Buckley's concluding note and footnotes, Hall's addenda.
# Inside the span, "arguments" drops the Iliad's prose ARGUMENT summary at the head of
# every book, and "indented_notes" drops Hall's Beowulf footnotes (lines indented by
# exactly 4 spaces; verse lines are indented by 5 or more).
BOOKS = {
    "The Iliad": {"start": "THE ILIAD.", "end": "CONCLUDING NOTE.", "arguments": True},
    "The Odyssey": {"start": "THE ODYSSEY", "end": "FOOTNOTES:"},
    "Beowulf: An Anglo-Saxon Epic Poem": {"start": "BEOWULF.", "end": "ADDENDA.", "indented_notes": True},
}


def load_gutenberg_text(path) -> str:
    """Return the body of a Project Gutenberg .txt file (between START and END markers)."""
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    if "*** START" not in s or "*** END" not in s:
        raise ValueError(f"{path}: Project Gutenberg START/END markers not found")
    start = s.find("\n", s.find("*** START"))
    end = s.find("*** END")
    return s[start + 1:end]


def gutenberg_title(path) -> str:
    """The value of the "Title:" line in the Gutenberg header ("" if there is none)."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("Title:"):
                return line[len("Title:"):].strip()
            if line.startswith("*** START"):
                break
    return ""


def _poem_span(lines: list[str], start: str, end: str, title: str) -> tuple[int, int]:
    """Line indices of the poem: the first `end` marker line, and the last `start` marker
    line before it (the table of contents repeats the start marker earlier on)."""
    ends = [i for i, ln in enumerate(lines) if ln.rstrip() == end]
    if not ends:
        raise ValueError(f"{title}: marker line {end!r} not found")
    starts = [i for i, ln in enumerate(lines[:ends[0]]) if ln.rstrip() == start]
    if not starts:
        raise ValueError(f"{title}: marker line {start!r} not found before {end!r}")
    return starts[-1] + 1, ends[0]


def drop_arguments(lines: list[str]) -> list[str]:
    """Remove each 'ARGUMENT.' heading together with the title block and the prose
    paragraph that follow it (the Iliad's editorial summaries)."""
    out, i = [], 0
    while i < len(lines):
        if re.fullmatch(r"ARGUMENT\.?(\[\d+\])?", lines[i].strip()):
            i += 1
            for _ in range(2):                       # title block, then the summary paragraph
                while i < len(lines) and not lines[i].strip():
                    i += 1
                while i < len(lines) and lines[i].strip():
                    i += 1
        else:
            out.append(lines[i])
            i += 1
    return out


def extract_poem(text: str, title: str) -> str:
    """Keep only the poem itself (see BOOKS). Books not listed in BOOKS are returned unchanged."""
    rules = BOOKS.get(title)
    if rules is None:
        return text
    lines = text.splitlines()
    first, last = _poem_span(lines, rules["start"], rules["end"], title)
    lines = lines[first:last]
    if rules.get("arguments"):
        lines = drop_arguments(lines)
    if rules.get("indented_notes"):
        lines = [ln for ln in lines if not re.match(r" {4}\S", ln)]
    return "\n".join(lines)


def load_book(path) -> str:
    """Gutenberg file -> cleaned text of the poem only."""
    return clean_text(extract_poem(load_gutenberg_text(path), gutenberg_title(path)))


def clean_text(text: str) -> str:
    """Remove the parts of a Gutenberg book that are not running text.

    - bracketed material: footnote markers "[78]", page numbers "[xii]",
      "[Illustration: ...]" captions;
    - braced material: Hall's marginal glosses in Beowulf, "{Scyld dies at the hour ...}";
    - lines written entirely in capitals (chapter headings such as "BOOK IV.");
    - typographic quotes are normalised to ASCII so that "king’s" == "king's",
      and ligatures / accents are removed (see strip_accents).
    """
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"\{[^}]*\}", " ", text)
    lines = [ln for ln in text.splitlines() if not (ln.strip() and ln.upper() == ln)]
    text = "\n".join(lines)
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return strip_accents(text)


def strip_accents(text: str) -> str:
    """'phœbus' -> 'phoebus', 'belovèd' -> 'beloved', 'Chryseïs' -> 'Chryseis'.
    Without this the tokenizer would split 'phœbus' into 'ph' + 'bus'."""
    for lig, rep in {"œ": "oe", "Œ": "Oe", "æ": "ae", "Æ": "Ae"}.items():
        text = text.replace(lig, rep)
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def tokenize_sentences(text: str) -> list[list[str]]:
    """Split text into sentences, and sentences into lowercase words.

    Sentence boundaries are . ? ! ; : (verse translations use ; and : as full stops,
    and line breaks inside a sentence are ignored). Leading/trailing apostrophes
    are stripped because they are usually single quotation marks.
    """
    sentences = re.split(r"[.?!;:]+", text.lower())
    result = []
    for sentence in sentences:
        words = [w.strip("'") for w in re.findall(r"[a-z']+", sentence)]
        words = [w for w in words if len(w) > 1 or w in SINGLE_LETTER_WORDS]
        if words:
            result.append(words)
    return result


def load_corpus(book_dir) -> list[list[str]]:
    """Load, clean and tokenise every .txt book in `book_dir` (sorted by name, so the
    result is deterministic)."""
    sentences = []
    for path in sorted(Path(book_dir).glob("*.txt")):
        sentences += tokenize_sentences(load_book(path))
    return sentences


def count_words(sentences: list[list[str]]) -> Counter:
    """Word frequencies over all sentences."""
    cnt = Counter()
    for words in sentences:
        cnt.update(words)
    return cnt


def load_cmu_dict() -> dict[str, list[list[str]]]:
    """The CMU Pronouncing Dictionary: word -> list of pronunciations (phoneme lists)."""
    return cmudict.dict()


def count_syllables(phonemes: list[str]) -> int:
    """Number of syllables of one pronunciation = number of vowel phonemes,
    i.e. phonemes that end with a stress digit (0, 1 or 2), e.g.
    'water' = W AO1 T ER0 -> 2."""
    return sum(ph[-1].isdigit() for ph in phonemes)


def syllables(word: str, cmu: dict) -> int:
    """Syllables of `word`, using its first CMU pronunciation."""
    return count_syllables(cmu[word][0])


def build_vocabulary(counts: Counter, cmu: dict, size: int) -> list[str]:
    """The `size` most frequent corpus words that are in CMUdict and have >= 1 syllable.

    The >= 1 syllable condition guarantees that every word action advances the
    current line, so episodes always terminate (see notebook 03).
    Ties in frequency are broken alphabetically so the vocabulary is deterministic.
    """
    vocab = []
    for word, _ in sorted(counts.items(), key=lambda wc: (-wc[1], wc[0])):
        if len(vocab) == size:
            break
        if word in cmu and syllables(word, cmu) >= 1:
            vocab.append(word)
    return vocab


def coverage(counts: Counter, vocab: list[str]) -> float:
    """Fraction of corpus tokens that belong to the vocabulary."""
    return sum(counts[w] for w in vocab) / counts.total()
