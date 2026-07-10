"""Russian text primitives shared across the pipeline.

Pure string logic: tokenization, syllable counting, and a corpus-learnable
rhyme key. No ML, no network. These are the low-level equivalents of the
production runtime's ``_TOKEN_RE`` / ``_slug`` helpers in
``worldpgt/cognition/semantic_thought_graph.py`` — small deterministic
functions every other layer reuses so behaviour stays inspectable.
"""

from __future__ import annotations

import re

VOWELS = "аеёиоуыэюя"
_WORD_RE = re.compile(r"[а-яё]+(?:-[а-яё]+)?", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"(?<=[.!?…])(?:[\"»”')\]]*)\s+")

# Closed-class function words carry no imagery signal, exactly like the
# production ``_STOPWORDS`` set — excluded from the concept graph so it is not
# dominated by prepositions and conjunctions.
STOPWORDS = frozenset(
    {
        "и", "а", "но", "да", "или", "то", "не", "ни", "как", "что", "чтоб",
        "чтобы", "бы", "б", "же", "ли", "ль", "уж", "вот", "так", "там",
        "тут", "где", "когда", "пока", "по", "на", "в", "во", "с", "со",
        "к", "ко", "о", "об", "от", "до", "за", "из", "у", "под", "над",
        "при", "про", "для", "без", "через", "меж", "меж", "я", "ты", "он",
        "она", "оно", "они", "мы", "вы", "мне", "мной", "меня", "тебя",
        "тебе", "его", "её", "их", "им", "ему", "ей", "нас", "вас", "себя",
        "мой", "моя", "моё", "мои", "твой", "твоя", "твоё", "свой", "своя",
        "своё", "этот", "эта", "это", "эти", "тот", "та", "те", "весь",
        "вся", "всё", "все", "уже", "ещё", "лишь", "только", "чем", "тем",
        "был", "была", "было", "были", "быть", "есть", "если",
        # Orthographic variants and contracted forms seen in the source texts.
        "ее", "еще", "иль", "кто", "нет", "нам", "моей", "может",
    }
)


def words(text: str) -> list[str]:
    """Lowercased word tokens (Cyrillic, hyphenated words kept whole)."""

    return [m.group(0).lower() for m in _WORD_RE.finditer(text or "")]


def content_words(text: str) -> list[str]:
    """Word tokens minus stopwords — the imagery-bearing vocabulary."""

    return [w for w in words(text) if w not in STOPWORDS and len(w) > 1]


def sentences(text: str) -> list[str]:
    """Split Russian prose into inspectable sentence-sized fragments.

    This is deliberately a conservative surface segmentation rule, not a
    parser.  It keeps dialogue dashes with their utterance and never supplies
    tokens that were not present in the local corpus.
    """

    return [part.strip() for part in _SENTENCE_RE.split(text or "") if words(part)]


def syllables(word_or_line: str) -> int:
    """Syllable count = vowel count. In Russian this is exact, which is why
    meter can be measured deterministically without a pronunciation model."""

    return sum(1 for ch in word_or_line.lower() if ch in VOWELS)


def rhyme_key(word: str) -> str:
    """Clausula-style rhyme key: the suffix from the second-to-last vowel to
    the end of the word. Two words rhyme when their keys match.

    This is a heuristic for stressed-ending (masculine/feminine) rhyme that
    works without a stress dictionary: Russian classical rhyme aligns the
    final stressed vowel and everything after it, and the penultimate-vowel
    window captures that region for the common two-syllable clausula. The key
    is *learned against the corpus* (see ``ingest`` building rhyme groups),
    not hardcoded per word — the same design principle as the production
    phrase graph learning its transitions from data rather than templates.
    """

    w = word.lower().replace("ё", "о")  # ё always stressed -> rhymes as о
    positions = [i for i, ch in enumerate(w) if ch in VOWELS]
    if not positions:
        return w
    start = positions[-2] if len(positions) >= 2 else positions[-1]
    return w[start:]


def last_word(line: str) -> str:
    toks = words(line)
    return toks[-1] if toks else ""


def line_syllables(line: str) -> int:
    return syllables(line)
