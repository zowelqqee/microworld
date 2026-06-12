"""
Quality policy for generated personal names (given names, surnames, or mixed).

A small, inspectable rule set that scores a generated name in [0.0, 1.0] and
explains *why*.  It is intentionally not Anglo-centric: common Russian, Georgian,
Armenian and broader European endings are recognised as *positive signals*, and
unicode (e.g. Cyrillic) names are not penalised for failing ASCII-only vowel
checks.

The policy does *not* require a classic surname ending.  Input datasets may
contain given names, surnames, or a mixture — the rules apply equally to all
personal-name forms.

There are no learned weights here — every deduction is an explicit, named rule.
"""
from __future__ import annotations

MIN_LENGTH = 2
MAX_LENGTH = 20
_VALID_THRESHOLD = 0.5

_PUNCT = frozenset("'-")

# Vowels across the scripts we expect (Latin + common accents + Cyrillic).
_VOWELS = frozenset(
    "aeiouy"
    "àáâäãåāèéêëēìíîïįıòóôöõøōùúûüūÿ"
    "аеёиоуыэюя"
)

# Endings that are well-known surname forms — used as a positive signal to
# relax the consonant-cluster and no-vowel checks.  Their absence is not
# penalised at all.
ALLOWED_ENDINGS: tuple[str, ...] = (
    "shvili", "dze", "ova", "eva", "ina", "sky", "ski", "son", "sen",
    "yan", "ian", "ich", "ov", "ev", "in", "ez",
)


def _is_vowel(ch: str) -> bool:
    return ch.lower() in _VOWELS


def _ends_allowed(name: str) -> bool:
    return any(name.endswith(suf) for suf in ALLOWED_ENDINGS)


def _longest_run(name: str, predicate) -> int:
    """Longest run of consecutive characters satisfying *predicate*."""
    best = run = 0
    for ch in name:
        if predicate(ch):
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def _longest_consonant_run(letters: list[str]) -> int:
    return _longest_run("".join(letters), lambda c: c.isalpha() and not _is_vowel(c))


def _longest_repeat(name: str) -> int:
    """Longest run of the same repeated character."""
    best = run = 0
    prev = None
    for ch in name:
        if ch == prev:
            run += 1
        else:
            run = 1
            prev = ch
        best = max(best, run)
    return best


def _evaluate(name: str) -> tuple[float, list[str]]:
    """Return ``(score, reasons)`` for *name*.

    ``reasons`` lists the rules that fired (penalties); an empty list means the
    name looked clean.
    """
    n = name.strip()
    reasons: list[str] = []

    if not n:
        return 0.0, ["empty name"]

    score = 1.0
    length = len(n)
    letters = [c for c in n if c.isalpha()]
    punct = [c for c in n if c in _PUNCT]
    invalid = [c for c in n if not c.isalpha() and c not in _PUNCT]
    has_nonascii = any(ord(c) > 127 for c in n)
    ends_ok = _ends_allowed(n)

    # Hard: 1-char names are always junk; 2-char names get a soft penalty
    # (they may appear in real datasets but are low-confidence output).
    if length < MIN_LENGTH:
        reasons.append("too short")
        score -= 0.6
    elif length == MIN_LENGTH:
        reasons.append("very short")
        score -= 0.2

    if length > MAX_LENGTH:
        reasons.append("too long")
        score -= 0.5

    if invalid:
        reasons.append("invalid characters")
        score -= 0.5

    if n[0] in _PUNCT or n[-1] in _PUNCT:
        reasons.append("starts or ends with punctuation")
        score -= 0.6

    if len(punct) > 2:
        reasons.append("too much punctuation")
        score -= 0.3

    vowels = [c for c in letters if _is_vowel(c)]
    vowel_ratio = len(vowels) / len(letters) if letters else 0.0

    # No-vowel garbage like "qzxqz".  Skipped for unicode names (different
    # vowel inventory) and for names with a recognised surname ending — the
    # ending acts as a positive signal that the form is intentional.
    if letters and not vowels and not has_nonascii and not ends_ok:
        reasons.append("no vowels")
        score -= 0.6

    if length >= 4 and vowel_ratio > 0.85:
        reasons.append("too many vowels")
        score -= 0.3

    if _longest_repeat(n) >= 3:
        reasons.append("too many repeated characters")
        score -= 0.3

    # Unusual consonant clusters are penalised only for ASCII names without a
    # recognised surname-style ending (the ending is a positive signal here).
    if not has_nonascii and not ends_ok and _longest_consonant_run(letters) >= 4:
        reasons.append("weird consonant cluster")
        score -= 0.4

    return max(0.0, min(1.0, score)), reasons


def surname_quality_score(name: str) -> float:
    """Return a quality score in [0.0, 1.0]; higher is more name-like."""
    return _evaluate(name)[0]


def explain_surname_quality(name: str) -> list[str]:
    """Return the list of quality rules that fired for *name*.

    An empty list means no penalties applied; the returned list always has at
    least one entry for problematic names.
    """
    score, reasons = _evaluate(name)
    if not reasons:
        return ["looks like a plausible name"]
    return reasons


def is_valid_generated_surname(
    name: str,
    *,
    source_names: set[str] | None = None,
    avoid_duplicates: bool = False,
    threshold: float = _VALID_THRESHOLD,
) -> bool:
    """Return True if *name* passes the quality policy.

    Works for given names, surnames, or mixed personal-name datasets.  When
    ``avoid_duplicates`` is set, a name that already exists in ``source_names``
    (compared case-insensitively) is rejected outright.
    """
    if avoid_duplicates and source_names is not None:
        if name.strip().lower() in source_names:
            return False
    return surname_quality_score(name) >= threshold
