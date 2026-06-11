"""
Node quality scoring for ConceptNet-derived graph nodes.

Heuristics (all multiplicative, result clamped to [0.0, 1.0]):

  1. Bad-token hard fail  : name or any underscore-token in DEFAULT_BAD_TOKENS → 0.0
  2. Non-ASCII characters : × 0.2  (heavy but not zero — unicode is rare, not impossible)
  3. Length penalty       : len(name) > 25  → × (25 / len)
  4. Token count penalty  : > 3 underscore tokens → × (3 / n_tokens)
  5. High digit ratio     : > 50 % of non-underscore chars are digits → × 0.3

Bad tokens are checked both as the whole name and as individual underscore-split tokens so
that "tu_hermana_en_bolas" is caught via the "bolas" token even though "bolas" alone is
listed rather than the full phrase.

Typo-like names (e.g. "watter", "recieve") pass all checks and receive score 1.0 —
surface misspellings are lightly penalised only if they also trigger a structural
heuristic (e.g. unusual length or digit ratio).
"""
from __future__ import annotations

DEFAULT_BAD_TOKENS: frozenset[str] = frozenset({
    "sister_naked",
    "epic_fail",
    "fail",
    "naked",
    "bolas",
})

_MAX_CLEAN_LEN    = 25   # names longer than this get a length penalty
_MAX_CLEAN_TOKENS = 3    # more than this many _ tokens → penalty


def node_quality(name: str) -> float:
    """Return a quality score in [0.0, 1.0].  1.0 = clean, 0.0 = junk."""
    # 1. Hard fail: exact name or any underscore-token in DEFAULT_BAD_TOKENS
    if name in DEFAULT_BAD_TOKENS:
        return 0.0
    tokens = name.split("_")
    for token in tokens:
        if token in DEFAULT_BAD_TOKENS:
            return 0.0

    score = 1.0

    # 2. Non-ASCII characters: unusual in normalised ConceptNet nodes
    if not name.isascii():
        score *= 0.2

    # 3. Length penalty: long names are often noise or multi-word phrases
    if len(name) > _MAX_CLEAN_LEN:
        score *= _MAX_CLEAN_LEN / len(name)

    # 4. Token count penalty: many underscore-segments suggest over-specific / noisy nodes
    n_tokens = len(tokens)
    if n_tokens > _MAX_CLEAN_TOKENS:
        score *= _MAX_CLEAN_TOKENS / n_tokens

    # 5. High digit ratio: numeric-heavy strings are usually IDs or codes
    alphanum = name.replace("_", "")
    if alphanum:
        digit_ratio = sum(c.isdigit() for c in alphanum) / len(alphanum)
        if digit_ratio > 0.5:
            score *= 0.3

    return max(0.0, min(1.0, score))


def is_low_quality_node(name: str, threshold: float = 0.3) -> bool:
    """Return True if *name* has quality below *threshold*."""
    return node_quality(name) < threshold
