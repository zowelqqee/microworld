"""
Quality-aware suppression policy for learned trust transfer.

Provides token quality scoring and suppression policy application on top
of the naive learned-trust suppression candidates.
"""
from __future__ import annotations

KNOWN_NOISY_TOKENS: frozenset[str] = frozenset({
    "oxegen",
    "talbe",
})

TOKEN_ALLOWLIST: frozenset[str] = frozenset({
    "h2o",
    "co2",
    "dna",
    "rna",
    "wood",
    "trees",
    "silica",
    "sand",
    "molten_sand",
    "paper",
    "water",
    "glass",
})

_VOWELS = frozenset("aeiou")


def token_quality_score(token: str) -> float:
    """Return a quality score in [0.0, 1.0]. Lower means more likely noisy.

    Penalties applied in order:
    - empty or whitespace-only → 0.0
    - TOKEN_ALLOWLIST          → 1.0  (scientific/material shortforms)
    - KNOWN_NOISY_TOKENS       → 0.0  (confirmed misspellings)
    - garbage characters       → heavy deduction proportional to garbage ratio
    - excessive underscores    → moderate deduction
    - single alpha-only char   → deduction (too short to be meaningful)
    - very low vowel ratio in long alphabetic token → deduction
    """
    t = token.strip().lower()

    if not t:
        return 0.0

    if t in TOKEN_ALLOWLIST:
        return 1.0

    if t in KNOWN_NOISY_TOKENS:
        return 0.0

    score = 1.0

    # Penalize garbage characters (anything not alphanumeric and not underscore)
    total = len(t)
    clean = sum(1 for c in t if c.isalnum() or c == "_")
    if clean < total:
        garbage_ratio = (total - clean) / total
        score -= garbage_ratio * 0.9

    if score <= 0.0:
        return 0.0

    # Penalize tokens where underscores dominate the content
    underscore_count = t.count("_")
    if underscore_count >= 2 and underscore_count / total > 0.40:
        score -= 0.35

    alpha_chars = [c for c in t if c.isalpha()]

    # Penalize single-character purely-alphabetic tokens
    if len(alpha_chars) == 1 and not any(c.isdigit() for c in t):
        score -= 0.45

    # Penalize very low vowel ratio for longer alphabetic tokens (>= 5 alpha chars)
    if len(alpha_chars) >= 5:
        vowel_count = sum(1 for c in alpha_chars if c in _VOWELS)
        vowel_ratio = vowel_count / len(alpha_chars)
        if vowel_ratio < 0.10:
            score -= 0.50

    return max(0.0, min(1.0, score))


def is_noisy_token(token: str) -> bool:
    """Return True if the token quality score is below 0.5."""
    return token_quality_score(token) < 0.5


def should_suppress_quality_aware(
    row: dict,
    *,
    min_token_quality: float = 0.55,
) -> bool:
    """Return True if the row should be suppressed based on target token quality.

    Only the target is evaluated for suppression: a noisy target means the
    prediction is factually suspect (e.g. "oxegen" is not a real substance).
    A noisy source indicates a typo/canonicalization issue in the node label,
    not a problem with the prediction itself, so it does not drive suppression.

    Source quality is still accessible via token_quality_score for downstream
    normalization/canonicalization use.
    """
    tgt_score = token_quality_score(row.get("target", ""))
    return tgt_score < min_token_quality


def apply_suppression_policy(
    rows: list[dict],
    policy: str,
    min_token_quality: float = 0.55,
) -> list[dict]:
    """Filter suppression candidate rows by the named policy.

    'naive'         → return all rows unchanged.
    'quality_aware' → return only rows where target token quality is low.

    Raises ValueError for unknown policy names.
    """
    if policy == "naive":
        return list(rows)
    if policy == "quality_aware":
        return [
            r for r in rows
            if should_suppress_quality_aware(r, min_token_quality=min_token_quality)
        ]
    raise ValueError(
        f"Unknown suppression policy: {policy!r}. Expected 'naive' or 'quality_aware'."
    )
