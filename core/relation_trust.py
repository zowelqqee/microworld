"""
Relation-level trust priors derived from human audit results.

Each value is the fraction of predictions for that relation type that were
labelled "correct" or "plausible" by a human reviewer.  Relations not in the
table default to UNKNOWN_RELATION_TRUST.

Usage in PatternBasedPredictor:
    final_confidence = base_confidence * hub_factor * relation_trust

Design choices
--------------
- UNKNOWN_RELATION_TRUST = 0.5
  A relation with no audit data is genuinely uncertain; 0.5 (rather than 1.0)
  encodes that uncertainty rather than silently granting maximum trust.
- Trust values are clamped to (0, 1] so multiplying can never flip sign or
  exceed the pre-trust confidence.
"""
from __future__ import annotations

DEFAULT_RELATION_TRUST: dict[str, float] = {
    "made_of":     0.862,
    "part_of":     0.767,
    "is_a":        0.756,
    "at_location": 0.1,
}

UNKNOWN_RELATION_TRUST: float = 0.5


def get_trust(relation_type: str, trust_table: dict[str, float] | None = None) -> float:
    """Return the trust prior for *relation_type*.

    Falls back to UNKNOWN_RELATION_TRUST for relations absent from the table.
    Values are clamped to (0, 1].
    """
    table = DEFAULT_RELATION_TRUST if trust_table is None else trust_table
    raw = table.get(relation_type, UNKNOWN_RELATION_TRUST)
    return max(1e-6, min(1.0, raw))
