"""Audit-driven bad-pattern mining for generated personal names.

The miner learns small, explicit character n-gram penalties from manual audit
labels.  It does not try to model names globally; it only identifies patterns
that are bad-heavy in the audited generated sample.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PatternMiningConfig:
    min_support: int = 2
    min_bad_count: int = 2
    bad_ratio: float = 0.60


def extract_name_patterns(
    name: str,
    *,
    lengths: tuple[int, ...] = (2, 3, 4),
) -> set[str]:
    """Return boundary-aware character patterns for *name*.

    For each requested length, the result includes plain n-grams plus prefix and
    suffix forms such as ``^ma`` and ``yn$``.
    """
    n = (name or "").strip().lower()
    patterns: set[str] = set()
    for length in lengths:
        if len(n) < length:
            continue
        patterns.add("^" + n[:length])
        patterns.add(n[-length:] + "$")
        for i in range(0, len(n) - length + 1):
            patterns.add(n[i : i + length])
    return patterns


def mine_pattern_counts(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Count good/bad/unclear labels for each unique pattern per row."""
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        name = (row.get("name") or "").strip()
        if not name:
            continue
        label = (row.get("manual_label") or "").strip().lower()
        if label not in {"good", "bad", "unclear"}:
            continue
        for pattern in extract_name_patterns(name):
            bucket = counts.setdefault(pattern, {"good": 0, "bad": 0, "unclear": 0})
            bucket[label] += 1
    return counts


def mine_pattern_trust(
    rows: list[dict],
    *,
    min_support: int = 2,
    min_bad_count: int = 2,
    bad_ratio: float = 0.60,
) -> tuple[dict[str, float], dict[str, dict[str, int]]]:
    """Return learned bad-pattern trust and stats from labelled audit rows.

    ``support`` is computed as good + bad.  Unclear labels are tracked in stats
    but never create a bad penalty by themselves.
    """
    counts = mine_pattern_counts(rows)
    pattern_trust: dict[str, float] = {}
    pattern_stats: dict[str, dict[str, int]] = {}

    for pattern, bucket in counts.items():
        good = bucket["good"]
        bad = bucket["bad"]
        support = good + bad
        if support < min_support:
            continue
        if bad < min_bad_count:
            continue
        ratio = bad / support if support else 0.0
        if ratio < bad_ratio:
            continue
        pattern_trust[pattern] = 0.70 if ratio >= 0.80 else 0.85
        pattern_stats[pattern] = dict(bucket)

    return (
        {k: pattern_trust[k] for k in sorted(pattern_trust)},
        {k: pattern_stats[k] for k in sorted(pattern_stats)},
    )

