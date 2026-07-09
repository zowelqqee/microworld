"""Temporal classification helpers for overlay facts.

This layer is deliberately orthogonal to the existing stability field:
stability says how risky/change-prone a claim is for QA policy, while
temporal_class says what kind of time relationship the fact has.
"""

from __future__ import annotations

from typing import Literal

TemporalClass = Literal["historical", "snapshot", "aggregate", "derived"]

HISTORICAL_RELATIONS: frozenset[str] = frozenset(
    {
        "is_a",
        "part_of",
        "founded",
        "founded_by",
        "created_by",
        "created",
        "born_in",
        "developed_by",
        "known_for",
        "develops",
        "produces",
        "publishes",
        "service_of",
        "subsidiary_of",
        "operated_by",
    }
)

SNAPSHOT_RELATIONS: frozenset[str] = frozenset(
    {
        "owned_by",
        "leader_of",
        "net_worth",
        "estimated_net_worth",
        "wealth_rank",
        "ranking",
        "current_price",
        "market_capitalization",
        "current_population",
        "current_office_holder",
    }
)

AGGREGATE_RELATIONS: frozenset[str] = frozenset(
    {
        "employee_count",
        "employees",
        "mission_count",
        "launched_missions",
        "launch_count",
        "revenue",
    }
)

_AGGREGATE_CLAIM_MARKERS = ("aggregate", "cumulative", "count", "employees", "missions")


def classify_temporal_class(
    predicate: str | None,
    stability: str | None = None,
    *,
    overlay_type: str | None = None,
    claim_type: str | None = None,
) -> TemporalClass | None:
    """Return the temporal class for a fact, or None when review is required."""
    pred = (predicate or "").strip()
    ctype = (claim_type or "").strip().lower()

    if overlay_type == "overlay_definition":
        return "historical"
    if overlay_type == "overlay_source_fact":
        if any(marker in ctype for marker in _AGGREGATE_CLAIM_MARKERS):
            return "aggregate"
        return "snapshot"

    if pred in SNAPSHOT_RELATIONS:
        return "snapshot"
    if pred in AGGREGATE_RELATIONS:
        return "aggregate"
    if pred in HISTORICAL_RELATIONS:
        return "historical"

    if (stability or "").strip() == "volatile":
        return "snapshot"
    return None


def temporal_rank(temporal_class: str | None) -> int:
    """Ordering for chain weakening: historical < snapshot < aggregate < derived."""
    return {
        "historical": 0,
        "snapshot": 1,
        "aggregate": 2,
        "derived": 3,
    }.get(temporal_class or "", -1)


def weakest_temporal_class(classes: list[str]) -> str:
    """Return the weakest temporal class present in a traversal chain."""
    if not classes:
        return "historical"
    return max(classes, key=temporal_rank)


def requires_as_of(temporal_class: str | None) -> bool:
    return temporal_class in {"snapshot", "aggregate"}


def temporal_caveat(temporal_class: str | None, as_of_values: list[str]) -> str:
    """Human-facing caveat for temporal facts."""
    dates = sorted({v for v in as_of_values if v})
    if temporal_class == "snapshot":
        if dates:
            return f"This is a snapshot fact as of {', '.join(dates)} and should be rechecked."
        return "This is a snapshot fact with no as_of date and should be rechecked before use."
    if temporal_class == "aggregate":
        if dates:
            return f"This aggregate fact is current as of {', '.join(dates)} and ages quickly."
        return "This aggregate fact has no as_of date and should be rechecked before use."
    return ""
