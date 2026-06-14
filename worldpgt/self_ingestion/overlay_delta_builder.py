"""Overlay delta builder for Wikipedia Self-Ingestion v1.

Selects only safe, new, stable/semi-stable candidates for the overlay delta
proposal. Excludes everything from a tainted (conflicting) source, all volatile
source-qualified facts (held for human review), duplicates, conflicts, and any
quarantined item. Produces a delta that is additive and benchmark-neutral.

Never includes: current/live facts as stable, private data, inverted relations,
unsupported universals, weak-link-as-proof claims, or conflicts with existing
stable facts. No ML, no network.
"""

from __future__ import annotations

from worldpgt.self_ingestion.types import (
    CONFLICT_EXISTING,
    NEW_CANDIDATE,
    ClassifiedItem,
)

# Overlay types that are safe to auto-accept when classified new_candidate.
_SAFE_TYPES = ("overlay_entity", "overlay_definition", "overlay_relation",
               "overlay_context_link")
_ALLOWED_RELATION_STABILITY = ("stable", "semi_stable")
_ALLOWED_RISK = ("low", "medium")


def build_delta(classified: list[ClassifiedItem]) -> tuple[list[dict], list[str]]:
    """Return (delta overlay items, tainted_source_ids).

    A source is *tainted* if any of its items is a conflict with the existing
    overlay; all of that source's items are then excluded from the delta.
    """
    tainted: set[str] = {
        c.source_id for c in classified if c.classification == CONFLICT_EXISTING
    }

    delta: list[dict] = []
    for c in classified:
        if c.source_id in tainted:
            continue
        if c.classification != NEW_CANDIDATE:
            continue
        item = c.overlay_item
        otype = item.get("overlay_type", "")
        if otype not in _SAFE_TYPES:
            continue
        if otype == "overlay_context_link":
            if item.get("strength") != "weak" or item.get("trust") != "weak_context_only":
                continue
        if otype == "overlay_relation":
            if item.get("stability") not in _ALLOWED_RELATION_STABILITY:
                continue
            if item.get("risk") not in _ALLOWED_RISK:
                continue
        if otype in ("overlay_entity", "overlay_definition"):
            if item.get("risk") != "low":
                continue
        delta.append(item)

    return delta, sorted(tainted)
