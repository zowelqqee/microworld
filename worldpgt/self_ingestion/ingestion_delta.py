"""Ingestion delta classification for Wikipedia Self-Ingestion v1.

Compares newly produced overlay items against the existing accepted overlay and
classifies each as new / duplicate / conflict / volatile-hold. Never overwrites
existing stable facts. Deterministic. No ML, no network.
"""

from __future__ import annotations

import re

from worldpgt.self_ingestion.conflict_detector import (
    ExistingOverlayIndex,
    detect_overlay_item,
)
from worldpgt.self_ingestion.types import (
    CONFLICT_EXISTING,
    DUPLICATE_EXISTING,
    NEW_CANDIDATE,
    QUARANTINE,
    REASON_CONFLICTS_EXISTING,
    REASON_ENTITY_TYPE_MISMATCH,
    REASON_VOLATILE_NEEDS_REVIEW,
    VOLATILE_REQUIRES_SOURCE,
    ClassifiedItem,
)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def item_key(item: dict) -> tuple:
    t = item.get("overlay_type", "")
    if t == "overlay_entity":
        return ("entity", _norm(item.get("label", "")))
    if t == "overlay_definition":
        return ("definition", _norm(item.get("subject", "")))
    if t == "overlay_relation":
        return ("relation", _norm(item.get("subject", "")), item.get("predicate", ""),
                _norm(item.get("object", "")))
    if t == "overlay_context_link":
        return ("context_link", _norm(item.get("source_page", "")),
                _norm(item.get("target", "")))
    if t == "overlay_source_fact":
        return ("source_fact", _norm(item.get("subject", "")), item.get("predicate", ""))
    return ("unknown", _norm(str(item)))


def classify(
    tagged_items: list[tuple[str, dict]],
    existing_items: list[dict],
) -> tuple[list[ClassifiedItem], dict]:
    """Classify (source_id, overlay_item) pairs against the existing overlay.

    Returns (classified_items, reason_by_item_index) where reason_by_item_index
    maps the classified-item list index to a quarantine/hold reason (or None).
    """
    index = ExistingOverlayIndex(existing_items)
    classified: list[ClassifiedItem] = []
    reasons: dict[int, str] = {}

    for source_id, item in tagged_items:
        key = item_key(item)
        reason = detect_overlay_item(item, index)

        if reason in (REASON_CONFLICTS_EXISTING, REASON_ENTITY_TYPE_MISMATCH):
            cls = CONFLICT_EXISTING
            reasons[len(classified)] = reason
        elif reason == REASON_VOLATILE_NEEDS_REVIEW:
            cls = VOLATILE_REQUIRES_SOURCE
            reasons[len(classified)] = reason
        elif reason is not None:
            cls = QUARANTINE
            reasons[len(classified)] = reason
        elif index.has_key(key):
            cls = DUPLICATE_EXISTING
        else:
            cls = NEW_CANDIDATE

        classified.append(ClassifiedItem(source_id, cls, item))

    return classified, reasons
