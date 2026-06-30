"""Precision gate specifically for schema_induced overlay items.

Wraps (and adds to) the existing precision firewall logic:

1. Generated facts (promotion_status == "generated") require
   schema_source_doc_count >= min_source_docs_generated (default 2).
2. Promoted facts pass through with default overlay gate rules.
3. Items missing subject or object/definition are rejected.
4. Dedup against Path-A (extraction_yield_v2) items:
   - same (subject, predicate, object) key → keep higher-stability version.
   - new from schema_induction → add with pump_source_kind="schema_induced".

Returns a dict mirroring apply_precision_firewall return shape so it can be
fed into the existing pipeline reporting.
"""

from __future__ import annotations

from typing import Any

_STABILITY_RANK = {"stable": 3, "semi_stable": 2, "volatile": 1}

_MIN_SOURCE_DOCS_GENERATED = 2  # configurable via argument


def _item_key(item: dict[str, Any]) -> tuple:
    otype = item.get("overlay_type", "")
    if otype == "overlay_relation":
        return (
            "relation",
            (item.get("subject") or "").strip().lower(),
            item.get("predicate", ""),
            (item.get("object") or "").strip().lower(),
        )
    if otype == "overlay_definition":
        return ("definition", (item.get("subject") or "").strip().lower())
    return ("other", str(item))


def _stability_rank(item: dict[str, Any]) -> int:
    return _STABILITY_RANK.get(item.get("stability", ""), 0)


def dedup_and_merge(
    path_a_items: list[dict[str, Any]],
    schema_items: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge schema_induction items with Path-A (extraction_yield_v2) items.

    - Duplicate (same key) → keep the version with higher stability_rank;
      if equal, prefer Path-A (existing wins).
    - New from schema → include with pump_source_kind preserved.

    Returns a dict with keys matching apply_precision_firewall for easy
    integration into the pump runner report.
    """
    # Build index from Path-A.
    path_a_by_key: dict[tuple, dict[str, Any]] = {}
    for item in path_a_items:
        k = _item_key(item)
        path_a_by_key[k] = item

    accepted: list[dict[str, Any]] = list(path_a_items)  # Path-A always included
    deduped: list[dict[str, Any]] = []
    upgraded: list[dict[str, Any]] = []
    new_induced: list[dict[str, Any]] = []

    for item in schema_items:
        k = _item_key(item)
        existing = path_a_by_key.get(k)
        if existing is not None:
            # Already covered by Path-A. Keep higher stability.
            if _stability_rank(item) > _stability_rank(existing):
                # Schema version is more stable; swap it in the accepted list.
                accepted = [i for i in accepted if _item_key(i) != k]
                accepted.append(item)
                upgraded.append(item)
            else:
                deduped.append(item)
        else:
            # Genuinely new from schema_induction.
            path_a_by_key[k] = item
            accepted.append(item)
            new_induced.append(item)

    return {
        "accepted": accepted,
        "deduped": deduped,
        "upgraded": upgraded,
        "new_induced": new_induced,
        "rejected": [],
        "quarantine": [],
    }


def apply_induced_precision_gate(
    schema_items: list[dict[str, Any]],
    *,
    min_source_docs_generated: int = _MIN_SOURCE_DOCS_GENERATED,
) -> dict[str, Any]:
    """Apply extra gate rules to schema_induced items before the main firewall.

    Returns accepted/rejected/quarantine lists (same shape as
    apply_precision_firewall).
    """
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    rejection_by_reason: dict[str, int] = {}

    def _reject(item: dict, reason: str) -> None:
        rejected.append({**item, "_gate_reason": reason})
        rejection_by_reason[reason] = rejection_by_reason.get(reason, 0) + 1

    def _quarantine(item: dict, reason: str) -> None:
        quarantine.append({**item, "_gate_reason": reason})
        rejection_by_reason[reason] = rejection_by_reason.get(reason, 0) + 1

    for item in schema_items:
        # Basic structural checks.
        subject = (item.get("subject") or "").strip()
        if not subject:
            _reject(item, "empty_subject")
            continue

        otype = item.get("overlay_type", "")
        if otype == "overlay_relation":
            obj = (item.get("object") or "").strip()
            if not obj:
                _reject(item, "empty_object")
                continue
            # Subject must not equal object.
            if subject.lower() == obj.lower():
                _reject(item, "self_relation")
                continue

        if otype == "overlay_definition":
            defn = (item.get("definition") or "").strip()
            if not defn:
                _reject(item, "empty_definition")
                continue

        # Generated-family stricter source-diversity gate.
        status = item.get("schema_promotion_status", "promoted")
        if status == "generated":
            src_count = int(item.get("schema_source_doc_count") or 0)
            if src_count < min_source_docs_generated:
                _quarantine(
                    item,
                    f"generated_family_insufficient_sources:{src_count}<{min_source_docs_generated}",
                )
                continue

        accepted.append(item)

    relation_before = sum(
        1 for i in schema_items if i.get("overlay_type") == "overlay_relation"
    )
    relation_after = sum(
        1 for i in accepted if i.get("overlay_type") == "overlay_relation"
    )

    return {
        "accepted": accepted,
        "rejected": rejected,
        "quarantine": quarantine,
        "property_candidates": [],
        "rejection_by_reason": rejection_by_reason,
        "quarantine_by_reason": {},
        "relation_before_count": relation_before,
        "relation_after_count": relation_after,
        "definition_before_count": len(schema_items) - relation_before,
        "definition_after_count": len(accepted) - relation_after,
    }
