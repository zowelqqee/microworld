"""Classify snapshot overlay items against accepted and promoted overlays."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldpgt.self_ingestion.conflict_detector import (
    ExistingOverlayIndex,
    detect_overlay_item,
    detect_raw_claim,
)
from worldpgt.self_ingestion.ingestion_delta import item_key
from worldpgt.self_ingestion.types import (
    REASON_CURRENT_NO_AS_OF,
    REASON_PRIVATE,
    REASON_UNIVERSAL,
    REASON_VOLATILE_NEEDS_REVIEW,
    REASON_WEAK_LINK_PROMOTED,
)
from worldpgt.wiki_snapshot_ingestion.snapshot_quarantine import quarantine_custom, quarantine_from_reason
from worldpgt.wiki_snapshot_ingestion.types import (
    SnapshotConflict,
    SnapshotDuplicate,
    SnapshotOverlayDeltaItem,
    SnapshotQuarantineItem,
)

_SAFE_TYPES = ("overlay_entity", "overlay_definition", "overlay_relation", "overlay_context_link")
_CURRENT_TERMS = ("current", "today", "latest", "stock price", "share price", "market cap", "net worth")
_PRIVATE_TERMS = ("private email", "personal email", "phone number", "home address", "@")
_UNIVERSAL_TERMS = ("all ", "every ", " any ")


def _text_for_item(item: dict[str, Any]) -> str:
    parts = [
        str(item.get("subject", "")),
        str(item.get("predicate", "")),
        str(item.get("object", "")),
        str(item.get("definition", "")),
        str(item.get("evidence_text", "")),
        str(item.get("label", "")),
    ]
    return " ".join(p for p in parts if p)


def _source_title(item: dict[str, Any]) -> str:
    return str(item.get("snapshot_source_title") or item.get("source_page") or "")


def _candidate_id(item: dict[str, Any], idx: int) -> str:
    return str(item.get("snapshot_candidate_id") or f"snapshot-overlay-{idx:05d}")


def _is_safe_new_delta(item: dict[str, Any]) -> bool:
    otype = item.get("overlay_type")
    if otype not in _SAFE_TYPES:
        return False
    if otype == "overlay_context_link":
        return item.get("strength") == "weak" and item.get("trust") == "weak_context_only"
    if otype == "overlay_relation":
        return item.get("risk") in ("low", "medium") and item.get("stability") in ("stable", "semi_stable")
    if otype in ("overlay_entity", "overlay_definition"):
        return item.get("risk") == "low"
    return False


def _extra_safety_reason(item: dict[str, Any]) -> str | None:
    text = _text_for_item(item).lower()
    raw_reason = detect_raw_claim(_text_for_item(item))
    if raw_reason:
        return raw_reason
    if any(term in text for term in _PRIVATE_TERMS):
        return REASON_PRIVATE
    if any(term in text for term in _UNIVERSAL_TERMS):
        return REASON_UNIVERSAL
    if item.get("overlay_type") == "overlay_source_fact":
        return REASON_VOLATILE_NEEDS_REVIEW
    if any(term in text for term in _CURRENT_TERMS) and "as of" not in text and "according to" not in text:
        return REASON_CURRENT_NO_AS_OF
    if item.get("overlay_type") == "overlay_context_link" and item.get("trust") != "weak_context_only":
        return REASON_WEAK_LINK_PROMOTED
    return None


def classify_snapshot_overlay_items(
    overlay_items: list[dict[str, Any]],
    accepted_items: list[dict[str, Any]],
    promoted_items: list[dict[str, Any]],
) -> tuple[list[SnapshotOverlayDeltaItem], list[SnapshotDuplicate], list[SnapshotDuplicate], list[SnapshotConflict], list[SnapshotQuarantineItem], list[str]]:
    accepted_index = ExistingOverlayIndex(accepted_items)
    promoted_index = ExistingOverlayIndex(promoted_items)
    accepted_keys = {item_key(item) for item in accepted_items}
    promoted_keys = {item_key(item) for item in promoted_items}

    delta: list[SnapshotOverlayDeltaItem] = []
    dup_accepted: list[SnapshotDuplicate] = []
    dup_promoted: list[SnapshotDuplicate] = []
    conflicts: list[SnapshotConflict] = []
    quarantine: list[SnapshotQuarantineItem] = []
    tainted_sources: set[str] = set()

    pending_delta: list[SnapshotOverlayDeltaItem] = []
    for idx, item in enumerate(overlay_items):
        cid = _candidate_id(item, idx)
        source_title = _source_title(item)
        key = item_key(item)
        text = _text_for_item(item)

        extra_reason = _extra_safety_reason(item)
        if extra_reason:
            quarantine.append(quarantine_from_reason(cid, source_title, extra_reason, text, item))
            continue

        accepted_reason = detect_overlay_item(item, accepted_index)
        promoted_reason = detect_overlay_item(item, promoted_index)
        if accepted_reason:
            conflict = SnapshotConflict(cid, source_title, "accepted", accepted_reason, item)
            conflicts.append(conflict)
            quarantine.append(quarantine_from_reason(cid, source_title, accepted_reason, text, item))
            tainted_sources.add(source_title)
            continue
        if promoted_reason:
            conflict = SnapshotConflict(cid, source_title, "promoted", promoted_reason, item)
            conflicts.append(conflict)
            quarantine.append(quarantine_from_reason(cid, source_title, promoted_reason, text, item))
            tainted_sources.add(source_title)
            continue
        if key in accepted_keys:
            dup_accepted.append(SnapshotDuplicate(cid, source_title, "accepted", item))
            continue
        if key in promoted_keys:
            dup_promoted.append(SnapshotDuplicate(cid, source_title, "promoted", item))
            continue
        if _is_safe_new_delta(item):
            pending_delta.append(SnapshotOverlayDeltaItem(cid, source_title, item))
        else:
            quarantine.append(
                quarantine_custom(
                    cid,
                    source_title,
                    "quarantine_required",
                    text or item.get("overlay_type", ""),
                    item,
                    risk="medium",
                )
            )

    for item in pending_delta:
        if item.source_doc_title not in tainted_sources:
            delta.append(item)

    return delta, dup_accepted, dup_promoted, conflicts, quarantine, sorted(tainted_sources)


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_delta_artifacts(
    out_dir: str | Path,
    delta: list[SnapshotOverlayDeltaItem],
    dup_accepted: list[SnapshotDuplicate],
    dup_promoted: list[SnapshotDuplicate],
    conflicts: list[SnapshotConflict],
) -> None:
    out = Path(out_dir)
    write_json(out / "snapshot_overlay_delta_proposal.json", [d.overlay_item for d in delta])
    write_json(
        out / "snapshot_ingestion_duplicates.json",
        {
            "accepted": [d.to_dict() for d in dup_accepted],
            "promoted": [d.to_dict() for d in dup_promoted],
        },
    )
    write_json(out / "snapshot_ingestion_conflicts.json", [c.to_dict() for c in conflicts])


def breakdown_conflicts(conflicts: list[SnapshotConflict]) -> dict[str, int]:
    return dict(sorted(Counter(c.reason for c in conflicts).items()))
