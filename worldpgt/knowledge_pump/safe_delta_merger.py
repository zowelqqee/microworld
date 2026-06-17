"""Merge safe snapshot and Relation Extraction v2 deltas into pump dry-run overlay."""

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


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def overlay_key(item: dict[str, Any]) -> tuple:
    t = item.get("overlay_type", "")
    if t == "overlay_entity":
        return ("entity", _norm(item.get("label", "")))
    if t == "overlay_definition":
        return ("definition", _norm(item.get("subject", "")))
    if t == "overlay_relation":
        return ("relation", _norm(item.get("subject", "")), item.get("predicate", ""), _norm(item.get("object", "")))
    if t == "overlay_context_link":
        return ("context", _norm(item.get("source_page", "")), _norm(item.get("target", "")))
    return ("other", json.dumps(item, sort_keys=True))


def compatible_inverse_key(item: dict[str, Any]) -> tuple | None:
    if item.get("overlay_type") != "overlay_relation":
        return None
    pred = item.get("predicate")
    if pred == "founded":
        return ("relation", _norm(item.get("object", "")), "founded_by", _norm(item.get("subject", "")))
    if pred == "founded_by":
        return ("relation", _norm(item.get("object", "")), "founded", _norm(item.get("subject", "")))
    return None


def relation_candidate_to_overlay(row: dict[str, Any]) -> dict[str, Any] | None:
    if not row.get("safe_for_overlay_delta"):
        return None
    if row.get("confidence") == "low" or row.get("stability") == "volatile":
        return None
    return {
        "overlay_type": "overlay_relation",
        "subject": row.get("subject", ""),
        "predicate": row.get("relation", ""),
        "object": row.get("object", ""),
        "source_page": row.get("source_title", ""),
        "evidence_text": row.get("evidence_sentence", ""),
        "trust": "overlay_candidate",
        "risk": row.get("risk", "medium"),
        "stability": row.get("stability", "semi_stable"),
    }


def _load_json(path: str | Path, default):
    p = Path(path)
    if not p.exists():
        return default
    return json.loads(p.read_text(encoding="utf-8"))


def _item_text(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key, ""))
        for key in (
            "label", "entity_type", "subject", "predicate", "object",
            "definition", "source_page", "target", "evidence_text",
        )
        if item.get(key)
    ).strip()


def _is_admissible_delta(item: dict[str, Any]) -> tuple[bool, str]:
    otype = item.get("overlay_type")
    if otype not in {
        "overlay_entity",
        "overlay_definition",
        "overlay_relation",
        "overlay_context_link",
    }:
        return False, "unsupported_overlay_type"
    if item.get("stability") == "volatile" or item.get("risk") == "high":
        return False, "volatile_or_high_risk"
    if otype == "overlay_context_link":
        if item.get("trust") != "weak_context_only" or item.get("strength") not in ("weak", "", None):
            return False, "weak_link_promoted_as_fact"
        return True, ""
    if otype == "overlay_relation":
        if item.get("risk") not in ("low", "medium", None):
            return False, "relation_risk_not_safe"
        if item.get("stability") not in ("stable", "semi_stable", None):
            return False, "relation_stability_not_safe"
    if otype in ("overlay_entity", "overlay_definition") and item.get("risk") not in ("low", None):
        return False, "entity_or_definition_risk_not_low"
    raw_reason = detect_raw_claim(_item_text(item))
    if raw_reason:
        return False, raw_reason
    return True, ""


def _classify_overlay_items(
    *,
    base_items: list[dict[str, Any]],
    candidate_items: list[dict[str, Any]],
    source_kind: str,
    seen: set[tuple],
) -> dict[str, Any]:
    index = ExistingOverlayIndex(base_items)
    safe: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []

    for idx, raw_item in enumerate(candidate_items):
        item = dict(raw_item)
        item.setdefault("pump_source_kind", source_kind)
        cid = str(item.get("snapshot_candidate_id") or item.get("id") or f"{source_kind}-{idx:05d}")
        source_title = str(item.get("snapshot_source_title") or item.get("source_page") or item.get("source_title") or "")
        ok, reason = _is_admissible_delta(item)
        if not ok:
            bucket = quarantine if reason != "unsupported_overlay_type" else rejected
            bucket.append(
                {
                    "candidate_id": cid,
                    "source_doc_title": source_title,
                    "source_kind": source_kind,
                    "reason": reason,
                    "overlay_item": item,
                }
            )
            continue

        conflict_reason = detect_overlay_item(item, index)
        if conflict_reason:
            row = {
                "candidate_id": cid,
                "source_doc_title": source_title,
                "source_kind": source_kind,
                "reason": conflict_reason,
                "overlay_item": item,
            }
            conflicts.append(row)
            quarantine.append(row)
            continue

        key = overlay_key(item)
        inverse = compatible_inverse_key(item)
        if key in seen or (inverse and inverse in seen):
            duplicates.append(
                {
                    "candidate_id": cid,
                    "source_doc_title": source_title,
                    "source_kind": source_kind,
                    "reason": "duplicate_existing_or_pump_delta",
                    "overlay_item": item,
                }
            )
            continue

        seen.add(key)
        if inverse:
            seen.add(inverse)
        safe.append(item)

    return {
        "safe": safe,
        "duplicates": duplicates,
        "conflicts": conflicts,
        "quarantine": quarantine,
        "rejected": rejected,
    }


def _legacy_candidate_items(snapshot_delta: list[dict[str, Any]], relation_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    relation_delta = [x for x in (relation_candidate_to_overlay(r) for r in relation_rows) if x]
    return list(snapshot_delta) + relation_delta, len(relation_delta)


def _merge_legacy_items(base_items: list[dict[str, Any]], candidate_items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Preserve Knowledge Pump v1 legacy merge semantics for old artifacts."""

    seen = {overlay_key(item) for item in base_items}
    seen |= {k for item in base_items for k in [compatible_inverse_key(item)] if k}
    merged: list[dict[str, Any]] = []
    for item in candidate_items:
        if item.get("overlay_type") == "overlay_context_link" and item.get("trust") != "weak_context_only":
            continue
        if item.get("stability") == "volatile" or item.get("risk") == "high":
            continue
        key = overlay_key(item)
        inverse = compatible_inverse_key(item)
        if key in seen or (inverse and inverse in seen):
            continue
        seen.add(key)
        if inverse:
            seen.add(inverse)
        merged.append(item)
    return merged


def merge_overlay_items(
    base_items: list[dict[str, Any]],
    candidate_items: list[dict[str, Any]],
    *,
    source_kind: str = "legacy",
    seed_delta: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen = {overlay_key(item) for item in base_items}
    seen |= {k for item in base_items for k in [compatible_inverse_key(item)] if k}
    if seed_delta:
        seen |= {overlay_key(item) for item in seed_delta}
        seen |= {k for item in seed_delta for k in [compatible_inverse_key(item)] if k}
    classified = _classify_overlay_items(
        base_items=base_items + list(seed_delta or []),
        candidate_items=candidate_items,
        source_kind=source_kind,
        seen=seen,
    )
    return classified["safe"], classified


def merge_with_fresh_deltas(
    *,
    base_overlay_path: str | Path,
    legacy_snapshot_delta_path: str | Path,
    legacy_relation_candidates_path: str | Path,
    fresh_snapshot_overlay_items: list[dict[str, Any]],
    fresh_relation_rows: list[dict[str, Any]],
    output_overlay_path: str | Path,
    output_delta_path: str | Path,
) -> dict[str, Any]:
    """Build the pump delta from legacy artifacts plus freshly recomputed rows.

    The base snapshot dry-run overlay is read-only. The returned ``pump_delta``
    is the only content added to the separate pump dry-run overlay.
    """

    base = _load_json(base_overlay_path, [])
    legacy_snapshot_delta = _load_json(legacy_snapshot_delta_path, [])
    legacy_relation_rows = _load_json(legacy_relation_candidates_path, [])
    legacy_items, legacy_relation_delta_count = _legacy_candidate_items(
        legacy_snapshot_delta, legacy_relation_rows
    )
    legacy_delta = _merge_legacy_items(base, legacy_items)

    fresh_relation_items: list[dict[str, Any]] = []
    relation_rejected: list[dict[str, Any]] = []
    for idx, row in enumerate(fresh_relation_rows):
        item = relation_candidate_to_overlay(row)
        if item is None:
            relation_rejected.append(
                {
                    "candidate_id": str(row.get("id") or f"fresh-relation-{idx:05d}"),
                    "source_doc_title": str(row.get("source_title") or ""),
                    "source_kind": "fresh_relation",
                    "reason": "relation_candidate_not_safe_for_overlay_delta",
                    "candidate": row,
                }
            )
        else:
            item["pump_relation_candidate_id"] = str(row.get("id") or f"fresh-relation-{idx:05d}")
            fresh_relation_items.append(item)

    fresh_snapshot_delta, fresh_snapshot_classified = merge_overlay_items(
        base,
        fresh_snapshot_overlay_items,
        source_kind="fresh_snapshot",
        seed_delta=legacy_delta,
    )
    fresh_relation_delta, fresh_relation_classified = merge_overlay_items(
        base,
        fresh_relation_items,
        source_kind="fresh_relation",
        seed_delta=legacy_delta + fresh_snapshot_delta,
    )

    pump_delta = legacy_delta + fresh_snapshot_delta + fresh_relation_delta
    dry_run = list(base) + pump_delta

    Path(output_delta_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_delta_path).write_text(
        json.dumps(pump_delta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(output_overlay_path).write_text(
        json.dumps(dry_run, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    fresh_duplicates = fresh_snapshot_classified["duplicates"] + fresh_relation_classified["duplicates"]
    fresh_conflicts = fresh_snapshot_classified["conflicts"] + fresh_relation_classified["conflicts"]
    fresh_quarantine = fresh_snapshot_classified["quarantine"] + fresh_relation_classified["quarantine"]
    fresh_rejected = fresh_snapshot_classified["rejected"] + fresh_relation_classified["rejected"] + relation_rejected
    quarantine_by_reason = Counter(str(row.get("reason", "unknown")) for row in fresh_quarantine + fresh_rejected)

    return {
        "legacy_delta": legacy_delta,
        "fresh_snapshot_delta": fresh_snapshot_delta,
        "fresh_relation_delta": fresh_relation_delta,
        "fresh_safe_delta": fresh_snapshot_delta + fresh_relation_delta,
        "pump_delta": pump_delta,
        "dry_run_overlay": dry_run,
        "fresh_duplicates": fresh_duplicates,
        "fresh_conflicts": fresh_conflicts,
        "fresh_quarantine": fresh_quarantine,
        "fresh_rejected": fresh_rejected,
        "fresh_quarantine_by_reason": dict(sorted(quarantine_by_reason.items())),
        "counts": {
            "safe_snapshot_delta_count": len(legacy_snapshot_delta),
            "safe_relation_v2_delta_count": legacy_relation_delta_count,
            "merged_safe_delta_count": len(legacy_delta),
            "fresh_safe_snapshot_delta_count": len(fresh_snapshot_delta),
            "fresh_safe_relation_delta_count": len(fresh_relation_delta),
            "fresh_merged_safe_delta_count": len(fresh_snapshot_delta) + len(fresh_relation_delta),
            "fresh_duplicates_count": len(fresh_duplicates),
            "fresh_conflicts_count": len(fresh_conflicts),
            "fresh_quarantined_count": len(fresh_quarantine),
            "fresh_rejected_count": len(fresh_rejected),
            "pump_safe_delta_total": len(pump_delta),
            "pump_dry_run_overlay_items_count": len(dry_run),
        },
    }


def merge_safe_deltas(
    base_overlay_path: str | Path,
    snapshot_delta_path: str | Path,
    relation_candidates_path: str | Path,
    output_path: str | Path,
) -> tuple[list[dict], dict[str, int]]:
    base = _load_json(base_overlay_path, [])
    snapshot_delta = _load_json(snapshot_delta_path, [])
    relation_rows = _load_json(relation_candidates_path, [])
    items, relation_delta_count = _legacy_candidate_items(snapshot_delta, relation_rows)
    merged_delta = _merge_legacy_items(base, items)

    dry_run = list(base) + merged_delta
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dry_run, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return merged_delta, {
        "safe_snapshot_delta_count": len(snapshot_delta),
        "safe_relation_v2_delta_count": relation_delta_count,
        "merged_safe_delta_count": len(merged_delta),
        "pump_dry_run_overlay_items_count": len(dry_run),
    }
