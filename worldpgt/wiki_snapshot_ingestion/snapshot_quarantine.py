"""Snapshot ingestion quarantine helpers."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldpgt.self_ingestion.quarantine import make_quarantine_item
from worldpgt.self_ingestion.types import ACTION_HUMAN_REVIEW, REASON_POLICY
from worldpgt.wiki_snapshot_ingestion.types import SnapshotQuarantineItem


def quarantine_from_reason(
    candidate_id: str,
    source_doc_title: str,
    reason: str,
    text: str,
    overlay_item: dict[str, Any],
) -> SnapshotQuarantineItem:
    q = make_quarantine_item(candidate_id, text, reason, source_doc_title)
    return SnapshotQuarantineItem(
        candidate_id=candidate_id,
        source_doc_title=source_doc_title,
        reason=q.reason,
        risk=q.risk,
        suggested_action=q.suggested_action,
        text=q.text,
        overlay_item=overlay_item,
    )


def quarantine_custom(
    candidate_id: str,
    source_doc_title: str,
    reason: str,
    text: str,
    overlay_item: dict[str, Any],
    risk: str = "high",
    suggested_action: str = ACTION_HUMAN_REVIEW,
) -> SnapshotQuarantineItem:
    if reason in REASON_POLICY:
        return quarantine_from_reason(candidate_id, source_doc_title, reason, text, overlay_item)
    return SnapshotQuarantineItem(
        candidate_id=candidate_id,
        source_doc_title=source_doc_title,
        reason=reason,
        risk=risk,
        suggested_action=suggested_action,
        text=text,
        overlay_item=overlay_item,
    )


def write_quarantine(items: list[SnapshotQuarantineItem], path: str | Path) -> dict[str, int]:
    payload = [item.to_dict() for item in items]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return dict(sorted(Counter(item.reason for item in items).items()))

