"""Summary and report builders for Wikipedia Snapshot Collector v1."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from worldpgt.wiki_snapshots.types import ManifestRow, ReadinessResult

NEXT_RECOMMENDED_ACTIONS = [
    "inspect snapshots",
    "run self-ingestion against selected local docs only",
    "quarantine",
    "promotion",
    "regression",
    "context consistency",
]


def count_files(path: str | Path, suffix: str) -> int:
    root = Path(path)
    if not root.is_dir():
        return 0
    return len(list(root.glob(f"*{suffix}")))


def build_summary(
    allowlist_total: int,
    requested_limit: int,
    fetched_count: int,
    skipped_count: int,
    rows: list[ManifestRow],
    raw_snapshots_total: int,
    normalized_docs_total: int,
    network_calls: int,
    allow_network: bool,
) -> dict[str, Any]:
    success_count = sum(1 for row in rows if row.fetch_status == "success")
    failed_count = sum(1 for row in rows if row.fetch_status != "success")
    ready_count = sum(1 for row in rows if row.ready_for_self_ingestion)
    return {
        "allowlist_total": allowlist_total,
        "requested_limit": requested_limit,
        "fetched_count": fetched_count,
        "success_count": success_count,
        "failed_count": failed_count,
        "skipped_count": skipped_count,
        "ready_for_self_ingestion_count": ready_count,
        "raw_snapshots_total": raw_snapshots_total,
        "normalized_docs_total": normalized_docs_total,
        "network_calls": network_calls,
        "allow_network": allow_network,
        "auto_ingest": False,
        "auto_promote": False,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "safe_for_general_runtime": False,
    }


def build_report(
    summary: dict[str, Any],
    errors: list[dict[str, Any]],
    warnings: list[str],
    failed_pages: list[str],
    skipped_pages: list[str],
    artifacts_written: list[str],
    readiness: list[ReadinessResult],
) -> dict[str, Any]:
    return {
        "summary": summary,
        "errors": errors,
        "warnings": warnings,
        "failed_pages": failed_pages,
        "skipped_pages": skipped_pages,
        "source_artifacts_written": artifacts_written,
        "readiness": [item.to_dict() for item in readiness],
        "next_recommended_actions": list(NEXT_RECOMMENDED_ACTIONS),
    }


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

