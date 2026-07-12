"""Report builders for Wikipedia Snapshot Self-Ingestion v1."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldpgt.wiki_snapshot_ingestion.types import (
    ReadySnapshotDoc,
    SnapshotConflict,
    SnapshotIngestionReport,
    SnapshotIngestionSummary,
    SnapshotOverlayDeltaItem,
    SnapshotQuarantineItem,
    SnapshotRegressionResult,
)

RECOMMENDED_NEXT_ACTIONS = [
    "inspect selected snapshot docs and provenance",
    "inspect quarantine and conflict reports",
    "review snapshot overlay delta proposal manually",
    "run promotion only after explicit approval",
    "run regression and context consistency after any promotion",
]


def build_summary(
    snapshot_docs_total: int,
    ready_docs_selected: int,
    not_ready_docs_skipped: int,
    doc_status: dict[str, int],
    candidates_total: int,
    candidates_by_type: dict[str, int],
    overlay_items_total: int,
    duplicates_accepted_count: int,
    duplicates_promoted_count: int,
    conflicts_count: int,
    quarantined_count: int,
    rejected_count: int,
    safe_delta_items_count: int,
    dry_run_overlay_items_count: int,
    regressions: list[SnapshotRegressionResult],
) -> SnapshotIngestionSummary:
    run_statuses = [r for r in regressions if r.status != "not_run_requires_adapter"]
    passed = sum(1 for r in run_statuses if r.status == "passed")
    failed = sum(1 for r in run_statuses if r.status == "failed")
    all_critical = failed == 0 and all(
        r.status in ("passed", "not_run_requires_adapter") for r in regressions
    )
    return SnapshotIngestionSummary(
        snapshot_docs_total=snapshot_docs_total,
        ready_docs_selected=ready_docs_selected,
        not_ready_docs_skipped=not_ready_docs_skipped,
        ingestion_docs_attempted=doc_status.get("attempted", 0),
        ingestion_docs_succeeded=doc_status.get("succeeded", 0),
        ingestion_docs_failed=doc_status.get("failed", 0),
        candidates_total=candidates_total,
        candidates_by_type=candidates_by_type,
        overlay_items_total=overlay_items_total,
        duplicates_accepted_count=duplicates_accepted_count,
        duplicates_promoted_count=duplicates_promoted_count,
        conflicts_count=conflicts_count,
        quarantined_count=quarantined_count,
        rejected_count=rejected_count,
        safe_delta_items_count=safe_delta_items_count,
        dry_run_overlay_items_count=dry_run_overlay_items_count,
        regressions_run_count=len(run_statuses),
        regressions_passed_count=passed,
        regressions_failed_count=failed,
        all_critical_passed=all_critical,
    )


def build_report(
    summary: SnapshotIngestionSummary,
    source_artifacts_read: list[str],
    selected_docs: list[ReadySnapshotDoc],
    skipped_docs: list[dict[str, Any]],
    quarantine: list[SnapshotQuarantineItem],
    conflicts: list[SnapshotConflict],
    delta: list[SnapshotOverlayDeltaItem],
    regressions: list[SnapshotRegressionResult],
    warnings: list[str],
    errors: list[dict[str, Any]],
) -> SnapshotIngestionReport:
    q_breakdown = dict(sorted(Counter(q.reason for q in quarantine).items()))
    c_breakdown = dict(sorted(Counter(c.reason for c in conflicts).items()))
    risky = [
        q.to_dict() for q in quarantine
        if q.reason in {
            "volatile_requires_source",
            "current_fact_without_as_of",
            "private_or_sensitive_data",
            "weak_link_promoted_to_fact",
        }
    ][:20]
    return SnapshotIngestionReport(
        summary=summary.to_dict(),
        source_artifacts_read=source_artifacts_read,
        selected_docs=[doc.to_dict() for doc in selected_docs],
        skipped_docs=skipped_docs,
        candidate_type_breakdown=summary.candidates_by_type,
        quarantine_breakdown=q_breakdown,
        conflict_breakdown=c_breakdown,
        safe_delta_examples=[d.to_dict() for d in delta[:20]],
        risky_current_volatile_examples=risky,
        regression_results=[r.to_dict() for r in regressions],
        warnings=warnings,
        errors=errors,
        recommended_next_actions=RECOMMENDED_NEXT_ACTIONS,
    )


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

