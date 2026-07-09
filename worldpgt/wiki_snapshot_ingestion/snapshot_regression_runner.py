"""Regression runner for snapshot dry-run overlays."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from worldpgt.experiments import (
    run_context_pack_qa_consistency_v1,
    run_cross_page_qa_v1,
    run_entity_qa_v1,
)
from worldpgt.context_pack.types import OVERLAY_PROMOTED
from worldpgt.wiki_snapshot_ingestion.types import SnapshotRegressionResult


def _status_from_summary(summary: dict[str, Any], expected: int | None = None) -> str:
    if expected is not None and summary.get("qa_total") != expected:
        return "failed"
    if summary.get("wrong_count", 0) != 0:
        return "failed"
    if summary.get("quality_flagged", 0) != 0:
        return "failed"
    return "passed"


def run_snapshot_regressions(
    experiments_dir: str | Path,
    dry_run_overlay_path: str | Path,
    out_dir: str | Path,
) -> list[SnapshotRegressionResult]:
    exp = Path(experiments_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: list[SnapshotRegressionResult] = []

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        suites = [
            ("entity_qa_v1", "entity_qa_prompts_v1.csv", 28),
            ("entity_qa_expansion_v1", "entity_qa_expansion_v1.csv", 111),
            ("entity_qa_adversarial_v1", "entity_qa_adversarial_v1.csv", 68),
        ]
        for name, csv_name, expected in suites:
            try:
                summary = run_entity_qa_v1.run(
                    str(exp / csv_name),
                    str(dry_run_overlay_path),
                    str(tmp / f"{name}.csv"),
                    str(tmp / f"{name}.json"),
                )
                results.append(SnapshotRegressionResult(name, _status_from_summary(summary, expected), summary))
            except Exception as exc:
                results.append(SnapshotRegressionResult(name, "failed", {}, str(exc)))

        try:
            summary = run_cross_page_qa_v1.run(
                str(exp / "cross_page_qa_v1.csv"),
                str(dry_run_overlay_path),
                str(tmp / "cross_page_qa_v1.csv"),
                str(tmp / "cross_page_qa_v1.json"),
            )
            results.append(SnapshotRegressionResult("cross_page_qa_v1", _status_from_summary(summary, 71), summary))
        except Exception as exc:
            results.append(SnapshotRegressionResult("cross_page_qa_v1", "failed", {}, str(exc)))

    try:
        original_promoted = run_context_pack_qa_consistency_v1._PROMOTED_OVERLAY
        run_context_pack_qa_consistency_v1._PROMOTED_OVERLAY = Path(dry_run_overlay_path)
        context_out = out / "context_consistency"
        result = run_context_pack_qa_consistency_v1.run_consistency_gate(
            experiments_dir=exp,
            overlay_mode=OVERLAY_PROMOTED,
            out_dir=context_out,
            write=True,
        )
        summary = result["summary"]
        status = "passed" if summary.get("all_critical_passed") else "failed"
        results.append(SnapshotRegressionResult("context_pack_qa_consistency_v1", status, summary))
    except Exception as exc:
        results.append(
            SnapshotRegressionResult(
                "context_pack_qa_consistency_v1",
                "not_run_requires_adapter",
                {},
                str(exc),
            )
        )
    finally:
        if "original_promoted" in locals():
            run_context_pack_qa_consistency_v1._PROMOTED_OVERLAY = original_promoted

    (out / "snapshot_regression_summary.json").write_text(
        json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return results

