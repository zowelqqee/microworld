"""Audit-reason mining layer.

Diagnostic only. Reads a Microworld controlled-continuation output CSV and
explains *why* each audited row did not continue, using deterministic rule-based
classification over the trace fields the engine already emits (see
``audit_reason_types``). It produces a machine-readable JSON report and an
optional flat CSV. It does not run the engine, change any policy, or alter
generation behavior.

Usage:
    python3 -m worldpgt.experiments.audit_reason_miner \
        --input worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
        --output-json worldpgt/experiments/microworld_continuation_v1_2_audit_reasons.json \
        --output-csv worldpgt/experiments/microworld_continuation_v1_2_audit_reasons.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Iterable, Optional

from worldpgt.experiments.audit_reason_types import (
    ALL_REASONS,
    SUGGESTED_ACTION,
    UNKNOWN_REASON,
    classify,
)

CSV_FIELDS = [
    "row_id",
    "term",
    "expected_sense",
    "selected_sense",
    "confidence",
    "decision",
    "primary_audit_reason",
    "secondary_reasons",
    "suggested_next_action",
    "prompt",
]


def _diagnose_row(row: dict) -> dict:
    classification = classify(row)
    confidence = row.get("confidence", "")
    confidence_val: Optional[float]
    try:
        confidence_val = float(confidence) if str(confidence).strip() else None
    except ValueError:
        confidence_val = None

    return {
        "row_id": row.get("id", ""),
        "prompt": row.get("prompt", ""),
        "term": (row.get("ambiguous_term", "") or "").strip(),
        "expected_sense": (row.get("expected_sense", "") or "").strip() or None,
        "selected_sense": (row.get("selected_sense", "") or "").strip() or None,
        "confidence": confidence_val,
        "decision": row.get("decision", ""),
        "primary_audit_reason": classification.primary_reason,
        "secondary_reasons": classification.secondary_reasons,
        "evidence": classification.evidence.as_dict(),
        "suggested_next_action": classification.suggested_next_action,
    }


def mine_rows(rows: Iterable[dict]) -> dict:
    """Classify all rows and build the structured report. Deterministic."""
    all_rows = list(rows)
    total_rows = len(all_rows)
    continue_count = sum(1 for r in all_rows if r.get("decision") == "continue")
    audit_rows = [r for r in all_rows if r.get("decision") == "audit"]
    audit_count = len(audit_rows)

    diagnostics = [_diagnose_row(r) for r in audit_rows]

    reason_counts = {reason: 0 for reason in ALL_REASONS}
    rows_by_reason: dict[str, list[str]] = {reason: [] for reason in ALL_REASONS}
    for diag in diagnostics:
        reason = diag["primary_audit_reason"]
        reason_counts[reason] += 1
        rows_by_reason[reason].append(diag["row_id"])

    # Drop empty buckets for a compact report, but keep deterministic ordering.
    reason_counts = {r: c for r, c in reason_counts.items() if c > 0}
    rows_by_reason = {r: ids for r, ids in rows_by_reason.items() if ids}
    reason_percentages = {
        reason: round(count / audit_count, 4) if audit_count else 0.0
        for reason, count in reason_counts.items()
    }

    summary = {
        "total_rows": total_rows,
        "continue_count": continue_count,
        "audit_count": audit_count,
        "reason_counts": reason_counts,
        "reason_percentages": reason_percentages,
        "unknown_count": reason_counts.get(UNKNOWN_REASON, 0),
        "rows_by_reason": rows_by_reason,
        "suggested_actions": {r: SUGGESTED_ACTION[r] for r in reason_counts},
    }
    return {"summary": summary, "rows": diagnostics}


def mine_file(input_path: str) -> dict:
    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        return mine_rows(csv.DictReader(handle))


def _write_json(report: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(report: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for diag in report["rows"]:
            writer.writerow(
                {
                    "row_id": diag["row_id"],
                    "term": diag["term"],
                    "expected_sense": diag["expected_sense"] or "",
                    "selected_sense": diag["selected_sense"] or "",
                    "confidence": "" if diag["confidence"] is None else diag["confidence"],
                    "decision": diag["decision"],
                    "primary_audit_reason": diag["primary_audit_reason"],
                    "secondary_reasons": ";".join(diag["secondary_reasons"]),
                    "suggested_next_action": diag["suggested_next_action"],
                    "prompt": diag["prompt"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine deterministic audit reasons from a continuation output CSV."
    )
    parser.add_argument("--input", required=True, help="Input controlled continuation output CSV")
    parser.add_argument("--output-json", required=True, help="Output JSON report path")
    parser.add_argument("--output-csv", required=False, help="Optional output CSV report path")
    args = parser.parse_args()

    report = mine_file(args.input)
    _write_json(report, args.output_json)
    if args.output_csv:
        _write_csv(report, args.output_csv)

    summary = report["summary"]
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
