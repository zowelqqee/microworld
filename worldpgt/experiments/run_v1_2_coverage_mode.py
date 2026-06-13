"""Run v1.2 coverage mode over trusted Microworld benchmark outputs.

Coverage mode preserves trusted decisions exactly and adds separate candidate
fields for supervised review. It does not run or alter the trusted policy.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter

from worldpgt.continuation.coverage_candidate_generator import (
    candidate_to_json,
    generate_untrusted_candidate,
    trusted_candidate,
)
from worldpgt.experiments.check_semantic_render_quality import check_rows
from worldpgt.experiments.risk_coverage_metrics import summarize_rows


EXTRA_FIELDS = [
    "trusted_decision",
    "trusted_continuation",
    "candidate_continuation",
    "candidate_full_text",
    "candidate_status",
    "candidate_risk",
    "candidate_source",
    "candidate_review_action",
    "candidate_reason",
    "candidate_selected_sense",
    "candidate_validation_status",
    "candidate_trace",
    "candidate_learning_payload",
]


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_plan(path: str) -> dict[str, dict]:
    with open(path, "r", encoding="utf-8") as handle:
        plan = json.load(handle)
    return {proposal["row_ids"][0]: proposal for proposal in plan.get("proposals", [])}


def _write_csv(rows: list[dict], path: str) -> None:
    fieldnames = list(rows[0].keys()) if rows else []
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(data: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _trusted_summary(rows: list[dict]) -> dict:
    trusted_rows = [
        {
            **row,
            "decision": row["trusted_decision"],
            "continuation": row["trusted_continuation"],
            "selected_sense": row.get("selected_sense", ""),
        }
        for row in rows
    ]
    risk = summarize_rows(trusted_rows)
    quality = check_rows(trusted_rows)
    return {
        "trusted_continue_count": risk["continue_count"],
        "trusted_audit_count": risk["audit_count"],
        "trusted_wrong_continue_count": risk["wrong_continue_count"],
        "trusted_semantic_quality_flagged": quality["flagged_count"],
    }


def summarize_coverage_rows(rows: list[dict]) -> dict:
    trusted = _trusted_summary(rows)
    total = len(rows)
    available = sum(bool(row.get("candidate_full_text", "").strip()) for row in rows)
    unavailable = total - available
    by_status = Counter(row.get("candidate_status", "") for row in rows)
    by_risk = Counter(row.get("candidate_risk", "") for row in rows)
    by_source = Counter(row.get("candidate_source", "") for row in rows)
    by_review = Counter(row.get("candidate_review_action", "") for row in rows)
    return {
        "total_rows": total,
        **trusted,
        "candidate_available_count": available,
        "candidate_unavailable_count": unavailable,
        "total_candidate_coverage": round(available / total, 4) if total else 0.0,
        "candidate_by_status": dict(sorted(by_status.items())),
        "candidate_by_risk": dict(sorted(by_risk.items())),
        "candidate_by_source": dict(sorted(by_source.items())),
        "candidate_by_review_action": dict(sorted(by_review.items())),
    }


def _coverage_row(trusted_row: dict, proposal: dict | None) -> dict:
    row = dict(trusted_row)
    trusted_decision = trusted_row.get("decision", "")
    trusted_text = trusted_row.get("continuation", "")
    row["trusted_decision"] = trusted_decision
    row["trusted_continuation"] = trusted_text

    if trusted_decision == "continue":
        candidate = trusted_candidate(
            trusted_row.get("prompt", ""),
            trusted_text,
            trusted_row.get("selected_sense", ""),
        )
    else:
        target_sense = ""
        if proposal is not None:
            target_sense = proposal.get("target_sense") or ""
        if not target_sense:
            target_sense = trusted_row.get("expected_sense", "")
        if proposal is None:
            proposal = {
                "proposal_type": "needs_trace_instrumentation",
                "recommended_action": "needs_instrumentation",
                "risk_level": "medium",
                "evidence": {},
                "proposed_change": {},
            }
        candidate = generate_untrusted_candidate(
            trusted_row.get("prompt", ""),
            trusted_row.get("ambiguous_term", ""),
            target_sense,
            proposal,
        )

    row.update(
        {
            "candidate_continuation": candidate.candidate_continuation,
            "candidate_full_text": candidate.candidate_full_text,
            "candidate_status": candidate.candidate_status,
            "candidate_risk": candidate.candidate_risk,
            "candidate_source": candidate.candidate_source,
            "candidate_review_action": candidate.candidate_review_action,
            "candidate_reason": candidate.candidate_reason,
            "candidate_selected_sense": candidate.candidate_selected_sense,
            "candidate_validation_status": candidate.candidate_validation_status,
            "candidate_trace": " | ".join(candidate.candidate_trace),
            "candidate_learning_payload": candidate_to_json(candidate),
        }
    )
    return row


def run(input_path: str, trusted_output_path: str, audit_plan_path: str, output_csv: str, output_json: str) -> tuple[list[dict], dict]:
    # ``input_path`` is accepted for CLI symmetry and row-count sanity, but the
    # trusted output CSV is the source of trusted behavior in coverage mode.
    prompt_rows = _read_csv(input_path)
    trusted_rows = _read_csv(trusted_output_path)
    if len(prompt_rows) != len(trusted_rows):
        raise ValueError("input prompt row count does not match trusted output row count")

    proposals = _read_plan(audit_plan_path)
    coverage_rows = [_coverage_row(row, proposals.get(row.get("id", ""))) for row in trusted_rows]
    summary = summarize_coverage_rows(coverage_rows)
    _write_csv(coverage_rows, output_csv)
    _write_json(summary, output_json)
    return coverage_rows, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run v1.2 coverage mode without changing trusted outputs.")
    parser.add_argument("--input", required=True, help="Input prompt CSV")
    parser.add_argument("--trusted-output", required=True, help="Trusted benchmark output CSV")
    parser.add_argument("--audit-plan", required=True, help="Audit improvement plan JSON")
    parser.add_argument("--output-csv", required=True, help="Coverage-mode output CSV")
    parser.add_argument("--output-json", required=True, help="Coverage-mode summary JSON")
    args = parser.parse_args()

    rows, summary = run(
        args.input,
        args.trusted_output,
        args.audit_plan,
        args.output_csv,
        args.output_json,
    )
    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
