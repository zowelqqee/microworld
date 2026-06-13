"""Build a curriculum for coverage-mode gaps.

Rows that already have a trusted continuation or an untrusted review candidate
are operationally covered. Rows with ``candidate_status=unavailable`` receive a
learning task that states what explicit memory, phrase, trace, or keep-audit
policy input is needed later.

This module is curriculum-only: it does not generate continuations, apply
memory, alter policy, or modify trusted/coverage outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter


CSV_FIELDS = [
    "task_id",
    "row_id",
    "gap_type",
    "priority",
    "required_input",
    "audit_reason",
    "minimal_human_question",
]

_TRUE_UNSAFE_MARKERS = {"anti_cue", "anti_cue_conflict", "only_negated_evidence", "negated_top_sense"}


def _read_csv(path: str) -> list[dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _task_id(row_id: str, gap_type: str) -> str:
    digest = hashlib.sha1(f"{row_id}|{gap_type}".encode("utf-8")).hexdigest()[:10]
    return f"task-{row_id}-{gap_type}-{digest}"


def _audit_by_id(audit_report: dict) -> dict[str, dict]:
    return {row["row_id"]: row for row in audit_report.get("rows", [])}


def _plan_by_id(plan: dict) -> dict[str, dict]:
    return {proposal["row_ids"][0]: proposal for proposal in plan.get("proposals", [])}


def _audit_reason(row: dict, audit_diag: dict | None) -> str:
    if audit_diag is not None:
        return audit_diag.get("primary_audit_reason", "")
    reasons = row.get("reasons", "")
    if "no_ambiguous_term" in reasons:
        return "missing_sense_memory"
    return "unknown_reason"


def _gap_type(row: dict, audit_reason: str) -> str:
    reasons = row.get("reasons", "")
    candidate_reason = row.get("candidate_reason", "")
    trace = row.get("candidate_trace", "")

    if audit_reason == "missing_sense_memory" or "no_ambiguous_term" in reasons:
        return "missing_sense_memory"
    if audit_reason == "true_unsafe" or any(marker in reasons for marker in _TRUE_UNSAFE_MARKERS):
        return "true_unsafe"
    if audit_reason == "unsupported_or_underconstrained_context":
        return "unsupported_context"
    if audit_reason == "no_safe_repaired_candidate":
        return "no_safe_rewrite"
    if "prompt_tail_validator=rejected" in reasons or "prompt_tail" in candidate_reason or "prompt_tail" in trace:
        return "prompt_tail_blocked"
    if audit_reason == "surface_validation_failed" or "no candidate survived" in candidate_reason:
        return "missing_phrase_candidate"
    if row.get("candidate_review_action") == "needs_memory":
        return "needs_trace_instrumentation"
    return "needs_trace_instrumentation"


def _required_input(gap_type: str) -> list[str]:
    if gap_type == "missing_sense_memory":
        return [
            "sense_memory_entry",
            "positive_cues",
            "anti_cues",
            "semantic_frame",
            "phrase_candidates",
            "human_label",
        ]
    if gap_type == "true_unsafe":
        return ["keep_audit_policy"]
    if gap_type == "unsupported_context":
        return ["human_label", "keep_audit_policy"]
    if gap_type == "no_safe_rewrite":
        return ["human_label", "phrase_candidates"]
    if gap_type == "missing_phrase_candidate":
        return ["phrase_candidates", "semantic_frame"]
    if gap_type == "prompt_tail_blocked":
        return ["prompt_tail_rule", "phrase_candidates"]
    return ["trace_instrumentation", "human_label"]


def _priority(row: dict, gap_type: str) -> str:
    expected = bool((row.get("expected_sense") or "").strip())
    if gap_type == "missing_sense_memory":
        return "high" if expected else "medium"
    if gap_type in {"missing_phrase_candidate", "prompt_tail_blocked", "no_safe_rewrite"}:
        return "medium"
    if gap_type == "true_unsafe":
        return "low"
    if gap_type == "unsupported_context":
        return "low"
    return "medium"


def _human_question(row: dict, gap_type: str) -> str:
    term = row.get("ambiguous_term", "") or "this term"
    expected = row.get("expected_sense", "") or "a known sense"
    if gap_type == "missing_sense_memory":
        return f"What explicit sense-memory entry and cues should represent {term}?"
    if gap_type == "true_unsafe":
        return "What evidence should be recorded to keep this row audited?"
    if gap_type == "unsupported_context":
        return "Should this prompt remain no-clear-answer, or is a missing label/cue available?"
    if gap_type == "no_safe_rewrite":
        return "Can a safe rewrite be stated without inventing unsupported entities or actions?"
    if gap_type == "missing_phrase_candidate":
        return f"What explicit phrase candidate safely realizes {term}:{expected} for this prompt tail?"
    if gap_type == "prompt_tail_blocked":
        return "What narrow prompt-tail rule or phrase candidate would fit without guessing?"
    return "What additional trace marker would explain why no candidate was available?"


def _payload(row: dict, gap_type: str, proposal: dict | None) -> dict:
    evidence = (proposal or {}).get("evidence", {})
    proposed = (proposal or {}).get("proposed_change", {})
    required = _required_input(gap_type)
    payload = {
        "term": row.get("ambiguous_term", ""),
        "target_sense": row.get("expected_sense", ""),
        "positive_cues_to_review": list(evidence.get("supporting_cues", [])),
        "anti_cues_to_review": list(evidence.get("conflicting_cues", [])),
        "guard_rule_to_review": proposed.get("add_guard_rule"),
        "semantic_frame_to_review": None,
        "phrase_candidates_to_review": [],
        "prompt_tail_rule_to_review": None,
        "keep_audit_rationale": None,
    }
    if "semantic_frame" in required:
        term = row.get("ambiguous_term", "")
        sense = row.get("expected_sense", "")
        payload["semantic_frame_to_review"] = f"{term}:{sense}" if term and sense else None
    if "phrase_candidates" in required:
        candidate = proposed.get("add_phrase_candidate")
        payload["phrase_candidates_to_review"] = [candidate] if candidate else []
    if "prompt_tail_rule" in required:
        payload["prompt_tail_rule_to_review"] = proposed.get("add_prompt_tail_rule") or "narrow prompt-tail rule for this prompt shape"
    if gap_type in {"true_unsafe", "unsupported_context", "no_safe_rewrite"}:
        payload["keep_audit_rationale"] = row.get("candidate_reason") or "keep audited unless human reviewer supplies safe explicit memory"
    return payload


def _why(row: dict, gap_type: str, audit_reason: str) -> str:
    candidate_reason = row.get("candidate_reason", "")
    if gap_type == "true_unsafe":
        return "Audit traces indicate contradictory, negated, or unsafe evidence; learn the keep-audit condition."
    if gap_type == "unsupported_context":
        return "The prompt is underconstrained or intentionally has no clear answer."
    if gap_type == "missing_sense_memory":
        return "The row lacks an explicit known term or sense-memory entry."
    if gap_type == "no_safe_rewrite":
        return "Surface repair could not produce a safe trusted or candidate rewrite."
    if candidate_reason:
        return candidate_reason
    return f"Unavailable candidate from audit reason {audit_reason}."


def _learning_task(row: dict, audit_diag: dict | None, proposal: dict | None) -> dict:
    audit_reason = _audit_reason(row, audit_diag)
    gap_type = _gap_type(row, audit_reason)
    required = _required_input(gap_type)
    return {
        "task_id": _task_id(row.get("id", ""), gap_type),
        "row_id": row.get("id", ""),
        "prompt": row.get("prompt", ""),
        "term": row.get("ambiguous_term", ""),
        "expected_sense": row.get("expected_sense", ""),
        "trusted_decision": row.get("trusted_decision", ""),
        "candidate_status": row.get("candidate_status", ""),
        "audit_reason": audit_reason,
        "gap_type": gap_type,
        "priority": _priority(row, gap_type),
        "required_input": required,
        "why_unavailable": _why(row, gap_type, audit_reason),
        "minimal_human_question": _human_question(row, gap_type),
        "proposed_learning_payload": _payload(row, gap_type, proposal),
        "safety_notes": ["do_not_auto_continue", "requires_human_review"],
    }


def build_curriculum(
    coverage_rows: list[dict],
    coverage_summary: dict,
    audit_report: dict,
    audit_plan: dict,
    dataset_rows: list[dict],
) -> dict:
    audit = _audit_by_id(audit_report)
    proposals = _plan_by_id(audit_plan)
    tasks = [
        _learning_task(row, audit.get(row.get("id", "")), proposals.get(row.get("id", "")))
        for row in coverage_rows
        if row.get("candidate_status") == "unavailable"
    ]
    total_rows = coverage_summary.get("total_rows", len(coverage_rows))
    trusted = coverage_summary.get("trusted_continue_count", 0)
    untrusted = coverage_summary.get("candidate_by_status", {}).get("untrusted", 0)
    operational = trusted + untrusted + len(tasks)
    required_counts = Counter()
    for task in tasks:
        required_counts.update(task["required_input"])
    summary = {
        "total_rows": total_rows,
        "trusted_continue_count": trusted,
        "untrusted_candidate_count": untrusted,
        "learning_task_count": len(tasks),
        "operational_coverage_count": operational,
        "operational_coverage_rate": round(operational / total_rows, 4) if total_rows else 0.0,
        "by_gap_type": dict(sorted(Counter(task["gap_type"] for task in tasks).items())),
        "by_required_input": dict(sorted(required_counts.items())),
        "by_priority": dict(sorted(Counter(task["priority"] for task in tasks).items())),
    }
    return {
        "summary": summary,
        "policy": {
            "mode": "curriculum_only",
            "generation_behavior_changed": False,
            "auto_apply": False,
            "trusted_policy_changed": False,
        },
        "learning_tasks": tasks,
    }


def write_json(curriculum: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(curriculum, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(curriculum: dict, path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for task in curriculum["learning_tasks"]:
            writer.writerow(
                {
                    "task_id": task["task_id"],
                    "row_id": task["row_id"],
                    "gap_type": task["gap_type"],
                    "priority": task["priority"],
                    "required_input": ";".join(task["required_input"]),
                    "audit_reason": task["audit_reason"],
                    "minimal_human_question": task["minimal_human_question"],
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build curriculum tasks for coverage-mode unavailable rows.")
    parser.add_argument("--coverage-output", required=True)
    parser.add_argument("--coverage-summary", required=True)
    parser.add_argument("--audit-reasons", required=True)
    parser.add_argument("--audit-plan", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=False)
    args = parser.parse_args()

    curriculum = build_curriculum(
        _read_csv(args.coverage_output),
        _read_json(args.coverage_summary),
        _read_json(args.audit_reasons),
        _read_json(args.audit_plan),
        _read_csv(args.dataset),
    )
    write_json(curriculum, args.output_json)
    if args.output_csv:
        write_csv(curriculum, args.output_csv)
    print(json.dumps(curriculum["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
