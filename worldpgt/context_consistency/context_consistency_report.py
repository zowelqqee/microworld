"""Summary + report builder for the Context-Pack QA Consistency Gate v1."""

from __future__ import annotations

from typing import Any, Dict, List

from .types import (
    FAIL_STATUSES,
    PASS_STATUSES,
    WARNING_STATUSES,
    STATUS_ANSWER_SUPPORTED,
    STATUS_ANSWER_WITHOUT_SUPPORT,
    STATUS_AUDIT_DESPITE_SAFE,
    STATUS_AUDIT_SUPPORTED,
    STATUS_CURRENT_LIVE_FALSE,
    STATUS_INVERSION_FALSE,
    STATUS_PRIVATE_FALSE,
    STATUS_SAFE_POLICY_ANSWER,
    STATUS_UNIVERSAL_FALSE,
    STATUS_VALIDATION_FAILED,
    STATUS_VOLATILE_FALSE,
    STATUS_WEAK_LINK_FALSE,
    ContextConsistencyResult,
)


def build_summary(
    results: List[ContextConsistencyResult],
    qa_rows_total: int,
    skipped_rows_total: int,
) -> Dict[str, Any]:
    def count(status: str) -> int:
        return sum(1 for r in results if r.consistency_status == status)

    answer_rows = sum(1 for r in results if (r.qa_decision or "").lower() == "answer")
    audit_rows = len(results) - answer_rows

    passed = sum(1 for r in results if r.consistency_status in PASS_STATUSES)
    warning = sum(1 for r in results if r.consistency_status in WARNING_STATUSES)
    failed = sum(1 for r in results if r.consistency_status in FAIL_STATUSES)

    answer_without = count(STATUS_ANSWER_WITHOUT_SUPPORT)
    weak_false = count(STATUS_WEAK_LINK_FALSE)
    volatile_false = count(STATUS_VOLATILE_FALSE)
    current_false = count(STATUS_CURRENT_LIVE_FALSE)
    inversion_false = count(STATUS_INVERSION_FALSE)
    private_false = count(STATUS_PRIVATE_FALSE)
    universal_false = count(STATUS_UNIVERSAL_FALSE)
    validation_failed = count(STATUS_VALIDATION_FAILED)

    all_critical_passed = (
        answer_without == 0
        and weak_false == 0
        and volatile_false == 0
        and current_false == 0
        and inversion_false == 0
        and private_false == 0
        and universal_false == 0
        and validation_failed == 0
    )

    return {
        "qa_rows_total": qa_rows_total,
        "checked_rows_total": len(results),
        "skipped_rows_total": skipped_rows_total,
        "answer_rows": answer_rows,
        "audit_rows": audit_rows,
        "answer_supported_by_context": count(STATUS_ANSWER_SUPPORTED),
        "audit_supported_by_context": count(STATUS_AUDIT_SUPPORTED),
        "safe_policy_answer_supported": count(STATUS_SAFE_POLICY_ANSWER),
        "answer_without_context_support": answer_without,
        "audit_despite_safe_context": count(STATUS_AUDIT_DESPITE_SAFE),
        "weak_link_false_support_count": weak_false,
        "volatile_false_stable_count": volatile_false,
        "current_live_false_support_count": current_false,
        "inversion_false_support_count": inversion_false,
        "private_false_support_count": private_false,
        "unsupported_universal_false_support_count": universal_false,
        "context_validation_failed_count": validation_failed,
        "consistency_passed_count": passed,
        "consistency_warning_count": warning,
        "consistency_failed_count": failed,
        "all_critical_passed": all_critical_passed,
        "safe_for_general_runtime": False,
        "runtime_behavior_modified": False,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "network_calls": False,
    }


def build_report(
    summary: Dict[str, Any],
    results: List[ContextConsistencyResult],
    source_artifacts: List[str],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    def _case(r: ContextConsistencyResult) -> dict:
        return {
            "id": r.id,
            "source_suite": r.source_suite,
            "question": r.question,
            "qa_decision": r.qa_decision,
            "consistency_status": r.consistency_status,
            "failure_reason": r.failure_reason,
            "risk_flags": r.risk_flags,
        }

    failed_cases = [_case(r) for r in results if r.consistency_status in FAIL_STATUSES]
    warning_cases = [_case(r) for r in results if r.consistency_status in WARNING_STATUSES]

    next_actions = [
        "Keep the context pack a DRY-RUN probe; do not wire it into planners "
        "based on this gate.",
        "Review audit_despite_safe_context warnings: QA is more conservative than "
        "the direction-agnostic context pack (e.g. inverted phrasings); keep QA's "
        "guard, do not relax it.",
    ]
    if not summary["all_critical_passed"]:
        next_actions.insert(
            0,
            "Investigate critical failures below WITHOUT weakening QA or validators; "
            "a false-support row indicates a real regression.",
        )

    return {
        "summary": summary,
        "source_qa_artifacts_read": list(source_artifacts),
        "warnings": list(warnings),
        "errors": list(errors),
        "failed_cases": failed_cases,
        "warning_cases": warning_cases,
        "safety_confirmations": {
            "safe_for_general_runtime": False,
            "runtime_behavior_modified": False,
            "trusted_memory_modified": False,
            "accepted_overlay_modified": False,
            "promoted_overlay_modified": False,
            "network_calls": False,
            "rules_activated": False,
            "patterns_activated": False,
            "qa_behavior_modified": False,
        },
        "next_recommended_actions": next_actions,
    }
