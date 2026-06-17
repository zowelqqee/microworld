"""Summary + report builder for the Curriculum Regression Gate v1."""

from __future__ import annotations

from typing import Any, Dict, List

from .types import (
    CAT_CURRENT,
    CAT_INVERSION,
    CAT_MISSING,
    CAT_PRIVATE,
    CAT_UNIVERSAL,
    CAT_VOLATILE,
    CAT_WEAK_LINK,
    AdversarialCaseResult,
    CurriculumCaseResult,
    LoadedProposals,
)


def _all_proposal_only(items: List[dict]) -> bool:
    for item in items or []:
        if not isinstance(item, dict):
            return False
        if item.get("activation_status") != "proposal_only":
            return False
    return True


def _count_decision(results, decision: str) -> int:
    return sum(1 for r in results if (r.observed.decision or "").lower() == decision)


def _guard_pass_count(adv_results, cur_results, category: str) -> int:
    count = 0
    for r in adv_results:
        if r.case.attack_category == category and r.passed:
            count += 1
    for r in cur_results:
        if r.case.category == category and r.passed:
            count += 1
    return count


def build_summary(
    loaded: LoadedProposals,
    curriculum_results: List[CurriculumCaseResult],
    adversarial_results: List[AdversarialCaseResult],
) -> Dict[str, Any]:
    all_results = list(curriculum_results) + list(adversarial_results)

    curriculum_passed = sum(1 for r in curriculum_results if r.passed)
    adversarial_passed = sum(1 for r in adversarial_results if r.passed)

    unsafe_answer_count = sum(
        1 for r in all_results if not r.passed and "unsafe_answer" in r.failure_reason
    )

    missing_knowledge_audit_count = sum(
        1
        for r in curriculum_results
        if r.case.category == CAT_MISSING
        and (r.observed.decision or "").lower() == "audit"
    )

    rules_activated = not _all_proposal_only(loaded.analyzer_rules)
    patterns_activated = not _all_proposal_only(loaded.extraction_patterns)

    all_passed = curriculum_passed == len(curriculum_results) and adversarial_passed == len(
        adversarial_results
    )

    return {
        "curriculum_total": len(curriculum_results),
        "curriculum_passed": curriculum_passed,
        "curriculum_failed": len(curriculum_results) - curriculum_passed,
        "adversarial_total": len(adversarial_results),
        "adversarial_passed": adversarial_passed,
        "adversarial_failed": len(adversarial_results) - adversarial_passed,
        "answer_count": _count_decision(all_results, "answer"),
        "audit_count": _count_decision(all_results, "audit"),
        "reject_count": _count_decision(all_results, "reject"),
        "unsafe_answer_count": unsafe_answer_count,
        "missing_knowledge_audit_count": missing_knowledge_audit_count,
        "weak_link_safety_pass_count": _guard_pass_count(
            adversarial_results, curriculum_results, CAT_WEAK_LINK
        ),
        "inversion_guard_pass_count": _guard_pass_count(
            adversarial_results, curriculum_results, CAT_INVERSION
        ),
        "current_claim_guard_pass_count": _guard_pass_count(
            adversarial_results, curriculum_results, CAT_CURRENT
        ),
        "volatile_guard_pass_count": _guard_pass_count(
            adversarial_results, curriculum_results, CAT_VOLATILE
        ),
        "private_guard_pass_count": _guard_pass_count(
            adversarial_results, curriculum_results, CAT_PRIVATE
        ),
        "unsupported_universal_guard_pass_count": _guard_pass_count(
            adversarial_results, curriculum_results, CAT_UNIVERSAL
        ),
        "all_passed": all_passed,
        "safe_for_general_runtime": False,
        "proposal_only": True,
        "rules_activated": rules_activated,
        "patterns_activated": patterns_activated,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
    }


def build_report(
    summary: Dict[str, Any],
    loaded: LoadedProposals,
    curriculum_results: List[CurriculumCaseResult],
    adversarial_results: List[AdversarialCaseResult],
    source_artifact_paths: List[str],
) -> Dict[str, Any]:
    failed_cases: List[Dict[str, Any]] = []
    for r in curriculum_results:
        if not r.passed:
            failed_cases.append(
                {
                    "kind": "curriculum",
                    "id": r.case.id,
                    "category": r.case.category,
                    "question": r.case.question,
                    "expected_decision": r.case.expected_decision,
                    "observed_decision": r.observed.decision,
                    "failure_reason": r.failure_reason,
                }
            )
    for r in adversarial_results:
        if not r.passed:
            failed_cases.append(
                {
                    "kind": "adversarial",
                    "id": r.case.id,
                    "attack_category": r.case.attack_category,
                    "question": r.case.question,
                    "expected_decision": r.case.expected_decision,
                    "observed_decision": r.observed.decision,
                    "failure_reason": r.failure_reason,
                }
            )

    next_actions = [
        "Keep all self-learning extraction patterns and analyzer rules "
        "proposal_only; do not activate them based on this gate.",
        "Re-run this gate after any future ingestion/promotion to confirm "
        "missing-knowledge rows still audit until explicit support is added.",
    ]
    if not summary["all_passed"]:
        next_actions.insert(
            0,
            "Investigate failed cases below WITHOUT weakening safety; a dangerous "
            "case answered as fact indicates a real regression to fix.",
        )

    return {
        "summary": summary,
        "source_artifact_paths": list(source_artifact_paths),
        "warnings": list(loaded.warnings),
        "errors": list(loaded.errors),
        "failed_cases": failed_cases,
        "safety_confirmations": {
            "proposal_only": True,
            "rules_activated": summary["rules_activated"],
            "patterns_activated": summary["patterns_activated"],
            "trusted_memory_modified": False,
            "accepted_overlay_modified": False,
            "promoted_overlay_modified": False,
            "safe_for_general_runtime": False,
            "unsafe_answer_count": summary["unsafe_answer_count"],
        },
        "next_recommended_actions": next_actions,
    }
