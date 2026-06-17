"""Metrics, failure taxonomy and artifact writers for Pump Fact QA Benchmark."""

from __future__ import annotations

from collections import Counter
from typing import Any

from worldpgt.pump_fact_qa.types import (
    CATEGORY_ADVERSARIAL,
    CATEGORY_CURRENT,
    CATEGORY_POSITIVE,
    CLASS_OK,
    CLASS_PLANNER_GAP,
    CLASS_SOURCE_OR_VOLATILITY_SENSITIVE,
    CLASS_UNSAFE_CURRENT_ANSWER,
    CLASS_UNSUPPORTED_ANSWER,
    CLASS_WRONG_ANSWER,
    CLASS_INVERSION_FALSE_ANSWER,
    FACT_KIND_DEFINITION,
    FACT_KIND_RELATION,
    PumpFact,
    QAResult,
)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def build_summary(facts: list[PumpFact], results: list[QAResult]) -> dict[str, Any]:
    relation_facts = [f for f in facts if f.kind == FACT_KIND_RELATION]
    definition_facts = [f for f in facts if f.kind == FACT_KIND_DEFINITION]

    by_cat: dict[str, list[QAResult]] = {
        CATEGORY_POSITIVE: [],
        CATEGORY_ADVERSARIAL: [],
        CATEGORY_CURRENT: [],
    }
    for result in results:
        by_cat.setdefault(result.prompt.category, []).append(result)

    positive = by_cat[CATEGORY_POSITIVE]
    adversarial = by_cat[CATEGORY_ADVERSARIAL]
    current = by_cat[CATEGORY_CURRENT]

    # Positive breakdown.
    positive_answer = [r for r in positive if r.valid and r.classification == CLASS_OK]
    positive_audit = [r for r in positive if r.decision == "audit"]
    positive_planner_gap = [
        r for r in positive
        if r.classification in (CLASS_PLANNER_GAP, CLASS_SOURCE_OR_VOLATILITY_SENSITIVE)
    ]
    positive_wrong = [r for r in positive if r.classification == CLASS_WRONG_ANSWER]
    positive_unsupported = [r for r in positive if r.classification == CLASS_UNSUPPORTED_ANSWER]

    # Adversarial / current breakdown.
    adversarial_pass = [r for r in adversarial if r.valid]
    adversarial_fail = [r for r in adversarial if not r.valid]
    current_pass = [r for r in current if r.valid]
    current_fail = [r for r in current if not r.valid]

    # Critical failure counters.
    wrong_count = len(positive_wrong)
    unsupported_count = len(positive_unsupported)
    unsafe_current_count = sum(
        1 for r in results if r.classification == CLASS_UNSAFE_CURRENT_ANSWER
    )
    inversion_count = sum(
        1 for r in results if r.classification == CLASS_INVERSION_FALSE_ANSWER
    )

    answered_positive_total = len(positive_answer) + wrong_count + unsupported_count

    all_critical_passed = (
        wrong_count == 0
        and unsupported_count == 0
        and unsafe_current_count == 0
        and inversion_count == 0
    )

    classification_counts = dict(
        sorted(Counter(r.classification for r in results).items())
    )

    return {
        "pump_fact_qa_enabled": True,
        "pump_fact_qa_fact_count": len(facts),
        "pump_fact_qa_relation_fact_count": len(relation_facts),
        "pump_fact_qa_definition_fact_count": len(definition_facts),
        "pump_fact_qa_positive_prompt_count": len(positive),
        "pump_fact_qa_adversarial_prompt_count": len(adversarial),
        "pump_fact_qa_current_prompt_count": len(current),
        "pump_fact_qa_total_prompt_count": len(results),
        "pump_fact_qa_positive_answer_count": len(positive_answer),
        "pump_fact_qa_positive_audit_count": len(positive_audit),
        "pump_fact_qa_positive_planner_gap_count": len(positive_planner_gap),
        "pump_fact_qa_positive_wrong_count": wrong_count,
        "pump_fact_qa_positive_unsupported_count": unsupported_count,
        "pump_fact_qa_adversarial_pass_count": len(adversarial_pass),
        "pump_fact_qa_adversarial_fail_count": len(adversarial_fail),
        "pump_fact_qa_current_safety_pass_count": len(current_pass),
        "pump_fact_qa_current_safety_fail_count": len(current_fail),
        "pump_fact_qa_answer_precision": _ratio(len(positive_answer), answered_positive_total),
        "pump_fact_qa_supported_answer_rate": _ratio(len(positive_answer), len(positive)),
        "pump_fact_qa_planner_gap_rate": _ratio(len(positive_planner_gap), len(positive)),
        # Critical counters (also surfaced under canonical names for the pump summary).
        "wrong_count": wrong_count,
        "unsupported_answer_count": unsupported_count,
        "unsafe_current_answer_count": unsafe_current_count,
        "inversion_false_answer_count": inversion_count,
        "pump_fact_qa_all_critical_passed": all_critical_passed,
        "classification_counts": classification_counts,
        # Short-name stable aliases (Part F).
        "fact_count": len(facts),
        "relation_fact_count": len(relation_facts),
        "definition_fact_count": len(definition_facts),
        "positive_prompt_count": len(positive),
        "positive_answer_count": len(positive_answer),
        "positive_audit_count": len(positive_audit),
        "positive_planner_gap_count": len(positive_planner_gap),
        "wrong_answer_count": wrong_count,
        "unsupported_answer_count": unsupported_count,
        "adversarial_fail_count": len(adversarial_fail),
        "current_safety_fail_count": len(current_fail),
        "answer_precision": _ratio(len(positive_answer), answered_positive_total),
        "supported_answer_rate": _ratio(len(positive_answer), len(positive)),
        "planner_gap_rate": _ratio(len(positive_planner_gap), len(positive)),
        "all_critical_passed": all_critical_passed,
    }


def collect_failures(results: list[QAResult]) -> list[dict[str, Any]]:
    """Genuine failures only: invalid results that are not expected planner gaps."""
    expected_non_failures = {CLASS_OK, CLASS_PLANNER_GAP, CLASS_SOURCE_OR_VOLATILITY_SENSITIVE}
    failures: list[dict[str, Any]] = []
    for result in results:
        if result.valid:
            continue
        if result.classification in expected_non_failures:
            continue
        failures.append(result.to_dict())
    return failures


def collect_planner_gaps(results: list[QAResult]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    for result in results:
        if result.classification in (CLASS_PLANNER_GAP, CLASS_SOURCE_OR_VOLATILITY_SENSITIVE):
            gaps.append(result.to_dict())
    return gaps


def compact_pump_section(summary: dict[str, Any]) -> dict[str, Any]:
    """The compact section merged into pump_summary.json / pump_report.json."""
    return {
        "pump_fact_qa_enabled": summary["pump_fact_qa_enabled"],
        "pump_fact_qa_fact_count": summary["pump_fact_qa_fact_count"],
        "pump_fact_qa_relation_fact_count": summary["pump_fact_qa_relation_fact_count"],
        "pump_fact_qa_definition_fact_count": summary["pump_fact_qa_definition_fact_count"],
        "pump_fact_qa_fact_count_after_v2": summary["pump_fact_qa_fact_count"],
        "pump_fact_qa_total_prompt_count": summary["pump_fact_qa_total_prompt_count"],
        "pump_fact_qa_positive_answer_count": summary["pump_fact_qa_positive_answer_count"],
        "pump_fact_qa_positive_planner_gap_count": summary["pump_fact_qa_positive_planner_gap_count"],
        "pump_fact_qa_wrong_count": summary["wrong_count"],
        "pump_fact_qa_unsupported_answer_count": summary["unsupported_answer_count"],
        "pump_fact_qa_all_critical_passed": summary["pump_fact_qa_all_critical_passed"],
    }
