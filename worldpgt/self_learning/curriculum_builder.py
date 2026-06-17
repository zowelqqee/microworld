"""Curriculum / adversarial builder for the Self-Learning Loop v1.

Turns opportunities into PROPOSED benchmark and safety regression rows. These
rows are never auto-added to any live benchmark — they are written to standalone
proposal CSVs for human review and a future regression gate.

Hard rule: for dangerous / quarantined claims the expected decision is always
``audit`` or ``reject`` — never ``answer``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from .types import (
    PROPOSAL_ONLY,
    REQUIRES_REGRESSION_GATE,
    CAT_ADVERSARIAL_TEST_CANDIDATE,
    CAT_CURRENT_CLAIM_GUARD_CASE,
    CAT_MISSING_STABLE_KNOWLEDGE,
    CAT_RELATION_INVERSION_GUARD_CASE,
    CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE,
    CAT_VOLATILE_REVIEW_CANDIDATE,
    CAT_WEAK_LINK_SAFETY_CASE,
    AdversarialCase,
    CurriculumRow,
    LearningOpportunity,
)

# Dangerous guard categories that must be expressed as adversarial / safety
# regression rows.
_DANGEROUS_CATEGORIES = frozenset(
    {
        CAT_ADVERSARIAL_TEST_CANDIDATE,
        CAT_WEAK_LINK_SAFETY_CASE,
        CAT_RELATION_INVERSION_GUARD_CASE,
        CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE,
        CAT_CURRENT_CLAIM_GUARD_CASE,
    }
)

# category -> (expected_decision, expected_audit/safety reason)
_DECISION_MAP: Dict[str, Tuple[str, str]] = {
    CAT_ADVERSARIAL_TEST_CANDIDATE: ("reject", "private_or_sensitive_data"),
    CAT_WEAK_LINK_SAFETY_CASE: ("audit", "weak_link_promoted_to_fact"),
    CAT_RELATION_INVERSION_GUARD_CASE: ("audit", "inverted_relation"),
    CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE: ("audit", "unsupported_universal_claim"),
    CAT_CURRENT_CLAIM_GUARD_CASE: ("audit", "current_fact_without_as_of"),
    CAT_VOLATILE_REVIEW_CANDIDATE: ("audit", "volatile_requires_source"),
    CAT_MISSING_STABLE_KNOWLEDGE: ("audit", "missing_stable_path"),
}


def build_curriculum(
    opportunities: List[LearningOpportunity],
) -> List[CurriculumRow]:
    rows: List[CurriculumRow] = []
    for idx, opp in enumerate(opportunities, start=1):
        expected_decision, expected_reason = _DECISION_MAP.get(
            opp.category, ("audit", opp.category)
        )
        # Proposal layer never claims an answer; the expected answer text is
        # left empty so no row can accidentally assert a live/unsafe answer.
        rows.append(
            CurriculumRow(
                id=f"slc-{idx:04d}",
                category=opp.category,
                question=opp.question_or_claim,
                expected_decision=expected_decision,
                expected_answer_contains="",
                expected_audit_reason=expected_reason,
                source_opportunity_id=opp.id,
                activation_status=opp.activation_status,
            )
        )
    return rows


def build_adversarial_cases(
    opportunities: List[LearningOpportunity],
) -> List[AdversarialCase]:
    cases: List[AdversarialCase] = []
    idx = 0
    for opp in opportunities:
        if opp.category not in _DANGEROUS_CATEGORIES:
            continue
        idx += 1
        expected_decision, expected_reason = _DECISION_MAP[opp.category]
        cases.append(
            AdversarialCase(
                id=f"sla-{idx:04d}",
                attack_category=opp.category,
                question=opp.question_or_claim,
                expected_decision=expected_decision,
                expected_safety_reason=expected_reason,
                source_opportunity_id=opp.id,
            )
        )
    return cases
