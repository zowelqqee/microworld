"""Deterministic data model for the Curriculum Regression Gate v1.

Plain dataclasses only. No I/O, no runtime coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# Categories whose claims must never be answered as stable / live fact.
# A weak-link case additionally permits a safe-policy explanation answer.
CAT_WEAK_LINK = "weak_link_safety_case"
CAT_INVERSION = "relation_inversion_guard_case"
CAT_CURRENT = "current_claim_guard_case"
CAT_UNIVERSAL = "unsupported_universal_guard_case"
CAT_PRIVATE = "adversarial_test_candidate"
CAT_VOLATILE = "volatile_review_candidate"
CAT_MISSING = "missing_stable_knowledge"

DANGEROUS_CATEGORIES = frozenset(
    {
        CAT_WEAK_LINK,
        CAT_INVERSION,
        CAT_CURRENT,
        CAT_UNIVERSAL,
        CAT_PRIVATE,
        CAT_VOLATILE,
    }
)

ROUTE_ENTITY = "entity_qa"
ROUTE_CROSS_PAGE = "cross_page_qa"

# Safe (non-failing) observed decisions for a dangerous / guard case.
SAFE_DECISIONS = frozenset({"audit", "reject"})


@dataclass
class CurriculumCase:
    """A proposed curriculum row to be replayed through the gate."""

    id: str
    category: str
    question: str
    expected_decision: str
    expected_answer_contains: str
    expected_audit_reason: str
    source_opportunity_id: str
    activation_status: str


@dataclass
class AdversarialCase:
    """A proposed adversarial / safety case to be replayed through the gate."""

    id: str
    attack_category: str
    question: str
    expected_decision: str
    expected_safety_reason: str
    source_opportunity_id: str


@dataclass
class ObservedResult:
    """What a QA pipeline actually decided for a question."""

    decision: str
    answer: str
    intent: str
    route: str
    audit_reason: str = ""


@dataclass
class CurriculumCaseResult:
    case: CurriculumCase
    observed: ObservedResult
    passed: bool
    failure_reason: str


@dataclass
class AdversarialCaseResult:
    case: AdversarialCase
    observed: ObservedResult
    passed: bool
    failure_reason: str


@dataclass
class LoadedProposals:
    """Everything the loader read from the self-learning proposal layer."""

    curriculum_cases: List[CurriculumCase] = field(default_factory=list)
    adversarial_cases: List[AdversarialCase] = field(default_factory=list)
    extraction_patterns: List[dict] = field(default_factory=list)
    analyzer_rules: List[dict] = field(default_factory=list)
    self_learning_report: Optional[dict] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_artifacts_read: List[str] = field(default_factory=list)
