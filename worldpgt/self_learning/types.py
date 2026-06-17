"""Deterministic data model for the Self-Learning Loop v1.

Plain dataclasses only. No behavior, no I/O, no runtime coupling. Each type is
JSON-serializable via :func:`dataclasses.asdict`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# Activation statuses.
#
# NOTE: there is intentionally NO ``auto_apply`` status. Nothing produced by
# this package may be applied automatically; every artifact is a proposal that
# requires either human review, routing through ingestion, or a regression gate
# before it could ever be acted on.
# --------------------------------------------------------------------------- #
PROPOSAL_ONLY = "proposal_only"
REQUIRES_HUMAN_REVIEW = "requires_human_review"
REQUIRES_INGESTION = "requires_ingestion"
REQUIRES_REGRESSION_GATE = "requires_regression_gate"

ALLOWED_ACTIVATION_STATUSES = frozenset(
    {
        PROPOSAL_ONLY,
        REQUIRES_HUMAN_REVIEW,
        REQUIRES_INGESTION,
        REQUIRES_REGRESSION_GATE,
    }
)

# --------------------------------------------------------------------------- #
# Opportunity categories.
# --------------------------------------------------------------------------- #
CAT_MISSING_STABLE_KNOWLEDGE = "missing_stable_knowledge"
CAT_ADVERSARIAL_TEST_CANDIDATE = "adversarial_test_candidate"
CAT_WEAK_LINK_SAFETY_CASE = "weak_link_safety_case"
CAT_VOLATILE_REVIEW_CANDIDATE = "volatile_review_candidate"
CAT_CURRENT_CLAIM_GUARD_CASE = "current_claim_guard_case"
CAT_RELATION_INVERSION_GUARD_CASE = "relation_inversion_guard_case"
CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE = "unsupported_universal_guard_case"
CAT_EXTRACTION_PATTERN_CANDIDATE = "extraction_pattern_candidate"
CAT_ANALYZER_RULE_CANDIDATE = "analyzer_rule_candidate"

OPPORTUNITY_CATEGORIES = frozenset(
    {
        CAT_MISSING_STABLE_KNOWLEDGE,
        CAT_ADVERSARIAL_TEST_CANDIDATE,
        CAT_WEAK_LINK_SAFETY_CASE,
        CAT_VOLATILE_REVIEW_CANDIDATE,
        CAT_CURRENT_CLAIM_GUARD_CASE,
        CAT_RELATION_INVERSION_GUARD_CASE,
        CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE,
        CAT_EXTRACTION_PATTERN_CANDIDATE,
        CAT_ANALYZER_RULE_CANDIDATE,
    }
)

PRIORITY_HIGH = "high"
PRIORITY_MEDIUM = "medium"
PRIORITY_LOW = "low"


@dataclass
class LearningSignal:
    """A normalized observation mined from a single source artifact row."""

    signal_type: str
    source_artifact: str
    text: str
    source_row_id: Optional[str] = None
    reason: Optional[str] = None
    risk: Optional[str] = None
    observed_decision: Optional[str] = None
    count: int = 1
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningOpportunity:
    """A structured, non-actionable improvement proposal."""

    id: str
    category: str
    priority: str
    source_artifact: str
    question_or_claim: str
    expected_policy: str
    reason: str
    suggested_next_step: str
    activation_status: str
    source_row_id: Optional[str] = None
    observed_decision: Optional[str] = None


@dataclass
class CurriculumRow:
    """A proposed benchmark row (never auto-added to any live benchmark)."""

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
    """A proposed adversarial / safety regression row."""

    id: str
    attack_category: str
    question: str
    expected_decision: str
    expected_safety_reason: str
    source_opportunity_id: str


@dataclass
class PatternProposal:
    """A proposed extraction pattern. Proposal only; never activated."""

    proposal_type: str
    pattern: str
    relation: str
    risk: str
    requires_tests: bool
    activation_status: str
    source_opportunity_id: Optional[str] = None


@dataclass
class AnalyzerRuleProposal:
    """A proposed analyzer rule. Proposal only; never activated."""

    proposal_type: str
    intent: str
    surface_patterns: List[str]
    expected_decision: str
    activation_status: str
    source_opportunity_id: Optional[str] = None


@dataclass
class SelfLearningReport:
    """Top-level report describing the proposal layer produced by one run."""

    safe_for_general_runtime: bool
    trusted_memory_modified: bool
    accepted_overlay_modified: bool
    promoted_overlay_modified: bool
    proposal_only: bool
    opportunities_total: int
    opportunities_by_category: Dict[str, int]
    opportunities_by_priority: Dict[str, int]
    curriculum_rows_total: int
    adversarial_rows_total: int
    pattern_proposals_total: int
    analyzer_rule_proposals_total: int
    warnings: List[str]
    errors: List[str]
    source_artifacts_read: List[str]
    regression_status_observed: Dict[str, Any]
    next_recommended_actions: List[str]
