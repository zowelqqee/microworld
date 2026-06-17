"""Deterministic data model for Context-Pack QA Consistency Gate v1.

Plain dataclasses only. No I/O, no runtime coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Consistency statuses.
# --------------------------------------------------------------------------- #
STATUS_ANSWER_SUPPORTED = "answer_supported_by_context"
STATUS_AUDIT_SUPPORTED = "audit_supported_by_context"
STATUS_SAFE_POLICY_ANSWER = "safe_policy_answer_supported"
STATUS_ANSWER_WITHOUT_SUPPORT = "answer_without_context_support"
STATUS_AUDIT_DESPITE_SAFE = "audit_despite_safe_context"
STATUS_WEAK_LINK_FALSE = "weak_link_false_support"
STATUS_VOLATILE_FALSE = "volatile_false_stable_support"
STATUS_CURRENT_LIVE_FALSE = "current_live_false_support"
STATUS_INVERSION_FALSE = "inversion_false_support"
STATUS_PRIVATE_FALSE = "private_false_support"
STATUS_UNIVERSAL_FALSE = "unsupported_universal_false_support"
STATUS_VALIDATION_FAILED = "context_validation_failed"

PASS_STATUSES = frozenset(
    {STATUS_ANSWER_SUPPORTED, STATUS_AUDIT_SUPPORTED, STATUS_SAFE_POLICY_ANSWER}
)
WARNING_STATUSES = frozenset({STATUS_AUDIT_DESPITE_SAFE})
# Failing statuses are everything else (false support / no support / validation).
FAIL_STATUSES = frozenset(
    {
        STATUS_ANSWER_WITHOUT_SUPPORT,
        STATUS_WEAK_LINK_FALSE,
        STATUS_VOLATILE_FALSE,
        STATUS_CURRENT_LIVE_FALSE,
        STATUS_INVERSION_FALSE,
        STATUS_PRIVATE_FALSE,
        STATUS_UNIVERSAL_FALSE,
        STATUS_VALIDATION_FAILED,
    }
)


@dataclass
class LoadedQAResult:
    id: str
    source_suite: str
    question: str
    qa_decision: str
    qa_intent: str
    qa_answer: str
    qa_correct: Optional[bool] = None
    expected_decision: str = ""


@dataclass
class ContextConsistencyResult:
    id: str
    source_suite: str
    question: str
    qa_decision: str
    qa_intent: str
    qa_answer: str
    qa_correct: Optional[bool]
    context_safe_for_answering: bool
    context_candidate_paths_count: int
    context_blocked_paths_count: int
    context_weak_links_count: int
    context_source_facts_count: int
    context_missing_hints_count: int
    validation_passed: bool
    consistency_status: str
    failure_reason: str
    risk_flags: List[str] = field(default_factory=list)


@dataclass
class ContextConsistencySummary:
    qa_rows_total: int
    checked_rows_total: int
    skipped_rows_total: int
    answer_rows: int
    audit_rows: int
    answer_supported_by_context: int
    audit_supported_by_context: int
    safe_policy_answer_supported: int
    answer_without_context_support: int
    audit_despite_safe_context: int
    weak_link_false_support_count: int
    volatile_false_stable_count: int
    current_live_false_support_count: int
    inversion_false_support_count: int
    private_false_support_count: int
    unsupported_universal_false_support_count: int
    context_validation_failed_count: int
    consistency_passed_count: int
    consistency_warning_count: int
    consistency_failed_count: int
    all_critical_passed: bool
    safe_for_general_runtime: bool = False
    runtime_behavior_modified: bool = False
    trusted_memory_modified: bool = False
    accepted_overlay_modified: bool = False
    promoted_overlay_modified: bool = False
    network_calls: bool = False


@dataclass
class ContextConsistencyReport:
    summary: dict
    source_qa_artifacts_read: List[str]
    warnings: List[str]
    errors: List[str]
    failed_cases: List[dict]
    warning_cases: List[dict]
    safety_confirmations: dict
    next_recommended_actions: List[str] = field(default_factory=list)
