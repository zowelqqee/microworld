"""Deterministic data model for Candidate Knowledge Request v1.

Plain dataclasses only. No I/O, no runtime coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------- #
# Activation statuses. There is intentionally NO auto_apply / auto_ingest /
# auto_promote / trusted_memory_write status: nothing here is ever applied.
# --------------------------------------------------------------------------- #
REQUEST_ONLY = "request_only"
REQUIRES_LOCAL_SOURCE_DOC = "requires_local_source_doc"
REQUIRES_SELF_INGESTION = "requires_self_ingestion"
REQUIRES_QUARANTINE_REVIEW = "requires_quarantine_review"
REQUIRES_PROMOTION_GATE = "requires_promotion_gate"
REQUIRES_REGRESSION_GATE = "requires_regression_gate"

ALLOWED_ACTIVATION_STATUSES = frozenset(
    {
        REQUEST_ONLY,
        REQUIRES_LOCAL_SOURCE_DOC,
        REQUIRES_SELF_INGESTION,
        REQUIRES_QUARANTINE_REVIEW,
        REQUIRES_PROMOTION_GATE,
        REQUIRES_REGRESSION_GATE,
    }
)

# --------------------------------------------------------------------------- #
# Request categories.
# --------------------------------------------------------------------------- #
REQ_MISSING_ENTITY_RELATION = "missing_entity_relation"
REQ_MISSING_PRODUCT_RELATION = "missing_product_relation"
REQ_MISSING_ORGANIZATION_RELATION = "missing_organization_relation"
REQ_MISSING_SOURCE_QUALIFIED_FACT = "missing_source_qualified_fact"
REQ_MISSING_POLICY_EXPLANATION = "missing_policy_explanation"
REQ_UNSUPPORTED_OR_TOO_FAR_PATH = "unsupported_or_too_far_path"

REQUEST_CATEGORIES = frozenset(
    {
        REQ_MISSING_ENTITY_RELATION,
        REQ_MISSING_PRODUCT_RELATION,
        REQ_MISSING_ORGANIZATION_RELATION,
        REQ_MISSING_SOURCE_QUALIFIED_FACT,
        REQ_MISSING_POLICY_EXPLANATION,
        REQ_UNSUPPORTED_OR_TOO_FAR_PATH,
    }
)

# Opportunity categories that must never become stable knowledge requests.
DANGEROUS_OPPORTUNITY_CATEGORIES = frozenset(
    {
        "current_claim_guard_case",
        "relation_inversion_guard_case",
        "weak_link_safety_case",
        "volatile_review_candidate",
        "unsupported_universal_guard_case",
        "adversarial_test_candidate",
    }
)

RISK_LOW = "low"
RISK_MEDIUM = "medium"
RISK_HIGH = "high"


@dataclass
class MissingKnowledgeInput:
    """A normalized missing-knowledge opportunity fed into the classifier."""

    id: str
    category: str
    question: str
    current_decision: str
    current_behavior: str
    source_artifact: str
    priority: Optional[str] = None
    observed_decision: Optional[str] = None
    expected_policy: Optional[str] = None
    reason: Optional[str] = None
    suggested_next_step: Optional[str] = None


@dataclass
class CandidateKnowledgeRequest:
    """A proposal-only request describing missing knowledge and how to add it."""

    id: str
    source_opportunity_id: str
    question: str
    current_decision: str
    current_behavior: str
    missing_knowledge_type: str
    needed_evidence_type: str
    candidate_subject: str
    candidate_relation: str
    candidate_object: str
    suggested_relation_path: str
    suggested_source_topic: str
    suggested_source_doc_title: str
    risk_level: str
    safety_constraints: List[str]
    must_remain_audit_until_supported: bool
    activation_status: str
    # The request category mirrors missing_knowledge_type but is kept explicit
    # for grouping / reporting convenience.
    category: str = ""


@dataclass
class SourceDocumentRequest:
    """A proposal-only request to author a local source document later."""

    id: str
    topic: str
    suggested_title: str
    related_request_ids: List[str]
    entities_to_cover: List[str]
    relations_to_verify: List[str]
    claims_to_avoid: List[str]
    required_safety_notes: List[str]
    next_pipeline_step: str


@dataclass
class SkippedInput:
    """A dangerous / unsafe opportunity that was NOT turned into a request."""

    opportunity_id: str
    question: str
    opportunity_category: str
    reason: str
    note: str


@dataclass
class KnowledgeRequestReport:
    summary: dict
    source_artifacts_read: List[str]
    warnings: List[str]
    errors: List[str]
    top_source_document_requests: List[dict]
    safety_notes: List[dict]
    safety_confirmations: dict
    next_recommended_actions: List[str] = field(default_factory=list)
