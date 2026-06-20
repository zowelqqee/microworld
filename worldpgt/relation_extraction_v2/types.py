"""Core dataclasses for Relation Extraction v2.

Plain dataclasses only. No I/O, no network, no ML.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

Confidence = Literal["high", "medium", "low"]
Stability = Literal["stable", "semi_stable", "volatile"]
Risk = Literal["low", "medium", "high"]
Directionality = Literal["forward", "reverse", "bidirectional", "unknown"]

# Relation types that v2 is allowed to produce.
ALLOWED_SEMI_STABLE_RELATIONS = frozenset({
    "founded_by",
    "founded",
    "develops",
    "developed_by",
    "manufactures",
    "produces",
    "product_of",
    "owned_by",
    "subsidiary_of",
    "parent_company_of",
    "headquartered_in",
    "industry",
    "operates",
    "created_by",
    "published_by",
    "service_of",
    "platform_of",
    "uses",
    "part_of",
})

ALLOWED_STABLE_RELATIONS = frozenset({
    "is_a",
    "type_of",
})

# Volatile relations: reflect real-time state that can change at any moment.
# Facts extracted with these relations carry stability="volatile" and must not
# be auto-accepted as stable memory without explicit human review.
ALLOWED_VOLATILE_RELATIONS = frozenset({
    "leader_of",   # Current leadership (CEO, president, head) — changes suddenly
    "valued_at",   # Financial valuation — changes with market conditions daily
})

ALLOWED_RELATIONS = (
    ALLOWED_SEMI_STABLE_RELATIONS
    | ALLOWED_STABLE_RELATIONS
    | ALLOWED_VOLATILE_RELATIONS
)

QUARANTINE_REASONS = frozenset({
    "weak_link_only",
    "current_or_live_claim",
    "volatile_claim",
    "private_or_sensitive",
    "unsupported_universal",
    "directionality_conflict",
    "entity_type_conflict",
    "low_confidence",
    "missing_explicit_evidence",
    "generic_entity_match",
    "conflict_existing_overlay",
    "ambiguous_relation",
})


@dataclass
class RelationExtractionEvidence:
    sentence: str
    pattern_id: str
    pattern_description: str
    subject_surface: str
    object_surface: str
    sentence_index: int
    paragraph_index: int

    def to_dict(self) -> dict:
        return {
            "sentence": self.sentence,
            "pattern_id": self.pattern_id,
            "pattern_description": self.pattern_description,
            "subject_surface": self.subject_surface,
            "object_surface": self.object_surface,
            "sentence_index": self.sentence_index,
            "paragraph_index": self.paragraph_index,
        }


@dataclass
class ExtractedRelationCandidate:
    id: str
    subject: str
    relation: str
    object: str
    confidence: Confidence
    stability: Stability
    risk: Risk
    source_title: str
    source_url: str
    retrieved_at: str
    raw_text_sha256: str
    evidence_sentence: str
    pattern_id: str
    directionality: Directionality
    requires_review: bool
    safe_for_overlay_delta: bool
    evidence: Optional[RelationExtractionEvidence] = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "confidence": self.confidence,
            "stability": self.stability,
            "risk": self.risk,
            "source_title": self.source_title,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at,
            "raw_text_sha256": self.raw_text_sha256,
            "evidence_sentence": self.evidence_sentence,
            "pattern_id": self.pattern_id,
            "directionality": self.directionality,
            "requires_review": self.requires_review,
            "safe_for_overlay_delta": self.safe_for_overlay_delta,
            "evidence": self.evidence.to_dict() if self.evidence else None,
            "notes": self.notes,
        }


@dataclass
class RelationExtractionQuarantineItem:
    id: str
    subject: str
    relation: str
    object: str
    reason: str
    source_title: str
    evidence_sentence: str
    pattern_id: str
    risk: Risk
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "reason": self.reason,
            "source_title": self.source_title,
            "evidence_sentence": self.evidence_sentence,
            "pattern_id": self.pattern_id,
            "risk": self.risk,
            "notes": self.notes,
        }


@dataclass
class RelationExtractionSummary:
    ready_docs_total: int = 0
    docs_processed: int = 0
    docs_failed: int = 0
    sentences_scanned: int = 0
    raw_relation_candidates_total: int = 0
    accepted_relation_candidates_total: int = 0
    quarantined_candidates_total: int = 0
    candidates_by_relation_type: dict = field(default_factory=dict)
    candidates_by_stability: dict = field(default_factory=dict)
    candidates_by_risk: dict = field(default_factory=dict)
    quarantine_by_reason: dict = field(default_factory=dict)
    duplicates_existing_count: int = 0
    conflicts_existing_count: int = 0
    safe_for_overlay_delta_count: int = 0
    network_calls: bool = False
    auto_ingest: bool = False
    auto_promote: bool = False
    trusted_memory_modified: bool = False
    accepted_overlay_modified: bool = False
    promoted_overlay_modified: bool = False
    snapshot_dry_run_overlay_modified: bool = False
    runtime_behavior_modified: bool = False
    safe_for_general_runtime: bool = False

    def to_dict(self) -> dict:
        return {
            "ready_docs_total": self.ready_docs_total,
            "docs_processed": self.docs_processed,
            "docs_failed": self.docs_failed,
            "sentences_scanned": self.sentences_scanned,
            "raw_relation_candidates_total": self.raw_relation_candidates_total,
            "accepted_relation_candidates_total": self.accepted_relation_candidates_total,
            "quarantined_candidates_total": self.quarantined_candidates_total,
            "candidates_by_relation_type": dict(self.candidates_by_relation_type),
            "candidates_by_stability": dict(self.candidates_by_stability),
            "candidates_by_risk": dict(self.candidates_by_risk),
            "quarantine_by_reason": dict(self.quarantine_by_reason),
            "duplicates_existing_count": self.duplicates_existing_count,
            "conflicts_existing_count": self.conflicts_existing_count,
            "safe_for_overlay_delta_count": self.safe_for_overlay_delta_count,
            "network_calls": self.network_calls,
            "auto_ingest": self.auto_ingest,
            "auto_promote": self.auto_promote,
            "trusted_memory_modified": self.trusted_memory_modified,
            "accepted_overlay_modified": self.accepted_overlay_modified,
            "promoted_overlay_modified": self.promoted_overlay_modified,
            "snapshot_dry_run_overlay_modified": self.snapshot_dry_run_overlay_modified,
            "runtime_behavior_modified": self.runtime_behavior_modified,
            "safe_for_general_runtime": self.safe_for_general_runtime,
        }


@dataclass
class RelationExtractionReport:
    summary: dict = field(default_factory=dict)
    top_accepted_candidates: List[dict] = field(default_factory=list)
    top_quarantined_candidates: List[dict] = field(default_factory=list)
    conflict_examples: List[dict] = field(default_factory=list)
    volatile_current_examples: List[dict] = field(default_factory=list)
    relation_type_examples: dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    recommended_next_actions: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "summary": self.summary,
            "top_accepted_candidates": self.top_accepted_candidates,
            "top_quarantined_candidates": self.top_quarantined_candidates,
            "conflict_examples": self.conflict_examples,
            "volatile_current_examples": self.volatile_current_examples,
            "relation_type_examples": self.relation_type_examples,
            "warnings": self.warnings,
            "errors": self.errors,
            "recommended_next_actions": self.recommended_next_actions,
        }
