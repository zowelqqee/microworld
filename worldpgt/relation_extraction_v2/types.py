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

# Polarity is the sign of the predicate, not a separate predicate.  It lets
# "X shall be entitled to a patent unless C" be stored as ``entitled_to`` with
# ``polarity="negate"`` and the condition C held separately, instead of welding
# the whole rule into a long predicate string (see the conditional-edge pilot,
# artifacts/legal_domain_pilot_v1/conditional_edge_v1/).
Polarity = Literal["affirm", "negate"]

# A clause is either a fact that must obtain (``factual``) or a restriction on
# the context in which the rule applies (``scope`` — "for purposes of ... under
# subsection (a)(2)").  Truth is preserved without the distinction; the tag only
# makes it inspectable.
ClauseKind = Literal["factual", "scope"]

# ``entity`` is a named entity (the default the whole graph assumed until now).
# ``class_subject`` is a description of a class of things/persons/situations —
# "whoever knowingly threatens ...", "any invention made in outer space" — which
# is a legitimate statutory subject but is *not* a named entity.  A class subject
# never auto-admits; it is always a review-only proposal.
NodeKind = Literal["entity", "class_subject"]

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
    "provides",
    "enables",
    "used_for",
    "works_by",
    "part_of",
    "alias",
    "based_at",
    "ceased_operations",
    "construction_started",
    "filed_for_bankruptcy",
    "first_released",
    "funded_by",
    "has_facility",
    "hosted_flight_to",
    "introduced",
    "located_in",
    "marketed_as",
    "merged_with",
    "offers",
    "runs_on",
    "supports",
    "variant_of",
    # Ad-hoc pilot predicate types (methodology / citation / topic families).
    "trained_on",
    "based_on",
    "extends",
    "about",
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

# Relations that locate an entity and therefore read naturally as a *subject
# noun-phrase post-modifier* rather than as a standalone predicate sentence:
# "a robotics company headquartered in Boston" rather than "It is
# headquartered in Boston." as its own choppy line.
#
# This is a facts-layer role declaration -- the worldpgt analogue of the
# ``kind="object_link"`` tag poetry_lab's ingest assigns to a preposition link
# ("дверь в коридор"). It marks a *role in composition*, so it lives with the
# relation vocabulary here; the reasoning layer selects a link to bundle by
# this membership (see ``entity_qa.synthesis_engine.synthesize``), and the
# speech layer derives the actual participial surface from the learned phrase
# fragment. Ownership relations are intentionally excluded: they entangle with
# the founded/owned-object enrichment and do not form a clean locative
# participle, so they stay ordinary predicate sentences for now.
SUBJECT_LOCATIVE_RELATIONS = frozenset({
    "headquartered_in",
    "located_in",
    "based_at",
    "based_in",
})

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


@dataclass(frozen=True)
class ConditionClause:
    """One evidence-anchored fragment attached to a relation.

    Used for a condition, an exception, or a disjunctive object alternative.
    ``evidence_span`` keeps the same literal-span verification discipline as
    every other node in the graph: each clause points at verbatim source text.
    """

    text: str
    evidence_span: str
    kind: str = "factual"  # ClauseKind for conditions/exceptions; "" for objects

    def to_dict(self) -> dict:
        payload = {"text": self.text, "evidence_span": self.evidence_span}
        if self.kind and self.kind != "factual":
            payload["kind"] = self.kind
        return payload


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
    # --- Optional conditional-edge extension --------------------------------
    # Every field below defaults to the pre-extension semantics, so an ordinary
    # simple edge is byte-identical in to_dict() and behaves exactly as before.
    # ``conditions`` is a conjunction (all must hold); ``exceptions`` is a
    # disjunction of defeaters; ``object_alternatives`` holds a disjunctive
    # consequence ("deemed X or subject to Y") without splitting it into two
    # edges that would read conjunctively.
    conditions: List["ConditionClause"] = field(default_factory=list)
    exceptions: List["ConditionClause"] = field(default_factory=list)
    object_alternatives: List["ConditionClause"] = field(default_factory=list)
    polarity: Polarity = "affirm"
    subject_kind: NodeKind = "entity"

    def is_simple(self) -> bool:
        """True when this edge is indistinguishable from a pre-extension edge."""
        return (
            not self.conditions
            and not self.exceptions
            and not self.object_alternatives
            and self.polarity == "affirm"
            and self.subject_kind == "entity"
        )

    def to_dict(self) -> dict:
        payload = {
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
        # Emit the extension keys only when they carry non-default content, so
        # existing overlays and their serialization are unaffected.
        if self.conditions:
            payload["conditions"] = [c.to_dict() for c in self.conditions]
        if self.exceptions:
            payload["exceptions"] = [c.to_dict() for c in self.exceptions]
        if self.object_alternatives:
            payload["object_alternatives"] = [c.to_dict() for c in self.object_alternatives]
        if self.polarity != "affirm":
            payload["polarity"] = self.polarity
        if self.subject_kind != "entity":
            payload["subject_kind"] = self.subject_kind
        return payload


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
