"""Core dataclasses for Entity QA v1 pipeline.

Rule-based and deterministic only. No ML. No embeddings. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

UnknownPosition = Literal["subject", "object", "relation", "path", "intersection"]
SemanticQueryType = Literal[
    "lookup",
    "inverse",
    "aggregative",
    "comparative",
    "is_a",
    "definition",
    "filtered_lookup",
    "open_synthesis",
    "multi_fact",
]

EntityQAIntent = Literal[
    "define_entity",
    "relation_lookup",
    "link_explanation",
    "source_fact_lookup",
    "open_synthesis",
    "unknown_or_unsupported",
]

# Confidence tiers used by the synthesis layer (layer 3).
SynthesisTier = Literal["VERIFIED", "SNAPSHOT", "INFERRED", "UNKNOWN"]

EntityQADecision = Literal["answer", "audit", "no"]


@dataclass
class SemanticQuery:
    entity_a: Optional[str]
    entity_b: Optional[str]
    relation_intent: Optional[str]
    unknown_position: UnknownPosition
    query_type: SemanticQueryType
    confidence: float
    filter_predicate: str | None = None
    filter_object: str | None = None

    def to_dict(self) -> dict:
        return {
            "entity_a": self.entity_a,
            "entity_b": self.entity_b,
            "relation_intent": self.relation_intent,
            "unknown_position": self.unknown_position,
            "query_type": self.query_type,
            "confidence": self.confidence,
            "filter_predicate": self.filter_predicate,
            "filter_object": self.filter_object,
        }


@dataclass
class AnalyzedEntityQuestion:
    question: str
    intent: EntityQAIntent
    subject: Optional[str]
    predicate_hint: Optional[str]
    secondary_entity: Optional[str]
    source_hint: Optional[str]
    is_current_query: bool
    is_unsupported: bool
    semantic_query: Optional[SemanticQuery] = None


@dataclass
class EntityQAEvidence:
    overlay_items_used: list[str] = field(default_factory=list)
    source_facts_used: list[str] = field(default_factory=list)
    weak_context_links_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "overlay_items_used": self.overlay_items_used,
            "source_facts_used": self.source_facts_used,
            "weak_context_links_used": self.weak_context_links_used,
        }


@dataclass
class EntityQAPlan:
    analyzed: AnalyzedEntityQuestion
    decision: EntityQADecision
    audit_reason: Optional[str]
    evidence: EntityQAEvidence
    render_template: str
    render_args: dict
    confidence: float


@dataclass
class SynthesisFactGroup:
    """One group of synthesized facts sharing a predicate and confidence tier.

    ``kind`` records how the fact was obtained relative to the synthesized
    subject:
      - ``"forward_relation"``  — subject is the relation *subject*  (SpaceX develops X)
      - ``"inverse_relation"``  — subject is the relation *object*   (X founded SpaceX)
      - ``"snapshot"``          — a dated, source-qualified estimate (net worth)
    ``objects`` always lists the *other* side of the relation as displayed.
    """

    kind: str
    predicate: str
    objects: list[str]
    tier: str  # "VERIFIED" | "SNAPSHOT"
    source_name: Optional[str] = None
    as_of: Optional[str] = None
    rule: Optional[str] = None
    confidence: Optional[float] = None
    chain: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "predicate": self.predicate,
            "objects": list(self.objects),
            "tier": self.tier,
            "source_name": self.source_name,
            "as_of": self.as_of,
            "rule": self.rule,
            "confidence": self.confidence,
            "chain": [list(step) for step in self.chain],
        }


@dataclass
class SynthesisEnrichment:
    """A single verified fact about an entity *related to* the synthesized subject.

    When the subject founds / owns / leads another entity ``object``, the engine
    pulls one key fact about that object (its definition, or a ranking) so the
    narrative can connect the two — "X founded Y, an aerospace manufacturer" —
    rather than naming Y bare. The fact is always backed by an overlay item on
    ``object``; the connector ("founded") is a relation the subject already has.
    """

    object: str  # the related entity Y being described
    note: str  # a verified descriptor of Y (definition / ranking), no article
    relation: str  # predicate linking subject -> Y (founded, owned_by, ...)
    kind: str = "definition"  # "definition" | "ranking"
    entity_type: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "object": self.object,
            "note": self.note,
            "relation": self.relation,
            "kind": self.kind,
            "entity_type": self.entity_type,
        }


@dataclass
class SynthesisAnswer:
    """Structured result of synthesizing everything known about one entity.

    The synthesis engine never invents facts: every clause is backed by an
    overlay item. ``matched`` is False when no entity could be resolved (even
    via keyword overlap), in which case the planner audits instead of answering.
    """

    subject: Optional[str]
    matched: bool
    match_kind: str  # "exact" | "keyword" | "none"
    definition: Optional[str]
    entity_type: Optional[str]
    groups: list[SynthesisFactGroup] = field(default_factory=list)
    unknown_notes: list[str] = field(default_factory=list)
    candidate_entities: list[str] = field(default_factory=list)
    enrichment: Optional[SynthesisEnrichment] = None
    # One locative relation the reasoning layer chose to fold into the subject's
    # noun phrase ("a robotics company headquartered in Boston") instead of a
    # separate predicate sentence. Still present in ``groups`` for inspection;
    # the speech layer consumes whichever one it can actually surface as a
    # participle and drops the duplicate, so a fold never loses a fact.
    subject_locative: Optional[SynthesisFactGroup] = None

    @property
    def verified_count(self) -> int:
        n = 1 if self.definition else 0
        for g in self.groups:
            if g.tier == "VERIFIED":
                n += len(g.objects)
        return n

    @property
    def snapshot_count(self) -> int:
        return sum(
            len(g.objects) for g in self.groups if g.tier == "SNAPSHOT"
        )

    @property
    def inferred_count(self) -> int:
        return sum(
            len(g.objects) for g in self.groups if g.tier == "INFERRED"
        )

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "matched": self.matched,
            "match_kind": self.match_kind,
            "definition": self.definition,
            "entity_type": self.entity_type,
            "groups": [g.to_dict() for g in self.groups],
            "unknown_notes": list(self.unknown_notes),
            "candidate_entities": list(self.candidate_entities),
            "enrichment": self.enrichment.to_dict() if self.enrichment else None,
            "subject_locative": self.subject_locative.to_dict() if self.subject_locative else None,
            "verified_count": self.verified_count,
            "snapshot_count": self.snapshot_count,
            "inferred_count": self.inferred_count,
        }


@dataclass
class EntityQAResult:
    row_id: str
    question: str
    intent: str
    subject: Optional[str]
    decision: EntityQADecision
    answer: str
    evidence: EntityQAEvidence
    audit_reason: Optional[str]
    confidence: float
    is_correct: bool = False
    quality_flagged: bool = False
    quality_reason: str = ""
