"""Typed structures for the schema induction layer.

All dataclasses are frozen so artifacts are immutable once produced. Field
names are intentionally generic — none of them encode a domain predicate.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocumentRecord:
    """A single input document."""

    doc_id: str
    title: str
    url: str
    text: str


@dataclass(frozen=True)
class SentenceRecord:
    """One sentence segmented from a document."""

    sentence_id: str       # e.g. "d1:s2"
    doc_id: str
    index: int             # 0-based sentence index within the document
    text: str


@dataclass(frozen=True)
class EntityMention:
    """A surface entity mention discovered in the corpus.

    ``context_terms`` are observed words that co-occur with / head the mention
    (e.g. "visa", "permit", "species"). They drive local type induction —
    there is no global rigid enum of types.
    """

    mention_id: str
    surface: str
    normalized: str
    doc_ids: tuple[str, ...]
    sentence_ids: tuple[str, ...]
    context_terms: tuple[str, ...]
    type_hints: tuple[str, ...]
    occurrences: int


@dataclass(frozen=True)
class RawClaim:
    """A surface relation extracted from one sentence.

    ``relation_surface`` is the observed verb/phrase ("requires", "migrates to",
    "was founded by") — NOT a canonical predicate ("requires_document",
    "founded_by"). ``modifiers`` carries auxiliary surface info (cause, time,
    condition, ...) discovered alongside the main relation.
    """

    claim_id: str
    subject: str
    relation_surface: str
    object: str | None
    sentence: str
    source_doc_id: str
    source_sentence_id: str
    modifiers: dict[str, str]
    extraction_method: str
    confidence: float


@dataclass(frozen=True)
class ArgumentFrame:
    """A raw claim lifted into generic semantic roles."""

    frame_id: str
    claim_ids: tuple[str, ...]
    trigger: str
    roles: dict[str, str]
    role_types: dict[str, str]
    domain_hint: str | None
    confidence: float


@dataclass(frozen=True)
class RelationFamily:
    """A cluster of frames sharing relation structure."""

    family_id: str
    canonical_label: str
    surface_forms: tuple[str, ...]
    roles: tuple[str, ...]
    role_type_profile: dict[str, tuple[str, ...]]
    example_claim_ids: tuple[str, ...]
    evidence_count: int
    source_doc_count: int
    promotion_status: str   # generated | promoted | rejected
    confidence: float
    frame_ids: tuple[str, ...] = field(default_factory=tuple)
    rejection_reason: str | None = None


@dataclass(frozen=True)
class LocalType:
    """A type induced from observed context terms / roles (not a global enum)."""

    type_id: str
    label: str
    members: tuple[str, ...]
    context_terms: tuple[str, ...]
    induced_from_roles: tuple[str, ...]
    confidence: float


@dataclass(frozen=True)
class PromotionDecision:
    """Outcome of running promotion gates on a relation family or local type."""

    target_id: str
    target_kind: str        # relation_family | local_type
    status: str             # generated | promoted | rejected
    passed: tuple[str, ...]
    failed: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True)
class SchemaInductionResult:
    """Everything produced by one schema induction run."""

    documents: tuple[DocumentRecord, ...]
    sentences: tuple[SentenceRecord, ...]
    entities: tuple[EntityMention, ...]
    claims: tuple[RawClaim, ...]
    frames: tuple[ArgumentFrame, ...]
    families: tuple[RelationFamily, ...]
    local_types: tuple[LocalType, ...]
    decisions: tuple[PromotionDecision, ...]
    summary: dict
