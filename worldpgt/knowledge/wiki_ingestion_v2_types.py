"""Dataclasses and type literals for Wikipedia/Wikidata Ingestion v2.

This pipeline is read-only / dry-run. Every structure produced here is a
*candidate* memory item, never trusted runtime memory. No module in this
package writes to sense_memory.py, accepted memory artifacts, or any
generation behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from worldpgt.knowledge.entity_types import CanonicalEntityType

# ---------------------------------------------------------------------------
# Shared literal vocabularies
# ---------------------------------------------------------------------------

ItemType = Literal[
    "entity_card",
    "definition_claim",
    "relation_claim",
    "context_link",
    "source_qualified_fact",
]

EntityType = CanonicalEntityType

Stability = Literal["stable", "semi_stable", "volatile"]

Risk = Literal["low", "medium", "high"]

TemporalClass = Literal["historical", "snapshot", "aggregate", "derived"]

# Status is always "candidate" for this pipeline — never trusted runtime memory.
Status = Literal["candidate"]

LinkStrength = Literal["weak"]

RelationPredicate = Literal[
    "known_for",
    "leader_of",
    "founded",
    "produces",
    "develops",
    "publishes",
    "estimates",
    "related_to",
    "part_of",
    "is_a",
]


# ---------------------------------------------------------------------------
# Input record shape
# ---------------------------------------------------------------------------


@dataclass
class WikiLink:
    """One outgoing hyperlink from a page's lead paragraph."""

    surface: str
    target: str


@dataclass
class WikiSource:
    """Provenance for a curated page record."""

    source_id: str
    source_type: str
    retrieved_at: str
    source_url: Optional[str] = None


@dataclass
class WikiPageRecord:
    """A curated Wikipedia-style page record (local fixture only)."""

    page_id: str
    title: str
    lead_paragraph: str
    source: WikiSource
    wikidata_id: Optional[str] = None
    entity_type_hint: Optional[str] = None
    infobox: dict[str, str] = field(default_factory=dict)
    links: list[WikiLink] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Output candidate items
# ---------------------------------------------------------------------------


@dataclass
class EntityCard:
    """One canonical entity record per page. Candidate only."""

    item_type: ItemType = "entity_card"
    entity_id: str = ""
    label: str = ""
    aliases: list[str] = field(default_factory=list)
    entity_type: EntityType = "other"
    source_page: str = ""
    risk: Risk = "low"
    status: Status = "candidate"


@dataclass
class DefinitionClaim:
    """A simple first-sentence `is_a` definition. Candidate only."""

    item_type: ItemType = "definition_claim"
    subject: str = ""
    predicate: str = "is_a"
    object: str = ""
    source_page: str = ""
    evidence_text: str = ""
    stability: Stability = "stable"
    temporal_class: TemporalClass = "historical"
    risk: Risk = "low"
    status: Status = "candidate"


@dataclass
class RelationClaim:
    """A deterministic relation claim. Candidate only."""

    item_type: ItemType = "relation_claim"
    subject: str = ""
    predicate: RelationPredicate = "related_to"
    object: str = ""
    source_page: str = ""
    evidence_text: str = ""
    stability: Stability = "semi_stable"
    temporal_class: TemporalClass = "historical"
    as_of: str = ""
    risk: Risk = "medium"
    status: Status = "candidate"


@dataclass
class ContextLink:
    """A weak contextual association derived from a hyperlink. Candidate only.

    A hyperlink alone is NOT a trusted fact. strength is always "weak".
    """

    item_type: ItemType = "context_link"
    source_page: str = ""
    surface: str = ""
    target: str = ""
    relation: str = "mentioned_with"
    strength: LinkStrength = "weak"
    status: Status = "candidate"


@dataclass
class SourceQualifiedFact:
    """A time-sensitive, source-dependent fact. Candidate only.

    These must never be promoted to stable relation claims.
    """

    item_type: ItemType = "source_qualified_fact"
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_name: str = ""
    as_of: str = ""
    claim_type: str = "time_sensitive_estimate"
    temporal_class: TemporalClass = "snapshot"
    source_page: str = ""
    evidence_text: str = ""
    stability: Stability = "volatile"
    risk: Risk = "high"
    status: Status = "candidate"
    requires_recheck: bool = True


# ---------------------------------------------------------------------------
# Review gate
# ---------------------------------------------------------------------------

ReviewSeverity = Literal["error", "warning"]


@dataclass
class ReviewIssue:
    """One problem found by the review gate for a single candidate."""

    severity: ReviewSeverity
    code: str
    message: str
    item_type: str
    item_index: int


# Safety invariant: this pipeline produces candidates only.
SAFE_FOR_RUNTIME_MEMORY: bool = False
