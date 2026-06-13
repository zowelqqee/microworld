"""Dataclasses for the knowledge ingestion pipeline v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class CuratedSnippet:
    """One curated wiki-style text snippet for a specific term/sense."""

    source_id: str
    source_type: str
    term: str
    sense: str
    title: str
    text: str
    source_url: Optional[str]
    trust_seed: float


FactType = Literal[
    "positive_cue",
    "anti_cue",
    "typical_action",
    "typical_location",
    "part",
    "object",
    "semantic_frame_hint",
    "phrase_candidate_hint",
]


@dataclass
class FactCandidate:
    """One candidate fact extracted from a curated snippet."""

    source_id: str
    term: str
    sense: str
    fact_type: FactType
    value: str
    confidence: float = 1.0
    is_broad: bool = False
    is_multiword: bool = False


@dataclass
class NormalizedFact:
    """A FactCandidate after normalization and conflict analysis."""

    source_id: str
    term: str
    sense: str
    fact_type: FactType
    value: str
    confidence: float
    is_broad: bool
    is_multiword: bool
    rejected: bool = False
    rejection_reason: str = ""
    conflict_senses: list[str] = field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"


@dataclass
class ProposedUpdate:
    """The structured memory fields proposed for one term/sense."""

    positive_cues: list[str] = field(default_factory=list)
    anti_cues: list[str] = field(default_factory=list)
    typical_actions: list[str] = field(default_factory=list)
    typical_locations: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    semantic_frame_hints: list[str] = field(default_factory=list)
    phrase_candidate_hints: list[str] = field(default_factory=list)


@dataclass
class ProposalEvidence:
    """Audit trail for one memory update proposal."""

    source_titles: list[str] = field(default_factory=list)
    matched_phrases: list[str] = field(default_factory=list)
    rejected_broad_cues: list[str] = field(default_factory=list)
    conflicting_cues: list[str] = field(default_factory=list)


SAFETY_CHECKS: list[str] = [
    "does_not_lower_thresholds",
    "does_not_weaken_validators",
    "wrong_continue_count_remains_0",
    "semantic_quality_flagged_remains_0",
]


@dataclass
class MemoryUpdateProposal:
    """A reviewable candidate memory update. Never auto-applied to sense_memory."""

    proposal_id: str
    source_ids: list[str]
    term: str
    sense: str
    proposal_type: str
    risk_level: Literal["low", "medium", "high"]
    recommended_action: Literal["auto_safe_later", "human_review", "reject"]
    proposed_update: ProposedUpdate
    evidence: ProposalEvidence
    safety_checks_required: list[str] = field(
        default_factory=lambda: list(SAFETY_CHECKS)
    )
