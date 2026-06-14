"""Dataclasses for Wiki Candidate Memory Overlay v1.

Overlay items are derived from ingestion v2 candidates. They are accepted
only for isolated entity QA overlay use. They must not replace or modify
accepted_knowledge_memory_v1.json.

safe_for_general_runtime is always False.
safe_for_entity_qa_overlay may be True if validation passes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

OverlayType = Literal[
    "overlay_entity",
    "overlay_definition",
    "overlay_relation",
    "overlay_context_link",
    "overlay_source_fact",
]

OverlayTrust = Literal[
    "overlay_candidate",
    "weak_context_only",
    "source_qualified_overlay_candidate",
]

OverlayRisk = Literal["low", "medium", "high"]
OverlayStability = Literal["stable", "semi_stable", "volatile"]

SAFE_FOR_GENERAL_RUNTIME: bool = False


@dataclass
class OverlayEntity:
    overlay_type: OverlayType = "overlay_entity"
    entity_id: str = ""
    label: str = ""
    aliases: list[str] = field(default_factory=list)
    entity_type: str = ""
    source_page: str = ""
    source_candidate_type: str = "entity_card"
    trust: OverlayTrust = "overlay_candidate"
    risk: OverlayRisk = "low"


@dataclass
class OverlayDefinition:
    overlay_type: OverlayType = "overlay_definition"
    subject: str = ""
    definition: str = ""
    predicate: str = "is_a"
    source_page: str = ""
    evidence_text: str = ""
    trust: OverlayTrust = "overlay_candidate"
    risk: OverlayRisk = "low"
    stability: OverlayStability = "stable"


@dataclass
class OverlayRelation:
    overlay_type: OverlayType = "overlay_relation"
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_page: str = ""
    evidence_text: str = ""
    trust: OverlayTrust = "overlay_candidate"
    risk: OverlayRisk = "medium"
    stability: OverlayStability = "semi_stable"


@dataclass
class OverlayContextLink:
    overlay_type: OverlayType = "overlay_context_link"
    source_page: str = ""
    surface: str = ""
    target: str = ""
    relation: str = "mentioned_with"
    strength: str = "weak"
    trust: OverlayTrust = "weak_context_only"


@dataclass
class OverlaySourceFact:
    overlay_type: OverlayType = "overlay_source_fact"
    subject: str = ""
    predicate: str = ""
    object: str = ""
    source_name: str = ""
    as_of: str = ""
    claim_type: str = "time_sensitive_estimate"
    source_page: str = ""
    evidence_text: str = ""
    requires_recheck: bool = True
    trust: OverlayTrust = "source_qualified_overlay_candidate"
    risk: OverlayRisk = "high"
    stability: OverlayStability = "volatile"


@dataclass
class OverlaySkipped:
    """Records one candidate that was filtered out by the overlay builder."""
    item_type: str = ""
    reason: str = ""
    label_or_subject: str = ""
