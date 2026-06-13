"""Dataclasses and type literals for wiki-derived sentence/phrase patterns.

All patterns are extracted deterministically from curated source text
using explicit rules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PatternType = Literal[
    "definition",
    "location",
    "action",
    "property",
    "relation",
    "causal",
    "contrast",
]

PatternDecision = Literal["accepted_auto", "needs_review", "rejected_auto"]

RiskLevel = Literal["low", "medium", "high"]

# Semantic frames recognised by the pipeline.  Patterns referencing a frame
# that is NOT in this set are assigned decision=needs_review.
KNOWN_SEMANTIC_FRAMES: frozenset[str] = frozenset({
    "financial_transaction",
    "natural_geography",
    "animal_behavior",
    "sports_equipment_use",
    "bird_behavior",
    "heavy_machinery_operation",
    "music_performance",
    "natural_material",
    "seasonal_transition",
    "mechanical_device",
    "document_closure",
})

# Templates that are always rejected: no concrete semantic content.
GENERIC_TEMPLATES: frozenset[str] = frozenset({
    "{subject} did something",
    "{subject} was there",
    "{subject} moved",
    "{subject} was important",
    "{subject} continued",
    "{subject} had something",
    "{subject} used something",
    "{subject} was notable",
    "{subject} occurred",
})


@dataclass
class PatternCandidate:
    """One deterministically extracted sentence/phrase pattern candidate.

    Linked to a specific (term, sense) pair and semantic frame.  Never
    auto-applied to live memory — output is a dry-run artifact only.
    """

    pattern_id: str
    source_id: str
    term: str
    sense: str
    pattern_type: PatternType
    semantic_frame: str
    template: str
    required_slots: list[str] = field(default_factory=list)
    example_source_text: str = ""
    example_realization: str = ""
    risk_level: RiskLevel = "low"
    decision: PatternDecision = "needs_review"
    reason: str = ""
