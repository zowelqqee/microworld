"""Deterministic semantic frame extraction for the v2 renderer.

This is a small, explicit, weightless layer. It converts an already-selected
sense into a coarse semantic frame (actor / intent / connector) using simple
lexical lookups. No ML, no neural weights, no probabilistic models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import re

from worldpgt.continuation.realization import classify_prompt_ending

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Obvious actor nouns we recognise in prompts, in no particular priority.
_ACTOR_NOUNS = [
    "customer",
    "client",
    "teller",
    "fisherman",
    "boat",
    "children",
    "ranger",
    "player",
    "batter",
    "seal",
    "crane",
    "operator",
    "foreman",
    "spring",
    "rain",
    "band",
    "guitarist",
    "drummer",
    "climber",
]

# sense_id -> coarse intent label. Stable, explicit mapping.
_SENSE_INTENT = {
    "financial_institution": "transaction",
    "river_edge": "river_edge_activity",
    "animal": "animal_behavior",
    "sports_equipment": "sports_action",
    "closure_stamp": "document_closure",
    "machine": "construction_action",
    "bird": "bird_behavior",
    "season": "seasonal_change",
    "coil": "mechanical_motion",
    "stone": "stone_motion",
    "music": "music_performance",
}

# Coarse location cues, scanned deterministically for the frame's location slot.
_LOCATION_CUES = {
    "river": "river",
    "stream": "river",
    "shore": "shore",
    "ocean": "ocean",
    "water": "water",
    "lake": "lake",
    "marsh": "marsh",
    "cave": "cave",
    "construction": "construction_site",
    "site": "construction_site",
    "building": "building",
    "stage": "stage",
    "concert": "stage",
    "garden": "garden",
    "envelope": "envelope",
    "document": "document",
}


@dataclass
class SemanticFrame:
    """Coarse semantic frame for one selected sense of one prompt."""

    term: str
    sense_id: str
    actor: Optional[str]
    action: Optional[str]
    object: Optional[str]
    location: Optional[str]
    intent: Optional[str]
    connector_type: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class FrameCandidate:
    """One rendered candidate continuation for a frame, with its ranking metadata."""

    text: str
    frame: SemanticFrame
    renderer_name: str
    score: float
    reasons: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _first_actor(tokens: list[str], term: str) -> Optional[str]:
    """First recognised actor noun in the prompt that is not the ambiguous term."""
    for token in tokens:
        if token == term:
            continue
        if token in _ACTOR_NOUNS:
            return token
    return None


def _first_location(tokens: list[str]) -> Optional[str]:
    for token in tokens:
        if token in _LOCATION_CUES:
            return _LOCATION_CUES[token]
    return None


def build_semantic_frame(
    prompt: str,
    term: str,
    sense_id: str,
    evidence: list[str],
) -> SemanticFrame:
    """Build a coarse, deterministic semantic frame for the selected sense."""
    tokens = _tokens(prompt)
    term_lower = term.lower()
    return SemanticFrame(
        term=term_lower,
        sense_id=sense_id,
        actor=_first_actor(tokens, term_lower),
        action=None,
        object=None,
        location=_first_location(tokens),
        intent=_SENSE_INTENT.get(sense_id),
        connector_type=classify_prompt_ending(prompt),
        evidence=list(evidence),
    )
