"""Types for the accepted knowledge memory artifact.

All pattern extraction is deterministic and rule-based.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ItemKind = Literal["fact", "pattern"]

AcceptedItemType = Literal[
    "positive_cue",
    "typical_action",
    "typical_location",
    "semantic_frame_hint",
    "phrase_candidate_hint",
    "definition_pattern",
    "action_pattern",
    "location_pattern",
    "property_pattern",
    "relation_pattern",
    "causal_pattern",
]

AcceptedRiskLevel = Literal["low", "medium", "high"]

_MEMORY_VERSION = "accepted_knowledge_memory_v1"
_MODE = "artifact_only"


@dataclass
class ItemProvenance:
    source_artifact: str
    source_proposal_id: Optional[str] = None
    source_pattern_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "source_artifact": self.source_artifact,
            "source_proposal_id": self.source_proposal_id,
            "source_pattern_id": self.source_pattern_id,
        }


@dataclass
class AcceptedMemoryItem:
    item_id: str
    item_kind: str
    term: str
    sense: str
    item_type: str
    value: str
    source_id: str
    risk_level: str = "low"
    decision: str = "accepted_auto"
    provenance: ItemProvenance = field(default_factory=lambda: ItemProvenance(""))

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "item_kind": self.item_kind,
            "term": self.term,
            "sense": self.sense,
            "item_type": self.item_type,
            "value": self.value,
            "source_id": self.source_id,
            "risk_level": self.risk_level,
            "decision": self.decision,
            "provenance": self.provenance.to_dict(),
        }
