"""Core dataclasses for Entity QA v1 pipeline.

Rule-based and deterministic only. No ML. No embeddings. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

EntityQAIntent = Literal[
    "define_entity",
    "relation_lookup",
    "link_explanation",
    "source_fact_lookup",
    "unknown_or_unsupported",
]

EntityQADecision = Literal["answer", "audit"]


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
