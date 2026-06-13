"""Core dataclasses for the AnswerPlanner v1 QA pipeline.

Rule-based and deterministic only.  No ML libraries.
No learned model parameters.  No language-model inference.  No back-prop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

QAIntent = Literal[
    "define_sense",
    "distinguish_senses",
    "explain_cue",
    "classify_context",
    "unknown_or_ambiguous",
]
QADecision = Literal["answer", "audit"]


@dataclass
class AnalyzedQuestion:
    """Parsed representation of a QA question."""

    question: str
    intent: QAIntent
    term: Optional[str]
    target_sense: Optional[str]
    second_sense: Optional[str]
    cues_in_question: list[str]
    inline_context: Optional[str]
    underconstrained: bool


@dataclass
class EvidenceBundle:
    """Explicit evidence used to build an answer."""

    facts_used: list[str] = field(default_factory=list)
    patterns_used: list[str] = field(default_factory=list)
    cues_used: list[str] = field(default_factory=list)
    provider_items_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "facts_used": self.facts_used,
            "patterns_used": self.patterns_used,
            "cues_used": self.cues_used,
            "provider_items_used": self.provider_items_used,
        }


@dataclass
class AnswerPlan:
    """Internal plan produced by AnswerPlanner before rendering."""

    analyzed: AnalyzedQuestion
    decision: QADecision
    audit_reason: Optional[str]
    evidence: EvidenceBundle
    render_template: str
    render_args: dict
    confidence: float


@dataclass
class QAResult:
    """Final output of one QA pipeline run."""

    row_id: str
    question: str
    intent: str
    term: Optional[str]
    target_sense: Optional[str]
    decision: QADecision
    answer: str
    evidence: EvidenceBundle
    audit_reason: Optional[str]
    confidence: float
    trace: dict
    quality_flagged: bool = False
    quality_reason: str = ""
