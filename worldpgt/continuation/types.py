"""Core dataclasses for the controlled continuation pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class SenseEntry:
    """One explicit sense of an ambiguous term, with lexical cues and allowed continuations."""

    term: str
    sense_id: str
    cues: list[str]
    continuations: list[str]
    continuation_templates: dict[str, list[str]] = field(default_factory=dict)
    trust: float = 1.0


@dataclass
class ContinuationPrompt:
    """A parsed prompt: the detected ambiguous term (if any) and its candidate sense ids."""

    prompt: str
    ambiguous_term: Optional[str]
    candidate_senses: list[str]


@dataclass
class ContinuationResult:
    """Full, auditable output of one controlled continuation."""

    prompt: str
    continuation: str
    ambiguous_term: Optional[str]
    selected_sense: Optional[str]
    confidence: float
    decision: Literal["continue", "audit", "suppress"]
    reasons: list[str] = field(default_factory=list)
    memory_hits: list[str] = field(default_factory=list)
    sense_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class ContinuationAuditRow:
    """One human-audited continuation outcome."""

    prompt: str
    continuation: str
    ambiguous_term: Optional[str]
    selected_sense: Optional[str]
    expected_sense: Optional[str]
    label: Literal["good", "bad", "unclear"]
    notes: str = ""
