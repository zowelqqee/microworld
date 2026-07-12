"""Core dataclasses for Cross-page Entity QA v1.

Rule-based and deterministic only. No ML. No embeddings. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

CrossPageIntent = Literal[
    "connection_path",
    "relation_chain",
    "source_qualified_list",
    "weak_link_list_or_policy",
    "unsupported_or_too_far",
]

CrossPageDecision = Literal["answer", "audit"]


@dataclass
class CrossPageQuestion:
    question: str
    intent: CrossPageIntent
    source: Optional[str]
    target: Optional[str]
    is_current_query: bool = False
    is_universal_query: bool = False


@dataclass
class PathEdge:
    """One directed overlay relation traversed in a path (canonical direction)."""

    subject: str
    predicate: str
    object: str
    stability: str


@dataclass
class CrossPageEvidence:
    relation_edges_used: list[str] = field(default_factory=list)
    weak_links_used: list[str] = field(default_factory=list)
    source_facts_used: list[str] = field(default_factory=list)


@dataclass
class CrossPagePlan:
    question: CrossPageQuestion
    decision: CrossPageDecision
    audit_reason: Optional[str]
    evidence: CrossPageEvidence
    render_template: str
    render_args: dict
    confidence: float


@dataclass
class ValidationResult:
    is_correct: bool
    quality_flagged: bool
    quality_reason: str
