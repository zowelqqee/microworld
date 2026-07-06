"""Dataclasses for low-trust community-context artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field


COMMUNITY_CONTEXT_TRUST = "community_context_only"
COMMUNITY_SOURCE_SYSTEM_REDDIT = "reddit"


@dataclass(frozen=True)
class RedditRecord:
    source_id: str
    source_kind: str
    subreddit: str
    title: str
    body: str
    author: str
    score: int | None = None
    permalink: str = ""
    created_utc: str = ""
    over_18: bool = False


@dataclass(frozen=True)
class CommunityContextItem:
    item_id: str
    source_system: str
    source_kind: str
    trust: str
    subreddit: str
    title: str
    text: str
    url: str
    score: int | None
    created_utc: str
    topic_terms: tuple[str, ...] = field(default_factory=tuple)
    flags: tuple[str, ...] = field(default_factory=tuple)
    risk: str = "medium"
    stability: str = "volatile"

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "source_system": self.source_system,
            "source_kind": self.source_kind,
            "trust": self.trust,
            "subreddit": self.subreddit,
            "title": self.title,
            "text": self.text,
            "url": self.url,
            "score": self.score,
            "created_utc": self.created_utc,
            "topic_terms": list(self.topic_terms),
            "flags": list(self.flags),
            "risk": self.risk,
            "stability": self.stability,
        }


@dataclass(frozen=True)
class CommunityQuarantineItem:
    source_id: str
    source_system: str
    reason: str
    text: str
    subreddit: str = ""
    url: str = ""

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "source_system": self.source_system,
            "reason": self.reason,
            "text": self.text,
            "subreddit": self.subreddit,
            "url": self.url,
        }


@dataclass(frozen=True)
class CommunitySearchResult:
    item: CommunityContextItem
    score: float
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "item": self.item.to_dict(),
            "score": self.score,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class CognitivePatternEvent:
    """A reusable cognitive pattern learned from non-factual context."""

    event_id: str
    kind: str
    topic: str
    pattern: str
    use_when: tuple[str, ...] = field(default_factory=tuple)
    avoid_when: tuple[str, ...] = field(default_factory=tuple)
    steps: tuple[str, ...] = field(default_factory=tuple)
    example_shape: str = ""
    confidence: str = "medium"
    source: str = "community_context"
    trust: str = "low_for_facts_high_for_style"
    graph_layers: tuple[str, ...] = field(default_factory=tuple)
    source_item_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_refs: tuple[str, ...] = field(default_factory=tuple)
    signal_count: int = 1
    factual_support_allowed: bool = False

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "kind": self.kind,
            "topic": self.topic,
            "pattern": self.pattern,
            "use_when": list(self.use_when),
            "avoid_when": list(self.avoid_when),
            "steps": list(self.steps),
            "example_shape": self.example_shape,
            "confidence": self.confidence,
            "source": self.source,
            "trust": self.trust,
            "graph_layers": list(self.graph_layers),
            "source_item_ids": list(self.source_item_ids),
            "evidence_refs": list(self.evidence_refs),
            "signal_count": self.signal_count,
            "factual_support_allowed": self.factual_support_allowed,
        }


@dataclass(frozen=True)
class CognitivePatternSearchResult:
    event: CognitivePatternEvent
    score: float
    matched_terms: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "event": self.event.to_dict(),
            "score": self.score,
            "matched_terms": list(self.matched_terms),
        }
