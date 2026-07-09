"""Optional low-trust community context for Assistant Surface.

Community context is useful for phrasing, examples, and common concerns. It is
never factual support and is kept separate from overlay memory.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from worldpgt.community_context.reddit_engine import (
    query_community_context,
    render_community_context,
)
from worldpgt.community_context.cognitive_pattern_pump import (
    plan_answer_with_cognitive_patterns,
    query_cognitive_patterns,
)
from worldpgt.community_context.types import (
    CognitivePatternSearchResult,
    CommunitySearchResult,
)


class CommunityContextProvider(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[CommunitySearchResult]:
        """Return low-trust community-context matches for ``query``."""


class CognitivePatternProvider(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[CognitivePatternSearchResult]:
        """Return cognitive-pattern matches, never factual support."""

    def plan(self, query: str, *, max_patterns: int = 5) -> dict:
        """Return an answer-planning scaffold from cognitive patterns."""


@lru_cache(maxsize=8)
def _load_context_items_cached(path_str: str, mtime_ns: int, size: int) -> tuple[dict, ...]:
    del mtime_ns, size
    rows = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict))


def load_context_items(path: str | Path) -> list[dict]:
    p = Path(path)
    stat = p.stat()
    return list(_load_context_items_cached(str(p), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=8)
def _load_cognitive_pattern_events_cached(path_str: str, mtime_ns: int, size: int) -> tuple[dict, ...]:
    del mtime_ns, size
    rows = json.loads(Path(path_str).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return ()
    return tuple(row for row in rows if isinstance(row, dict))


def load_cognitive_pattern_events(path: str | Path) -> list[dict]:
    p = Path(path)
    stat = p.stat()
    return list(_load_cognitive_pattern_events_cached(str(p), stat.st_mtime_ns, stat.st_size))


class FileCommunityContextProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def search(self, query: str, *, max_results: int = 5) -> list[CommunitySearchResult]:
        items = load_context_items(self.path)
        return query_community_context(items, query, max_results=max_results)

    def count(self) -> int:
        return len(load_context_items(self.path))


class FileCognitivePatternProvider:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def search(self, query: str, *, max_results: int = 5) -> list[CognitivePatternSearchResult]:
        events = load_cognitive_pattern_events(self.path)
        return query_cognitive_patterns(events, query, max_results=max_results)

    def plan(self, query: str, *, max_patterns: int = 5) -> dict:
        events = load_cognitive_pattern_events(self.path)
        return plan_answer_with_cognitive_patterns(query, events, max_patterns=max_patterns)

    def count(self) -> int:
        return len(load_cognitive_pattern_events(self.path))


def render_context_answer(query: str, results: list[CommunitySearchResult]) -> str:
    return render_community_context(query, results)


def apply_community_tone(
    question: str,
    answer_text: str,
    *,
    support_kind: str,
    source_system: str,
) -> str:
    """Reserved hook for genuine forum-style phrasing shifts.

    Community context has no differentiated content to add for a plain
    supported answer today, so this is currently a pass-through: it never
    changes the factual claim, and it never wraps the answer in commentary
    that doesn't say anything the answer didn't already say.
    """

    del question, support_kind, source_system
    return answer_text


