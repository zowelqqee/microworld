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
    """Apply forum-style phrasing without changing the factual support.

    This is intentionally a surface pass. Community context is allowed to shape
    how the answer is presented, but the factual claim remains the exact answer
    already produced by overlay QA or live web search.
    """

    del question
    text = (answer_text or "").strip()
    if not text or _already_has_community_tone(text):
        return answer_text
    if source_system == "community_context":
        return answer_text
    if support_kind == "web_search_result" or source_system == "web_search":
        return _apply_community_tone_to_web_answer(text)
    return _apply_community_tone_to_supported_answer(text, source_system=source_system)


def _already_has_community_tone(text: str) -> bool:
    lowered = text.lower()
    return (
        lowered.startswith("short version:")
        or lowered.startswith("short version (live web):")
        or "style note: community" in lowered
        or "style note: shaped from low-trust community phrasing" in lowered
    )


def _source_label(source_system: str) -> str:
    labels = {
        "entity_qa": "Microworld memory",
        "cross_page_qa": "Microworld relation memory",
        "query_engine": "Microworld query memory",
        "context_pack": "Microworld policy context",
        "web_search": "live web search",
    }
    return labels.get(source_system or "", source_system or "the supported source")


def _apply_community_tone_to_supported_answer(text: str, *, source_system: str) -> str:
    source = _source_label(source_system)
    return "\n".join(
        [
            "Short version:",
            text,
            "",
            (
                "In plain terms: if you only need the takeaway, that is it. "
                "The phrasing is community-shaped; the fact still comes from "
                f"{source}."
            )
        ]
    )


def _apply_community_tone_to_web_answer(text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return text

    lead_lines: list[str] = []
    source_lines: list[str] = []
    fetch_line = ""
    for line in lines:
        if line.startswith("Based on a live web search"):
            fetch_line = line
        elif line.startswith("Source:") or line.startswith("Additional sources:"):
            source_lines.append(line)
        elif "live web search data" in line or "not Microworld memory" in line:
            source_lines.append(line)
        else:
            lead_lines.append(line)

    lead = " ".join(lead_lines).strip() or text
    result = [
        "Short version (live web):",
        lead,
        "",
    ]
    if fetch_line:
        result.extend([fetch_line, ""])
    result.extend(
        [
            "Plainly put: this is the current web-backed answer, so treat it "
            "as checkable rather than permanent.",
            "",
        ]
    )
    result.extend(source_lines)
    result.extend(
        [
            "",
            "The phrasing is community-shaped; the facts still come from live web search.",
        ]
    )
    return "\n".join(result).strip()
