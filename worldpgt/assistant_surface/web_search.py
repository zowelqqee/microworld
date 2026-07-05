"""Optional live web-search types/rendering for Assistant Surface.

This module contains no network client. Concrete HTTP providers live outside
``assistant_surface`` so the core surface can keep its no-network-import
contract unless an explicit provider is enabled by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Protocol


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    snippet: str
    url: str


class WebSearchProvider(Protocol):
    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        """Return source-bearing web results for ``query``."""


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text.strip(" \t\r\n")


def _results_from_instant_answer(data: dict, *, max_results: int) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    heading = _clean(str(data.get("Heading") or ""))
    abstract = _clean(str(data.get("AbstractText") or data.get("Answer") or ""))
    abstract_url = _clean(str(data.get("AbstractURL") or ""))
    if abstract and abstract_url:
        results.append(WebSearchResult(title=heading or "DuckDuckGo instant answer", snippet=abstract, url=abstract_url))

    def add_topic(topic: dict) -> None:
        text = _clean(str(topic.get("Text") or ""))
        first_url = _clean(str(topic.get("FirstURL") or ""))
        if text and first_url:
            title = text.split(" - ", 1)[0].strip() or "Related result"
            results.append(WebSearchResult(title=title, snippet=text, url=first_url))

    for topic in data.get("RelatedTopics") or []:
        if len(results) >= max_results:
            break
        if not isinstance(topic, dict):
            continue
        if isinstance(topic.get("Topics"), list):
            for nested in topic["Topics"]:
                if len(results) >= max_results:
                    break
                if isinstance(nested, dict):
                    add_topic(nested)
        else:
            add_topic(topic)

    deduped: list[WebSearchResult] = []
    seen: set[str] = set()
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        deduped.append(result)
        if len(deduped) >= max_results:
            break
    return deduped


def render_web_answer(query: str, results: list[WebSearchResult]) -> str:
    """Render volatile search results without claiming accepted-memory support."""

    if not results:
        return ""
    lines = [
        "Live web search result, not Microworld memory:",
        f"Query: {query}",
    ]
    for idx, result in enumerate(results, start=1):
        snippet = result.snippet
        if len(snippet) > 320:
            snippet = snippet[:320].rsplit(" ", 1)[0].rstrip(" ,.;:") + "."
        lines.append(f"{idx}. {result.title}: {snippet} Source: {result.url}")
    return "\n".join(lines)
