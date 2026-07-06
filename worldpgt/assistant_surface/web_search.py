"""Optional live web-search types/rendering for Assistant Surface.

This module contains no network client. Concrete HTTP providers live outside
``assistant_surface`` so the core surface can keep its no-network-import
contract unless an explicit provider is enabled by the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
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


def _truncate(text: str, limit: int = 400) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") + "."


def render_web_answer(
    query: str,
    results: list[WebSearchResult],
    *,
    fetched_at: str | None = None,
    from_cache: bool = False,
    lead: str | None = None,
) -> str:
    """Render a direct-sounding live answer that still discloses it is
    unverified web data, never accepted/promoted Microworld memory.

    The lead line states the answer, then names every source so the claim can
    be checked. Pass ``lead`` with an extracted answer sentence to state the
    specific fact; otherwise the top result's snippet (an intro/abstract) is
    truncated as the lead. Pass the original fetch time as ``fetched_at`` (not
    ``None``) when rendering a cache hit, so staleness stays visible instead of
    being silently reset to "now".
    """

    if not results:
        return ""

    primary, *rest = results
    lead = _clean(lead) if lead else _truncate(_clean(primary.snippet))
    fetched_label = fetched_at or datetime.now(timezone.utc).isoformat()
    cache_note = " (cached)" if from_cache else ""

    lines = [
        f"Based on a live web search as of {fetched_label}{cache_note}:",
        lead,
        f"Source: {primary.title} — {primary.url}",
    ]
    if rest:
        extra = "; ".join(f"{r.title} — {r.url}" for r in rest)
        lines.append(f"Additional sources: {extra}")
    lines.append(
        "This is live web search data, not Microworld memory, and may change "
        "over time — treat it as unverified and re-check before relying on it."
    )
    return "\n".join(lines)
