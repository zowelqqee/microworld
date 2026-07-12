"""Claude-backed web search provider — DuckDuckGo replacement.

DuckDuckGo's scraped endpoints are the most fragile provider in this
pipeline (see ``duckduckgo.py``'s docstring: historically 0/40 useful on
real natural-language questions, and in some network environments its
circuit breaker trips on the very first call because the anti-bot system
blocks the request outright). Rather than keep tuning a scraper against a
moving target, this provider replaces it with Claude's own server-side
``web_search`` tool: an agentic search that runs entirely on Anthropic's
infrastructure (the search + page fetch happen server-side, not from this
process), so it isn't subject to the same IP-level blocking, and it can
follow up on ambiguous results the way a scraped single-page fetch cannot.

Graceful degradation is the whole point of slotting this in as a drop-in
``WebSearchProvider``: if the ``anthropic`` package isn't installed or no
credentials are configured, ``search()`` returns ``[]`` immediately (no
network attempt, no exception) exactly like an empty result from any other
provider — ``CompositeSearchProvider`` just moves on. This mirrors the
project's stance that web search is an opt-in slow fallback, never a hard
dependency (see the "project identity: speed" note this codebase follows).

Model choice: Haiku 4.5 (fastest/cheapest tier), not Opus. This provider
only ever answers a single-fact question from a short snippet -- there is no
reasoning depth for the extra cost of a bigger model to buy here, and the
composite provider already races this against Wikipedia/Wikidata under a
wall-clock deadline, so latency matters more than raw capability.
"""

from __future__ import annotations

import os

from worldpgt.assistant_surface.web_search import WebSearchResult

try:  # pragma: no cover - exercised indirectly via _has_sdk()
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None  # type: ignore[assignment]

DEFAULT_MODEL = "claude-haiku-4-5"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TIMEOUT_SEC = 20.0
DEFAULT_MAX_SEARCH_USES = 3


def _has_credentials() -> bool:
    """Best-effort, side-effect-free check for a usable credential.

    Doesn't attempt to validate an ``ant auth login`` profile on disk (that
    would mean shelling out or duplicating the SDK's resolution order) --
    just the common cases so we can skip the network call entirely when we
    already know it can't work, rather than paying a doomed request's
    latency before catching the resulting exception.
    """
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    config_dir = os.environ.get(
        "ANTHROPIC_CONFIG_DIR",
        os.path.expanduser("~/.config/anthropic"),
    )
    return os.path.isdir(os.path.join(config_dir, "credentials"))


class ClaudeWebSearchProvider:
    """``WebSearchProvider`` backed by Claude's server-side web search tool."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout_sec: float = DEFAULT_TIMEOUT_SEC,
        client=None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec
        self._client = client

    def _get_client(self):
        if self._client is not None:
            return self._client
        if anthropic is None or not _has_credentials():
            return None
        self._client = anthropic.Anthropic().with_options(timeout=self._timeout_sec)
        return self._client

    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        q = (query or "").strip()
        if not q:
            return []
        client = self._get_client()
        if client is None:
            return []

        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                tools=[{
                    "type": "web_search_20260209",
                    "name": "web_search",
                    "max_uses": DEFAULT_MAX_SEARCH_USES,
                }],
                messages=[{
                    "role": "user",
                    "content": (
                        f"Search the web and answer this question in 1-3 concise "
                        f"sentences, stating the specific fact asked for: {q}"
                    ),
                }],
            )
        except Exception:
            return []

        return self._to_results(response, max_results=max_results)

    @staticmethod
    def _to_results(response, *, max_results: int) -> list[WebSearchResult]:
        answer_text = ""
        title = "Claude web search"
        url = ""
        for block in getattr(response, "content", None) or []:
            block_type = getattr(block, "type", None)
            if block_type == "text":
                text = getattr(block, "text", "") or ""
                if text.strip():
                    answer_text = f"{answer_text} {text}".strip() if answer_text else text.strip()
            elif block_type == "web_search_tool_result" and not url:
                content = getattr(block, "content", None)
                if isinstance(content, list) and content:
                    first = content[0]
                    url = getattr(first, "url", "") or ""
                    result_title = getattr(first, "title", "") or ""
                    if result_title:
                        title = result_title

        if not answer_text:
            return []
        return [WebSearchResult(title=title, snippet=answer_text, url=url)][:max_results]
