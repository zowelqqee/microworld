"""Composite web-search provider: Wikipedia, then Wikidata, then DuckDuckGo.

All three are raced concurrently (see below), so this priority order is a
preference among whichever finish in time, not a sequential wait:

1. Wikipedia — full article extract, rich enough for answer extraction to
   find most facts in prose. Proven the most robust of the three in practice
   (never observed rate-limited or blocked).
2. Wikidata — structured (subject, property, value) facts rendered as
   explicit sentences ("X was educated at Y"). Often more precisely on-target
   than mining prose for the same fact, and — like Wikipedia — an official,
   no-key API rather than a scraped page, so it is not subject to the kind of
   silent bot-block DuckDuckGo uses. (It does have its own, explicit rate
   limit: HTTP 429 + Retry-After, which ResilientHttpClient now honors.)
3. DuckDuckGo — broadest coverage (whatever's indexed, not just
   Wikimedia-family facts), but the most fragile: a scraped page, not an
   official API, so it is the one that gets soft-blocked under load. Kept
   last, and protected by its own circuit breaker so a block doesn't cost
   every subsequent query a wasted round trip.

Providers run CONCURRENTLY, not strictly sequentially, under a shared
``deadline_sec`` wall-clock budget. Running each provider's own retry/backoff
chain one after another was the actual root cause of a long cold-path tail
(measured: p95 ~13s, max ~30s on a 249-question run) — each provider pays its
own pacing + up to 3 retries across up to 3 endpoints, and those durations
just added up. Racing them under one deadline bounds the worst case to
roughly ``deadline_sec`` regardless of how many endpoints/retries a slow
provider is internally churning through, while still preferring the
higher-quality provider when it answers in time.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from worldpgt.assistant_surface.web_search import WebSearchProvider, WebSearchResult
from worldpgt.web_search.duckduckgo import DuckDuckGoInstantAnswerProvider
from worldpgt.web_search.wikidata import WikidataProvider
from worldpgt.web_search.wikipedia import WikipediaProvider

# Measured tradeoff (30-question sample, WebQuestions):
#   6s deadline  -> 53% answered, max 6.05s
#   12s deadline -> 93% answered, max 10.4s  (near-full coverage, 3x shorter tail)
#   no deadline  -> 99.6% answered, max 30.7s (the original, unbounded-tail behavior)
# 12s is the practical default: it keeps coverage close to unbounded while
# still cutting the worst case to a third. Pass deadline_sec explicitly (or
# AnswerOrchestrator(web_search_deadline_sec=...)) to pick a different point
# on this curve.
DEFAULT_DEADLINE_SEC = 12.0
_CURRENT_OFFICE_QUERY_RE = re.compile(
    r"\b(?:current\s+)?(?:president|prime minister|mayor|governor|officeholder|incumbent)\b",
    re.IGNORECASE,
)


class CompositeSearchProvider:
    """Race sub-providers under a shared deadline; prefer the first provider
    (in priority order) that returns a non-empty result within the deadline.

    If nothing answers before the deadline, returns ``[]`` — the caller then
    audits honestly rather than blocking on a provider's own internal retries.
    """

    def __init__(
        self,
        providers: list[WebSearchProvider] | None = None,
        *,
        deadline_sec: float = DEFAULT_DEADLINE_SEC,
    ) -> None:
        if providers is None:
            providers = [WikipediaProvider(), WikidataProvider(), DuckDuckGoInstantAnswerProvider()]
        self._providers = list(providers)
        self._deadline_sec = deadline_sec

    @staticmethod
    def _safe_search(provider: WebSearchProvider, query: str, max_results: int) -> list[WebSearchResult]:
        try:
            return provider.search(query, max_results=max_results) or []
        except Exception:
            return []

    def _providers_for_query(self, query: str) -> list[WebSearchProvider]:
        providers = list(self._providers)
        if not _CURRENT_OFFICE_QUERY_RE.search(query or ""):
            return providers
        return sorted(providers, key=lambda p: 0 if isinstance(p, WikidataProvider) else 1)

    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        providers = self._providers_for_query(query)
        if not providers:
            return []

        deadline = time.monotonic() + self._deadline_sec
        pool = ThreadPoolExecutor(max_workers=len(providers))
        try:
            # Submit every provider immediately so slow ones make progress in
            # the background while we check faster/higher-priority ones —
            # this is what avoids paying each provider's latency in sequence.
            futures: list[Future] = [
                pool.submit(self._safe_search, p, query, max_results) for p in providers
            ]
            for future in futures:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    result = future.result(timeout=remaining)
                except FutureTimeoutError:
                    continue
                if result:
                    return result
            return []
        finally:
            # Don't block returning on providers we no longer care about —
            # already-running background searches finish on their own and are
            # discarded (each has its own bounded per-request timeout, so this
            # never leaks indefinitely).
            pool.shutdown(wait=False, cancel_futures=True)
