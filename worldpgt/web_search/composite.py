"""Composite web-search provider: Wikipedia, then Wikidata, then Claude.

All three are raced concurrently (see below), so this priority order is a
preference among whichever finish in time, not a sequential wait:

1. Wikipedia — full article extract, rich enough for answer extraction to
   find most facts in prose. Proven the most robust of the three in practice
   (never observed rate-limited or blocked, prior to the shared-rate-limiter
   incident noted below).
2. Wikidata — structured (subject, property, value) facts rendered as
   explicit sentences ("X was educated at Y"). Often more precisely on-target
   than mining prose for the same fact, and — like Wikipedia — an official,
   no-key API rather than a scraped page.
3. Claude web search — an agentic search that runs server-side on
   Anthropic's infrastructure, not a scraped page from this process. Used
   last since it costs an API call and (unlike Wikipedia/Wikidata) is not
   free. Replaces the previous DuckDuckGo scraper, which was the most
   fragile provider here (historically useful on ~0/40 real questions, and
   in some network environments outright blocked from the first request).
   Gracefully returns ``[]`` with no network attempt when no Anthropic
   credentials are configured — see ``claude_search.py``.

Providers run CONCURRENTLY, not strictly sequentially, under a shared
``deadline_sec`` wall-clock budget. Running each provider's own retry/backoff
chain one after another was the actual root cause of a long cold-path tail
(measured: p95 ~13s, max ~30s on a 249-question run) — each provider pays its
own pacing + up to 3 retries across up to 3 endpoints, and those durations
just added up. Racing them under one deadline bounds the worst case to
roughly ``deadline_sec`` regardless of how many endpoints/retries a slow
provider is internally churning through, while still preferring the
higher-quality provider when it answers in time.

Query routing: Wikipedia gets the regex-reformed ``intent.search_query``
(its full-text index benefits from keyword reformation); Wikidata and Claude
both get the raw original question, since both do their own real
understanding of it (structured entity matching, and language understanding
respectively) rather than needing our query-reformation heuristics — see
``search()`` below and ``test_composite_query_routing_v1.py``.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import Future, ThreadPoolExecutor, wait

from worldpgt.assistant_surface.web_search import WebSearchProvider, WebSearchResult
from worldpgt.web_search.claude_search import ClaudeWebSearchProvider
from worldpgt.web_search.http import SharedRateLimiter
from worldpgt.web_search.query_intent import build_web_query_intent, filter_and_rank_results
from worldpgt.web_search.wikidata import WikidataProvider
from worldpgt.web_search.wikipedia import WikipediaProvider

# Shared across every default Wikipedia/Wikidata client this module creates
# (both of Wikipedia's own concurrent sub-fetches, plus Wikidata's client) so
# they collectively -- not just individually -- stay under a safe request
# rate against Wikimedia's infrastructure. See SharedRateLimiter's docstring
# for the concrete rate-limiting incident (a benchmark run) that motivated
# this: per-client pacing alone doesn't stop *separate* client instances from
# bursting against the same backend at once.
WIKIMEDIA_RATE_LIMIT_INTERVAL_SEC = 0.5

# Measured tradeoff (30-question sample, WebQuestions):
#   6s deadline  -> 53% answered, max 6.05s
#   12s deadline -> 93% answered, max 10.4s  (near-full coverage, 3x shorter tail)
#   no deadline  -> 99.6% answered, max 30.7s (the original, unbounded-tail behavior)
# 12s is the practical default: it keeps coverage close to unbounded while
# still cutting the worst case to a third. Pass deadline_sec explicitly (or
# AnswerOrchestrator(web_search_deadline_sec=...)) to pick a different point
# on this curve.
DEFAULT_DEADLINE_SEC = 12.0
# A short preference window for higher-priority providers. If Wikipedia is
# already close to returning, keep its richer extract; if it is stuck in
# retries/backoff, let a lower-priority relevant Wikidata/DDG result answer
# instead of burning the whole question deadline.
PRIORITY_GRACE_SEC = 0.8
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
            wikimedia_rate_limiter = SharedRateLimiter(min_interval_sec=WIKIMEDIA_RATE_LIMIT_INTERVAL_SEC)
            providers = [
                WikipediaProvider(rate_limiter=wikimedia_rate_limiter),
                WikidataProvider(rate_limiter=wikimedia_rate_limiter),
                ClaudeWebSearchProvider(),
            ]
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

        intent = build_web_query_intent(query)
        deadline = time.monotonic() + self._deadline_sec
        pool = ThreadPoolExecutor(max_workers=len(providers))
        try:
            # Submit every provider immediately so slow ones make progress in
            # the background while we check faster/higher-priority ones —
            # this is what avoids paying each provider's latency in sequence.
            futures: list[tuple[int, Future]] = []
            for provider in providers:
                # Wikipedia's full-text search benefits from the regex-based
                # keyword reformation (it's a bare keyword index, not language
                # understanding) -- but that reformation is also where
                # entity-resolution bugs live (e.g. "what capital of austria?"
                # -> "capital of austria capital"; "who played on the
                # jeffersons?" -> the entity itself gets stripped by a
                # relation-tail heuristic that assumes verb-after-entity
                # order). Wikidata and Claude both do their own real
                # understanding of the question (structured entity matching,
                # and actual language understanding respectively), so handing
                # them our fragile regex output instead of the original
                # question only imports our bugs into providers that don't
                # have them -- give both the raw question instead.
                provider_query = (
                    intent.search_query
                    if isinstance(provider, WikipediaProvider)
                    else query
                )
                future = pool.submit(self._safe_search, provider, provider_query, max_results)
                futures.append((len(futures), future))

            pending: set[Future] = {future for _priority, future in futures}
            priorities = {future: priority for priority, future in futures}
            buffered: dict[Future, list[WebSearchResult]] = {}

            while pending or buffered:
                for future in list(pending):
                    if not future.done():
                        continue
                    pending.remove(future)
                    result = future.result()
                    if result:
                        _intent, filtered = filter_and_rank_results(query, result)
                        if filtered:
                            buffered[future] = filtered[:max_results]

                if buffered:
                    best_future = min(buffered, key=lambda f: priorities[f])
                    best_priority = priorities[best_future]
                    higher_pending = [
                        f for f in pending
                        if priorities[f] < best_priority
                    ]
                    if not higher_pending:
                        return buffered[best_future]

                    grace_remaining = min(PRIORITY_GRACE_SEC, max(0.0, deadline - time.monotonic()))
                    if grace_remaining <= 0:
                        return buffered[best_future]
                    done, _not_done = wait(higher_pending, timeout=grace_remaining)
                    for future in done:
                        pending.discard(future)
                        result = future.result()
                        if result:
                            _intent, filtered = filter_and_rank_results(query, result)
                            if filtered:
                                buffered[future] = filtered[:max_results]
                    best_future = min(buffered, key=lambda f: priorities[f])
                    return buffered[best_future]

                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                done, _not_done = wait(pending, timeout=remaining)
                if not done:
                    break
            return []
        finally:
            # Don't block returning on providers we no longer care about —
            # already-running background searches finish on their own and are
            # discarded (each has its own bounded per-request timeout, so this
            # never leaks indefinitely).
            pool.shutdown(wait=False, cancel_futures=True)
