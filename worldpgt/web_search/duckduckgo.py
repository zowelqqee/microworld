"""DuckDuckGo search provider for optional live QA fallback.

DuckDuckGo's free endpoints (lite HTML, full HTML) are rate-limited and
reject bot-looking clients: a non-browser User-Agent gets throttled or
served empty pages, and rapid bursts trip a soft block that returns HTTP 202
/ empty bodies. This client is built to survive that:

- realistic rotating browser User-Agents;
- a minimum interval between outbound requests (burst pacing), kept
  per-endpoint so racing them concurrently isn't serialized by a shared gate;
- retry with exponential backoff on transient failures / empty responses;
- lite and full HTML are raced concurrently, not tried one after another —
  measured: sequential lite-then-html cost up to ~3.4s on some queries;
  racing bounds this to roughly the slower of the two, not their sum.

The Instant Answer JSON endpoint is deliberately NOT queried here: measured
on 40 real natural-language questions that reach this fallback (i.e. ones
Wikipedia already had no article for), it returned a usable result 0/40
times. It only helps for clean single-entity queries ("Elon Musk"), and
Wikipedia already resolves those better via a full article extract — by the
time a query reaches DuckDuckGo in the composite pipeline, it is essentially
guaranteed to be exactly the kind of natural-language question this endpoint
cannot answer, so querying it was pure wasted latency (one more network
round trip and pacing turn) with no offsetting value.

Circuit breaker: under sustained load DuckDuckGo soft-blocks by serving its
own generic homepage (HTTP 200, ~14KB of real HTML, valid-looking) instead of
an error or empty body — so it looks like "no results" per query rather than
"provider unavailable", and every single query keeps paying a full round
trip (and retries) against a backend that is not going to answer anything
until the block lifts. Once that signature is seen, the breaker trips: for
`cooldown_sec` this provider returns `[]` immediately with no network call at
all, instead of re-discovering the block query by query. Any real
(non-blocked-looking) response resets the breaker immediately, whether or not
it happened to contain matches for that particular query.

Everything is stdlib-only and dependency-injectable (``opener`` / ``sleep``)
so retry, pacing, backoff, and the breaker are testable without real network
access.
"""

from __future__ import annotations

import re
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import urlopen

from worldpgt.assistant_surface.web_search import WebSearchResult
from worldpgt.web_search.http import ResilientHttpClient


DEFAULT_BREAKER_COOLDOWN_SEC = 300.0  # 5 minutes; a starting guess, not yet
# empirically tuned against how long a real DDG soft-block actually lasts.

# The block signature: DDG's own homepage, not a query-specific results page.
_BLOCKED_CANONICAL = 'href="https://duckduckgo.com/"'
_BLOCKED_TITLE_RE = re.compile(r"<title>\s*DuckDuckGo\s*</title>", re.IGNORECASE)


def _is_blocked_page(html: str) -> bool:
    """True if ``html`` is DuckDuckGo's generic homepage rather than a
    query-specific results (or genuine no-results) page."""
    if not html:
        return False
    return _BLOCKED_CANONICAL in html and bool(_BLOCKED_TITLE_RE.search(html))


class DuckDuckGoInstantAnswerProvider:
    """Resilient stdlib-only DuckDuckGo client (rotating UA, pacing, retry,
    and a circuit breaker for sustained soft-blocks)."""

    def __init__(
        self,
        *,
        timeout_sec: float = 6.0,
        user_agent: str | None = None,
        max_retries: int = 3,
        backoff_base_sec: float = 0.8,
        min_interval_sec: float = 1.2,
        breaker_cooldown_sec: float = DEFAULT_BREAKER_COOLDOWN_SEC,
        opener=urlopen,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        def _client() -> ResilientHttpClient:
            return ResilientHttpClient(
                timeout_sec=timeout_sec,
                user_agent=user_agent,
                max_retries=max_retries,
                backoff_base_sec=backoff_base_sec,
                min_interval_sec=min_interval_sec,
                opener=opener,
                sleep=sleep,
                monotonic=monotonic,
            )

        # Separate client per endpoint: each keeps its own independent pacing
        # state, so racing them concurrently isn't throttled by one shared
        # rate limiter (that would defeat the point of racing).
        self._lite_http = _client()
        self._html_http = _client()
        self._monotonic = monotonic
        self._breaker_cooldown_sec = breaker_cooldown_sec
        self._blocked_until: float | None = None

    def _circuit_open(self) -> bool:
        return self._blocked_until is not None and self._monotonic() < self._blocked_until

    def _record_blocked(self) -> None:
        self._blocked_until = self._monotonic() + self._breaker_cooldown_sec

    def _record_healthy(self) -> None:
        self._blocked_until = None

    @property
    def circuit_open(self) -> bool:
        """Whether the breaker is currently tripped (for observability/tests)."""
        return self._circuit_open()

    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        q = quote_plus((query or "").strip())
        if not q:
            return []
        if self._circuit_open():
            # Known-blocked: skip the network entirely rather than
            # re-discovering the same block on every query.
            return []

        def try_lite() -> list[WebSearchResult]:
            body = self._lite_http.get_text(f"https://lite.duckduckgo.com/lite/?q={q}")
            if not body:
                return []
            if _is_blocked_page(body):
                self._record_blocked()
                return []
            self._record_healthy()
            return _results_from_lite_html(body, max_results=max_results)

        def try_html() -> list[WebSearchResult]:
            body = self._html_http.get_text(f"https://html.duckduckgo.com/html/?q={q}")
            if not body:
                return []
            if _is_blocked_page(body):
                self._record_blocked()
                return []
            self._record_healthy()
            return _results_from_html(body, max_results=max_results)

        pool = ThreadPoolExecutor(max_workers=2)
        try:
            pending = {pool.submit(try_lite), pool.submit(try_html)}
            while pending:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    try:
                        result = future.result()
                    except Exception:
                        result = []
                    if result:
                        return result
            return []
        finally:
            pool.shutdown(wait=False, cancel_futures=True)


_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r".*?"
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_LITE_RESULT_RE = re.compile(
    r"<a[^>]+href=\"(?P<url>[^\"]+)\"[^>]+class='result-link'[^>]*>(?P<title>.*?)</a>"
    r".*?"
    r"<td[^>]+class='result-snippet'[^>]*>(?P<snippet>.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG_RE.sub(" ", text or ""))).strip()


# Sponsored/tracking redirects DuckDuckGo's html/lite endpoints render inline
# with organic results (same result__a / result-link markup), e.g.
# "https://duckduckgo.com/y.js?ad_domain=...&ad_provider=bingv7aa..." or
# "https://www.bing.com/aclick?ld=...". These are ad clickthroughs, not
# sources, so they must never be cited as a "Source:" in a live answer.
_AD_HOST_PATH_PATTERNS = (
    ("duckduckgo.com", "/y.js"),
    ("bing.com", "/aclick"),
)
_AD_HOST_MARKERS = ("doubleclick.net", "googleadservices.com", "googlesyndication.com")
_AD_QUERY_MARKERS = ("ad_domain=", "ad_provider=", "ad_type=")


def _is_ad_or_tracking_url(url: str) -> bool:
    if not url:
        return False
    parsed = urlparse(url)
    host = parsed.netloc.lower()
    path = parsed.path.lower()
    for ad_host, ad_path in _AD_HOST_PATH_PATTERNS:
        if ad_host in host and path.startswith(ad_path):
            return True
    if any(marker in host for marker in _AD_HOST_MARKERS):
        return True
    query = parsed.query.lower()
    return any(marker in query for marker in _AD_QUERY_MARKERS)


def _results_from_html(html: str, *, max_results: int) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    for match in _RESULT_RE.finditer(html or ""):
        title = _strip_html(match.group("title"))
        snippet = _strip_html(match.group("snippet"))
        url = unescape(match.group("url")).strip()
        if not title or not snippet or not url:
            continue
        if _is_ad_or_tracking_url(url):
            continue
        results.append(WebSearchResult(title=title, snippet=snippet, url=url))
        if len(results) >= max_results:
            break
    return results


def _normalize_duckduckgo_url(url: str) -> str:
    url = unescape(url or "").strip()
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _results_from_lite_html(html: str, *, max_results: int) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    for match in _LITE_RESULT_RE.finditer(html or ""):
        title = _strip_html(match.group("title"))
        snippet = _strip_html(match.group("snippet"))
        raw_url = unescape(match.group("url")).strip()
        if not title or not snippet or not raw_url:
            continue
        if _is_ad_or_tracking_url(raw_url):
            continue
        url = _normalize_duckduckgo_url(raw_url)
        if _is_ad_or_tracking_url(url):
            continue
        results.append(WebSearchResult(title=title, snippet=snippet, url=url))
        if len(results) >= max_results:
            break
    return results
