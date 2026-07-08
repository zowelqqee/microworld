"""Shared resilient HTTP client for live web-search providers.

Free public endpoints throttle bot-looking or high-volume clients — either
silently (DuckDuckGo serves its own generic homepage instead of an error) or
explicitly (Wikidata returns a proper HTTP 429 with a ``Retry-After`` header,
e.g. "wait 38 seconds"). This client survives both with rotating browser
User-Agents, request pacing, retry-with-backoff, and — when the server
bothers to say so — honoring ``Retry-After`` instead of guessing a shorter
wait and getting 429'd again immediately.

Stdlib-only and fully dependency-injectable (``opener`` / ``sleep`` /
``monotonic``) so pacing, retry, and backoff are testable without network.
"""

from __future__ import annotations

import itertools
import random
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# Realistic desktop-browser User-Agents. Rotated to avoid looking scripted.
USER_AGENTS = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64; rv:123.0) Gecko/20100101 Firefox/123.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.3 Safari/605.1.15",
)

# Statuses worth retrying: 202 (still computing / soft-block), 429 (rate
# limited), and any 5xx.
RETRYABLE_STATUSES = frozenset({202, 429, 500, 502, 503, 504})

# Cap on how long a single retry will honor a server-provided Retry-After
# value. Real-world observed value from Wikidata: 24-38s. Composite-level
# deadlines (e.g. CompositeSearchProvider) already bound total wall-clock
# time across providers, so this only needs to avoid one retry alone eating
# an unreasonable chunk of that budget.
MAX_RETRY_AFTER_SEC = 8.0


def _parse_retry_after(value: str | None) -> float | None:
    """Parse a Retry-After header value (integer seconds form only — the
    HTTP-date form is rare on these APIs and not worth the parsing surface)."""
    if not value:
        return None
    try:
        seconds = float(value.strip())
    except ValueError:
        return None
    return seconds if seconds >= 0 else None


class SharedRateLimiter:
    """Pacing gate shared across multiple :class:`ResilientHttpClient`
    instances that hit the same rate-limited backend, so independent
    clients/threads don't collectively exceed a safe aggregate request rate
    even though each already paces itself individually.

    Concretely: Wikipedia and Wikidata are separate providers (and Wikipedia
    itself now fires two concurrent sub-requests -- full-question and
    bare-entity -- see ``wikipedia.py``), each with its own per-instance
    pacing. Per-instance pacing alone doesn't stop those *different*
    instances from all dispatching at once, and Wikimedia's rate limit is
    almost certainly enforced per source IP across the whole family of
    endpoints, not per API separately. A 250-question benchmark run against
    a shared/sandboxed egress IP demonstrated this concretely: raw ``curl``
    calls bypassing this codebase entirely still came back HTTP 429 for
    several minutes afterward -- confirming the rate limit is real and
    triggered by aggregate request volume, not a bug in retry/relevance
    logic. Passing one instance of this limiter into every Wikimedia-family
    client closes that gap by serializing actual dispatch across all of them
    to a single minimum interval, regardless of how many client instances or
    concurrent threads are involved.
    """

    def __init__(
        self,
        *,
        min_interval_sec: float,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        self._min_interval_sec = min_interval_sec
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.Lock()
        self._last_call_ts: float | None = None

    def acquire(self) -> None:
        """Block (if needed) until at least ``min_interval_sec`` has passed
        since the last acquire across ALL callers sharing this instance."""
        with self._lock:
            if self._last_call_ts is not None:
                remaining = self._min_interval_sec - (self._monotonic() - self._last_call_ts)
                if remaining > 0:
                    self._sleep(remaining)
            self._last_call_ts = self._monotonic()


class ResilientHttpClient:
    """GET text over HTTP with UA rotation, pacing, and retry/backoff."""

    def __init__(
        self,
        *,
        timeout_sec: float = 6.0,
        user_agent: str | None = None,
        max_retries: int = 3,
        backoff_base_sec: float = 0.8,
        min_interval_sec: float = 1.2,
        rate_limiter: SharedRateLimiter | None = None,
        opener=urlopen,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        self.timeout_sec = timeout_sec
        # A fixed user_agent pins the pool to one value (handy for tests);
        # otherwise rotate through the realistic pool.
        self._ua_cycle = itertools.cycle([user_agent] if user_agent else USER_AGENTS)
        self.max_retries = max(1, max_retries)
        self.backoff_base_sec = backoff_base_sec
        self.min_interval_sec = min_interval_sec
        # When a SharedRateLimiter is given, it replaces this client's own
        # local pacing -- see SharedRateLimiter's docstring for why per-
        # instance pacing alone isn't enough when several client instances
        # hit the same rate-limited backend.
        self._rate_limiter = rate_limiter
        self._opener = opener
        self._sleep = sleep
        self._monotonic = monotonic
        self._last_request_ts: float | None = None

    def _pace(self) -> None:
        """Sleep just enough to keep >= min_interval_sec between requests."""
        if self._rate_limiter is not None:
            self._rate_limiter.acquire()
            return
        if self._last_request_ts is not None and self.min_interval_sec > 0:
            remaining = self.min_interval_sec - (self._monotonic() - self._last_request_ts)
            if remaining > 0:
                self._sleep(remaining)
        self._last_request_ts = self._monotonic()

    def get_text(self, url: str, *, extra_headers: dict | None = None) -> str | None:
        """Fetch ``url`` as text, or None on a non-retryable/exhausted failure.

        Transient failures (retryable HTTP status, URL/timeout errors, empty
        body) are retried with exponential backoff + jitter.
        """
        for attempt in range(self.max_retries):
            self._pace()
            headers = {
                "User-Agent": next(self._ua_cycle),
                "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
            if extra_headers:
                headers.update(extra_headers)
            retry_after: float | None = None
            try:
                with self._opener(Request(url, headers=headers), timeout=self.timeout_sec) as response:
                    body = response.read().decode("utf-8", errors="replace")
                if body.strip():
                    return body
            except HTTPError as exc:
                if exc.code not in RETRYABLE_STATUSES:
                    return None
                if exc.headers is not None:
                    retry_after = _parse_retry_after(exc.headers.get("Retry-After"))
            except (URLError, TimeoutError, OSError):
                pass  # transient — fall through to backoff/retry

            if attempt < self.max_retries - 1:
                if retry_after is not None:
                    # The server told us exactly how long to wait — that is
                    # more informative than our own guess, so honor it
                    # (capped, so one retry can't eat the whole deadline
                    # budget of a caller racing multiple providers).
                    self._sleep(min(retry_after, MAX_RETRY_AFTER_SEC))
                else:
                    backoff = self.backoff_base_sec * (2 ** attempt)
                    self._sleep(backoff + random.uniform(0, 0.3))
        return None
