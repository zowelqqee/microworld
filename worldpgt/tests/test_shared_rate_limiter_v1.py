"""Tests for the process-wide Wikimedia rate limiter (2026-07-06).

Context: a full WebQuestions benchmark run drove Wikipedia+Wikidata request
volume high enough to trip Wikimedia's rate limiting from this sandbox's
shared egress IP -- confirmed with raw ``curl`` calls that bypassed this
codebase entirely and still came back HTTP 429 for several minutes
afterward. Per-``ResilientHttpClient``-instance pacing didn't prevent this,
because Wikipedia's own two concurrent sub-fetches (full-question,
bare-entity) and the separate Wikidata provider are all *different* client
instances racing concurrently in ``CompositeSearchProvider`` -- each paced
individually, but never coordinated with each other. ``SharedRateLimiter``
closes that gap.
"""

from __future__ import annotations

import threading
import time

from worldpgt.web_search.composite import CompositeSearchProvider
from worldpgt.web_search.http import ResilientHttpClient, SharedRateLimiter
from worldpgt.web_search.wikidata import WikidataProvider
from worldpgt.web_search.wikipedia import WikipediaProvider


def test_acquire_enforces_min_interval_between_sequential_calls() -> None:
    clock = [0.0]
    sleeps: list[float] = []

    def monotonic():
        return clock[0]

    def sleep(seconds: float):
        sleeps.append(seconds)
        clock[0] += seconds

    limiter = SharedRateLimiter(min_interval_sec=0.5, sleep=sleep, monotonic=monotonic)
    limiter.acquire()  # first call: no prior timestamp, no sleep
    clock[0] += 0.1  # only 0.1s elapsed before the next acquire
    limiter.acquire()

    assert sleeps == [0.4]  # topped up to the full 0.5s interval


def test_acquire_does_not_sleep_when_enough_time_already_elapsed() -> None:
    clock = [0.0]
    sleeps: list[float] = []
    limiter = SharedRateLimiter(
        min_interval_sec=0.5,
        sleep=lambda s: sleeps.append(s),
        monotonic=lambda: clock[0],
    )
    limiter.acquire()
    clock[0] += 1.0  # plenty of time has passed
    limiter.acquire()

    assert sleeps == []


def test_acquire_serializes_concurrent_callers_across_threads() -> None:
    """Two threads racing acquire() must not both pass through immediately --
    proof the lock genuinely coordinates across threads, not just within one."""
    call_log: list[float] = []
    lock_for_log = threading.Lock()
    real_start = time.monotonic()

    limiter = SharedRateLimiter(min_interval_sec=0.2, sleep=time.sleep, monotonic=time.monotonic)

    def worker():
        limiter.acquire()
        with lock_for_log:
            call_log.append(time.monotonic() - real_start)

    threads = [threading.Thread(target=worker) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)

    call_log.sort()
    assert len(call_log) == 3
    # Each successive acquire should be spaced by roughly the min interval,
    # not all three firing at ~t=0.
    assert call_log[1] - call_log[0] >= 0.15
    assert call_log[2] - call_log[1] >= 0.15


def test_resilient_http_client_uses_shared_limiter_instead_of_own_pacing() -> None:
    calls: list[float] = []
    limiter_acquire_calls = []

    class _FakeLimiter:
        def acquire(self):
            limiter_acquire_calls.append(1)

    import io

    class _FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def opener(req, timeout=None):
        return _FakeResponse(b"ok body")

    client = ResilientHttpClient(
        rate_limiter=_FakeLimiter(),
        opener=opener,
        sleep=lambda s: calls.append(s),
        min_interval_sec=999,  # would sleep ~999s if the shared limiter were NOT used
    )
    body = client.get_text("https://en.wikipedia.org/w/api.php")

    assert body == "ok body"
    assert limiter_acquire_calls == [1]
    assert calls == []  # own min_interval_sec pacing must be bypassed entirely


def test_composite_default_providers_share_one_wikimedia_rate_limiter() -> None:
    """Wikipedia's two internal clients and Wikidata's client must all funnel
    through the SAME SharedRateLimiter instance -- otherwise coordination
    across them (the whole point of this fix) doesn't actually happen."""
    comp = CompositeSearchProvider()
    wikipedia = next(p for p in comp._providers if isinstance(p, WikipediaProvider))
    wikidata = next(p for p in comp._providers if isinstance(p, WikidataProvider))

    limiter_a = wikipedia._http._rate_limiter
    limiter_b = wikipedia._bare_http._rate_limiter
    limiter_c = wikidata._http._rate_limiter

    assert limiter_a is not None
    assert limiter_a is limiter_b is limiter_c
