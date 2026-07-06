"""Tests for optional live web-search providers."""

from __future__ import annotations

import io
from urllib.error import HTTPError

import pytest

from worldpgt.web_search.duckduckgo import (
    DuckDuckGoInstantAnswerProvider,
    _results_from_html,
    _results_from_lite_html,
)


class _FakeResponse(io.BytesIO):
    """Minimal context-manager HTTP response returning a fixed body."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def _lite_html(title: str, url: str, snippet: str) -> str:
    return f"""
    <tr><td>
      <a href="{url}" class='result-link'>{title}</a>
    </td></tr>
    <tr><td class='result-snippet'>{snippet}</td></tr>
    """


def test_duckduckgo_html_parser_extracts_source_results() -> None:
    html = """
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://example.com/france">
        President of France
      </a>
      <a class="result__snippet">The current president of France is listed here.</a>
    </div>
    """

    results = _results_from_html(html, max_results=3)

    assert len(results) == 1
    assert results[0].title == "President of France"
    assert results[0].snippet == "The current president of France is listed here."
    assert results[0].url == "https://example.com/france"


def test_duckduckgo_html_parser_drops_ad_results() -> None:
    html = """
    <div class="result">
      <a rel="nofollow" class="result__a"
         href="https://duckduckgo.com/y.js?ad_domain=viator.com&amp;ad_provider=bingv7aa&amp;ad_type=txad">
        Top Attractions in Atlanta
      </a>
      <a class="result__snippet">Book tickets to top museums and attractions in Atlanta.</a>
    </div>
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://www.eventbrite.com/d/ga--atlanta/events--today/">
        Atlanta, GA Family Events Today | Eventbrite
      </a>
      <a class="result__snippet">Family Events and Things to do in Atlanta, GA today.</a>
    </div>
    """

    results = _results_from_html(html, max_results=3)

    assert len(results) == 1
    assert results[0].url == "https://www.eventbrite.com/d/ga--atlanta/events--today/"


def test_duckduckgo_lite_parser_drops_ad_results() -> None:
    html = """
    <tr>
      <td>
        <a rel="nofollow"
           href="https://duckduckgo.com/y.js?ad_domain=familyhotelsguide.com&amp;ad_provider=bingv7aa&amp;ad_type=txad"
           class='result-link'>Atlanta Hotels with family</a>
      </td>
    </tr>
    <tr>
      <td class='result-snippet'>Top 10 Coolest Family Hotels for 2026.</td>
    </tr>
    <tr>
      <td>
        <a rel="nofollow"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fwww.eventbrite.com%2Fd%2Fga--atlanta%2Fevents--today%2F&amp;rut=abc"
           class='result-link'>Atlanta, GA Family Events Today | Eventbrite</a>
      </td>
    </tr>
    <tr>
      <td class='result-snippet'>Family Events and Things to do in Atlanta, GA today.</td>
    </tr>
    """

    results = _results_from_lite_html(html, max_results=3)

    assert len(results) == 1
    assert results[0].url == "https://www.eventbrite.com/d/ga--atlanta/events--today/"


def test_duckduckgo_lite_parser_extracts_redirect_target() -> None:
    html = """
    <tr>
      <td>
        <a rel="nofollow"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FEmmanuel_Macron&amp;rut=abc"
           class='result-link'>Emmanuel Macron - Wikipedia</a>
      </td>
    </tr>
    <tr>
      <td class='result-snippet'>
        Emmanuel Macron is a French politician serving as <b>President</b> of France.
      </td>
    </tr>
    """

    results = _results_from_lite_html(html, max_results=3)

    assert len(results) == 1
    assert results[0].title == "Emmanuel Macron - Wikipedia"
    assert results[0].snippet == "Emmanuel Macron is a French politician serving as President of France."
    assert results[0].url == "https://en.wikipedia.org/wiki/Emmanuel_Macron"


# --------------------------------------------------------------------------- #
# Retry-After: honor the server's explicit wait time instead of guessing.
# Real-world trigger: Wikidata returns 429 + Retry-After (observed 24-38s) —
# our own exponential backoff (0.5-1.6s) was far too short, so a retry almost
# always hit the same rate limit again.
# --------------------------------------------------------------------------- #
def test_parse_retry_after_integer_seconds():
    from worldpgt.web_search.http import _parse_retry_after
    assert _parse_retry_after("38") == 38.0
    assert _parse_retry_after("0") == 0.0


def test_parse_retry_after_invalid_or_missing_returns_none():
    from worldpgt.web_search.http import _parse_retry_after
    assert _parse_retry_after(None) is None
    assert _parse_retry_after("") is None
    assert _parse_retry_after("Wed, 21 Oct in the future") is None
    assert _parse_retry_after("-5") is None


def test_get_text_honors_retry_after_header_over_default_backoff():
    from worldpgt.web_search.http import ResilientHttpClient

    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "5"}, None)
        return _FakeResponse(b"ok")

    sleeps: list[float] = []
    client = ResilientHttpClient(
        opener=opener,
        sleep=lambda s: sleeps.append(s),
        min_interval_sec=0.0,
        backoff_base_sec=0.1,  # default backoff would be far shorter than 5s
        max_retries=2,
    )

    result = client.get_text("https://example.com/x")

    assert result == "ok"
    assert sleeps == [5.0]  # honored the header, not the 0.1s exponential guess


def test_get_text_caps_an_excessive_retry_after():
    from worldpgt.web_search.http import ResilientHttpClient, MAX_RETRY_AFTER_SEC

    def opener(req, timeout=None):
        raise HTTPError(req.full_url, 429, "Too Many Requests", {"Retry-After": "300"}, None)

    sleeps: list[float] = []
    client = ResilientHttpClient(
        opener=opener,
        sleep=lambda s: sleeps.append(s),
        min_interval_sec=0.0,
        max_retries=2,
    )

    client.get_text("https://example.com/x")

    assert sleeps == [MAX_RETRY_AFTER_SEC]


def test_get_text_falls_back_to_backoff_without_retry_after_header():
    from worldpgt.web_search.http import ResilientHttpClient

    def opener(req, timeout=None):
        raise HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    sleeps: list[float] = []
    client = ResilientHttpClient(
        opener=opener,
        sleep=lambda s: sleeps.append(s),
        min_interval_sec=0.0,
        backoff_base_sec=0.5,
        max_retries=2,
    )

    client.get_text("https://example.com/x")

    assert len(sleeps) == 1
    assert 0.5 <= sleeps[0] <= 0.8  # backoff_base_sec + jitter, not a fixed 5s


# --------------------------------------------------------------------------- #
# Network resilience: pacing, retry/backoff, endpoint fallback.
# --------------------------------------------------------------------------- #
def _provider(opener, *, sleeps: list[float], **kwargs):
    return DuckDuckGoInstantAnswerProvider(
        opener=opener,
        sleep=lambda s: sleeps.append(s),
        min_interval_sec=0.0,
        backoff_base_sec=0.5,
        **kwargs,
    )


def test_retries_on_202_then_succeeds():
    calls: dict[str, int] = {}
    good = _lite_html("Wikipedia result", "https://en.wikipedia.org/wiki/X", "A real snippet.")

    def opener(req, timeout=None):
        calls[req.full_url] = calls.get(req.full_url, 0) + 1
        if "lite.duckduckgo.com" in req.full_url:
            # 202 once, then succeeds with real content.
            if calls[req.full_url] == 1:
                raise HTTPError(req.full_url, 202, "Accepted", {}, None)
            return _FakeResponse(good.encode("utf-8"))
        # html endpoint (raced concurrently): never has anything usable.
        return _FakeResponse(b"")

    sleeps: list[float] = []
    provider = _provider(opener, sleeps=sleeps)
    results = provider.search("who is x", max_results=3)

    assert len(results) == 1
    assert results[0].url == "https://en.wikipedia.org/wiki/X"
    # The 202 triggered at least one backoff sleep.
    assert any(s > 0 for s in sleeps)


def test_gives_up_after_max_retries_returns_empty():
    def opener(req, timeout=None):
        raise HTTPError(req.full_url, 429, "Too Many Requests", {}, None)

    sleeps: list[float] = []
    provider = _provider(opener, sleeps=sleeps, max_retries=3)
    results = provider.search("who is x")

    assert results == []


def test_non_retryable_status_is_not_retried():
    attempts: list[str] = []

    def opener(req, timeout=None):
        attempts.append(req.full_url)
        raise HTTPError(req.full_url, 404, "Not Found", {}, None)

    provider = _provider(opener, sleeps=[], max_retries=5)
    results = provider.search("who is x")

    assert results == []
    # 404 is terminal: exactly one attempt per endpoint (lite + html), no retries.
    assert len(attempts) == 2


def test_lite_and_html_are_raced_not_paced_against_each_other():
    """lite and html now use SEPARATE clients (independent pacing state) so
    racing them isn't serialized by one shared rate limiter — each endpoint's
    FIRST request within a single search() call must be free of pacing."""
    clock = {"t": 0.0}

    def opener(req, timeout=None):
        return _FakeResponse(b"")  # empty -> falls through to the other

    sleeps: list[float] = []
    provider = DuckDuckGoInstantAnswerProvider(
        opener=opener,
        sleep=lambda s: sleeps.append(s),
        monotonic=lambda: clock["t"],
        min_interval_sec=1.5,
        max_retries=1,
    )
    provider.search("who is x")

    # Neither endpoint's first-ever request should be paced.
    assert not any(s == pytest.approx(1.5) for s in sleeps)


def test_pacing_applies_per_endpoint_across_separate_search_calls():
    """Pacing still protects against a burst of DIFFERENT queries: the SECOND
    call to search() must pace each endpoint's client against its own first
    call (independently — no shared cross-endpoint rate limiter)."""
    clock = {"t": 0.0}

    def opener(req, timeout=None):
        return _FakeResponse(b"")

    sleeps: list[float] = []
    provider = DuckDuckGoInstantAnswerProvider(
        opener=opener,
        sleep=lambda s: sleeps.append(s),
        monotonic=lambda: clock["t"],
        min_interval_sec=1.5,
        max_retries=1,
    )
    provider.search("first query")
    sleeps.clear()
    provider.search("second query")

    # Second call: both lite's and html's client now have a prior request on
    # record, so both must pace by the full interval (clock never advances).
    pacing_sleeps = [s for s in sleeps if s == pytest.approx(1.5)]
    assert len(pacing_sleeps) >= 2


# --------------------------------------------------------------------------- #
# Circuit breaker: DuckDuckGo's own homepage returned instead of results.
# --------------------------------------------------------------------------- #
_BLOCKED_HOMEPAGE_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <link rel="canonical" href="https://duckduckgo.com/">
    <title>
        DuckDuckGo
    </title>
</head>
<body></body>
</html>
"""


def test_is_blocked_page_detects_generic_homepage():
    from worldpgt.web_search.duckduckgo import _is_blocked_page
    assert _is_blocked_page(_BLOCKED_HOMEPAGE_HTML) is True


def test_is_blocked_page_false_for_genuine_results_page():
    from worldpgt.web_search.duckduckgo import _is_blocked_page
    good = _lite_html("Elon Musk - Wikipedia", "https://en.wikipedia.org/wiki/Elon_Musk", "Snippet.")
    assert _is_blocked_page(good) is False


def test_is_blocked_page_false_for_empty_or_none():
    from worldpgt.web_search.duckduckgo import _is_blocked_page
    assert _is_blocked_page("") is False
    assert _is_blocked_page(None) is False


def test_breaker_trips_on_blocked_page_and_skips_subsequent_network_calls():
    call_count = {"n": 0}

    def opener(req, timeout=None):
        call_count["n"] += 1
        return _FakeResponse(_BLOCKED_HOMEPAGE_HTML.encode("utf-8"))

    provider = DuckDuckGoInstantAnswerProvider(
        opener=opener, sleep=lambda s: None, min_interval_sec=0.0, max_retries=1,
    )

    first = provider.search("who is x")
    calls_after_first = call_count["n"]
    assert first == []
    assert provider.circuit_open is True
    assert calls_after_first > 0  # the first call DID hit the network

    second = provider.search("who is y")

    assert second == []
    assert call_count["n"] == calls_after_first  # no new network calls at all


def test_breaker_resets_on_a_genuine_response_once_cooldown_elapses():
    # While open, EVERY call is skipped (no network at all, see the
    # skip-subsequent-calls test above) — there is no early "probe" attempt.
    # Recovery is only checked once cooldown_sec has actually elapsed.
    clock = {"t": 0.0}
    good = _lite_html("Elon Musk - Wikipedia", "https://en.wikipedia.org/wiki/Elon_Musk", "Snippet.")

    responses = iter([
        _BLOCKED_HOMEPAGE_HTML.encode("utf-8"),  # lite: blocked
        b"",                                      # html: empty (irrelevant)
        good.encode("utf-8"),                     # lite: genuine, after cooldown
        b"",
    ])

    def opener(req, timeout=None):
        return _FakeResponse(next(responses))

    provider = DuckDuckGoInstantAnswerProvider(
        opener=opener, sleep=lambda s: None, monotonic=lambda: clock["t"],
        min_interval_sec=0.0, max_retries=1, breaker_cooldown_sec=10.0,
    )

    provider.search("who is x")
    assert provider.circuit_open is True

    clock["t"] += 11.0  # cooldown has elapsed
    results = provider.search("who is elon musk")

    assert provider.circuit_open is False
    assert len(results) == 1


def test_breaker_reopens_after_cooldown_elapses():
    clock = {"t": 0.0}

    def opener(req, timeout=None):
        return _FakeResponse(_BLOCKED_HOMEPAGE_HTML.encode("utf-8"))

    provider = DuckDuckGoInstantAnswerProvider(
        opener=opener,
        sleep=lambda s: None,
        monotonic=lambda: clock["t"],
        min_interval_sec=0.0,
        max_retries=1,
        breaker_cooldown_sec=60.0,
    )

    provider.search("who is x")
    assert provider.circuit_open is True

    clock["t"] += 30.0  # still within cooldown
    assert provider.circuit_open is True

    clock["t"] += 31.0  # cooldown has now elapsed
    assert provider.circuit_open is False


# --------------------------------------------------------------------------- #
# Composite provider: Wikipedia first, DuckDuckGo fallback.
# --------------------------------------------------------------------------- #
class _StubProvider:
    def __init__(self, results):
        self._results = results
        self.calls = 0

    def search(self, query, *, max_results=3):
        self.calls += 1
        return list(self._results)


def _res(title):
    from worldpgt.assistant_surface.web_search import WebSearchResult
    return WebSearchResult(title=title, snippet="s", url="https://example.com")


def test_composite_returns_first_non_empty_and_skips_later_providers():
    from worldpgt.web_search.composite import CompositeSearchProvider

    primary = _StubProvider([_res("Wikipedia hit")])
    secondary = _StubProvider([_res("DDG hit")])
    comp = CompositeSearchProvider([primary, secondary])

    results = comp.search("q")

    assert [r.title for r in results] == ["Wikipedia hit"]
    assert primary.calls == 1
    assert secondary.calls == 0  # short-circuited


def test_composite_falls_through_to_second_provider_when_first_empty():
    from worldpgt.web_search.composite import CompositeSearchProvider

    primary = _StubProvider([])
    secondary = _StubProvider([_res("DDG hit")])
    comp = CompositeSearchProvider([primary, secondary])

    results = comp.search("q")

    assert [r.title for r in results] == ["DDG hit"]
    assert primary.calls == 1 and secondary.calls == 1


def test_composite_survives_a_provider_that_raises():
    from worldpgt.web_search.composite import CompositeSearchProvider

    class _Boom:
        def search(self, query, *, max_results=3):
            raise RuntimeError("network down")

    secondary = _StubProvider([_res("DDG hit")])
    comp = CompositeSearchProvider([_Boom(), secondary])

    results = comp.search("q")

    assert [r.title for r in results] == ["DDG hit"]


class _SlowProvider:
    """Sleeps ``delay_sec`` before returning, to test racing/deadline behavior."""

    def __init__(self, delay_sec, results):
        self.delay_sec = delay_sec
        self._results = results
        self.calls = 0

    def search(self, query, *, max_results=3):
        import time
        self.calls += 1
        time.sleep(self.delay_sec)
        return list(self._results)


def test_composite_does_not_wait_for_slow_lower_priority_provider():
    """A fast primary result must return promptly even if a lower-priority
    provider is still (slowly) running in the background."""
    from worldpgt.web_search.composite import CompositeSearchProvider
    import time

    primary = _StubProvider([_res("Wikipedia hit")])  # instant
    secondary = _SlowProvider(2.0, [_res("DDG hit")])  # slow, should be ignored
    comp = CompositeSearchProvider([primary, secondary], deadline_sec=5.0)

    t0 = time.perf_counter()
    results = comp.search("q")
    elapsed = time.perf_counter() - t0

    assert [r.title for r in results] == ["Wikipedia hit"]
    assert elapsed < 1.0  # did not wait for the 2s-slow secondary


def test_composite_falls_through_to_second_provider_after_first_empty_slow():
    """A slow-but-empty primary must not block the secondary from being
    tried; total time is bounded by the deadline, not summed per-provider."""
    from worldpgt.web_search.composite import CompositeSearchProvider
    import time

    primary = _SlowProvider(0.3, [])       # slow AND empty
    secondary = _StubProvider([_res("DDG hit")])  # instant, but started only
    comp = CompositeSearchProvider([primary, secondary], deadline_sec=5.0)

    t0 = time.perf_counter()
    results = comp.search("q")
    elapsed = time.perf_counter() - t0

    assert [r.title for r in results] == ["DDG hit"]
    assert elapsed < 1.0  # bounded by primary's own delay, not primary+secondary


def test_composite_gives_up_at_deadline_returns_empty_not_hanging():
    """If every provider is still running when the deadline passes, return
    [] promptly instead of blocking until the slow provider finally finishes."""
    from worldpgt.web_search.composite import CompositeSearchProvider
    import time

    primary = _SlowProvider(5.0, [_res("Wikipedia hit")])
    secondary = _SlowProvider(5.0, [_res("DDG hit")])
    comp = CompositeSearchProvider([primary, secondary], deadline_sec=0.5)

    t0 = time.perf_counter()
    results = comp.search("q")
    elapsed = time.perf_counter() - t0

    assert results == []
    assert elapsed < 1.5  # bounded by the 0.5s deadline, not the 5s providers


def test_composite_default_deadline_is_bounded():
    from worldpgt.web_search.composite import DEFAULT_DEADLINE_SEC
    assert 0 < DEFAULT_DEADLINE_SEC <= 20
