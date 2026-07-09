"""Tests for WikipediaProvider's fast-fail defaults and parallel fetch.

Covers the 2026-07-06 speedup: tighter retry/timeout defaults (Wikipedia is
not observed rate-limited, unlike Wikidata/DuckDuckGo, so the old generous
backoff only added latency) and running the full-question + bare-entity
fetches concurrently instead of sequentially.
"""

from __future__ import annotations

import io
import json

from worldpgt.web_search.wikipedia import WikipediaProvider


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def _page_response(title: str, extract: str) -> _FakeResponse:
    body = json.dumps({"query": {"pages": {"1": {"title": title, "extract": extract}}}})
    return _FakeResponse(body.encode("utf-8"))


def test_defaults_are_tighter_than_the_old_generous_backoff() -> None:
    """Wikipedia isn't rate-limited in practice, so its defaults should fail
    faster than Wikidata/DuckDuckGo's (which tolerate real 429s)."""
    provider = WikipediaProvider()
    assert provider._http.timeout_sec <= 4.0
    assert provider._http.max_retries <= 2
    assert provider._http.backoff_base_sec <= 0.5


def test_single_word_query_skips_the_bare_entity_fetch() -> None:
    """No relation clause to strip -> bare == full query -> only one request."""
    calls = []

    def opener(req, timeout=None):
        calls.append(req.full_url)
        return _page_response("Everest", "Mount Everest is the tallest mountain.")

    provider = WikipediaProvider(opener=opener, sleep=lambda s: None)
    results = provider.search("everest")

    assert len(calls) == 1
    assert results and results[0].title == "Everest - Wikipedia"


def test_full_and_bare_entity_fetches_run_concurrently_not_sequentially() -> None:
    """Both fetches should be in flight before either's opener call returns --
    proof they run as concurrent futures, not one after another."""
    import threading

    barrier = threading.Barrier(2, timeout=2.0)

    def opener(req, timeout=None):
        barrier.wait()  # only succeeds if both requests reached this point together
        if "franklin" in req.full_url.lower() and "invent" not in req.full_url.lower():
            return _page_response("Benjamin Franklin", "Benjamin Franklin was a polymath.")
        return _page_response("What Ben Franklin Invented (article)", "A listicle.")

    provider = WikipediaProvider(opener=opener, sleep=lambda s: None)
    results = provider.search("what else did ben franklin invent?")

    assert results  # both requests completed without deadlocking/timing out
    assert "Franklin" in results[0].title


def test_prefers_bare_entity_title_when_full_question_result_is_off_topic() -> None:
    def opener(req, timeout=None):
        if "marry" in req.full_url.lower():  # the full-question fetch
            return _page_response("Serbian scientists", "A list of notable Serbian scientists.")
        return _page_response("Nikola Tesla", "Nikola Tesla was an inventor and engineer.")

    provider = WikipediaProvider(opener=opener, sleep=lambda s: None)
    results = provider.search("who did nikola tesla marry?")

    assert results and results[0].title == "Nikola Tesla - Wikipedia"
