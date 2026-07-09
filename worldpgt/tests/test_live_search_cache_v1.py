"""Tests for the volatile live web-search cache and its rendering.

Covers LiveSearchCache (get/put/TTL/normalization) and the confident-lead
format of render_web_answer, including the cache-hit path.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from worldpgt.assistant_surface.web_search import WebSearchResult, render_web_answer
from worldpgt.web_search.live_cache import LiveSearchCache, _normalize_question


def _result(**overrides) -> WebSearchResult:
    defaults = dict(
        title="President of France",
        snippet="The current president of France is Emmanuel Macron.",
        url="https://example.com/france-president",
    )
    defaults.update(overrides)
    return WebSearchResult(**defaults)


def test_normalize_question_ignores_case_and_punctuation():
    assert _normalize_question("Who is the President of France?") == _normalize_question(
        "who is the president of france"
    )


def test_put_then_get_round_trips(tmp_path):
    cache = LiveSearchCache(tmp_path / "cache.json")
    cache.put("Who is the current president of France?", [_result()])

    entry = cache.get("who is the current president of france")
    assert entry is not None
    assert entry.results[0].title == "President of France"
    assert entry.results[0].url == "https://example.com/france-president"


def test_get_misses_when_nothing_cached(tmp_path):
    cache = LiveSearchCache(tmp_path / "cache.json")
    assert cache.get("anything") is None


def test_get_misses_when_entry_expired(tmp_path):
    path = tmp_path / "cache.json"
    stale = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
    path.write_text(json.dumps({
        "by_question": {
            "who is the current president of france": {
                "question": "Who is the current president of France?",
                "fetched_at": stale,
                "results": [{"title": "x", "snippet": "y", "url": "https://example.com"}],
            }
        },
        "by_entity": {},
    }), encoding="utf-8")

    cache = LiveSearchCache(path, ttl_hours=168.0)
    assert cache.get("Who is the current president of France?") is None


def test_get_hits_when_entry_fresh(tmp_path):
    path = tmp_path / "cache.json"
    fresh = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps({
        "by_question": {
            "who is the current president of france": {
                "question": "Who is the current president of France?",
                "fetched_at": fresh,
                "results": [{"title": "x", "snippet": "y", "url": "https://example.com"}],
            }
        },
        "by_entity": {},
    }), encoding="utf-8")

    cache = LiveSearchCache(path, ttl_hours=168.0)
    assert cache.get("Who is the current president of France?") is not None


def test_legacy_flat_schema_file_is_ignored_not_crashed_on(tmp_path):
    """The cache is volatile, not durable state — an old-schema file on disk
    (pre entity-keying) is treated as empty rather than migrated or erroring."""
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({
        "who is x": {"question": "who is x", "fetched_at": "2026-01-01T00:00:00+00:00", "results": []}
    }), encoding="utf-8")

    cache = LiveSearchCache(path)
    assert cache.get("who is x") is None


# --------------------------------------------------------------------------- #
# Entity-keyed cache: reused across differently-worded questions.
# --------------------------------------------------------------------------- #
def test_put_entity_then_find_in_a_completely_different_question(tmp_path):
    from worldpgt.web_search.live_cache import entity_key_from_title

    cache = LiveSearchCache(tmp_path / "cache.json")
    result = _result(title="Sanae Takaichi - Wikipedia", snippet="Sanae Takaichi is ...")
    cache.put_entity(entity_key_from_title(result.title), [result])

    hit = cache.find_entity_in_question("When was Sanae Takaichi born?")
    assert hit is not None
    assert hit.results[0].title == "Sanae Takaichi - Wikipedia"


def test_find_entity_in_question_misses_unrelated_question(tmp_path):
    from worldpgt.web_search.live_cache import entity_key_from_title

    cache = LiveSearchCache(tmp_path / "cache.json")
    result = _result(title="Sanae Takaichi - Wikipedia", snippet="...")
    cache.put_entity(entity_key_from_title(result.title), [result])

    assert cache.find_entity_in_question("What is SpaceX?") is None


def test_find_entity_in_question_prefers_longest_match(tmp_path):
    cache = LiveSearchCache(tmp_path / "cache.json")
    cache.put_entity("john adams", [_result(title="John Adams - Wikipedia")])
    cache.put_entity("john quincy adams", [_result(title="John Quincy Adams - Wikipedia")])

    hit = cache.find_entity_in_question("What was John Quincy Adams famous for?")
    assert hit is not None
    assert hit.results[0].title == "John Quincy Adams - Wikipedia"


def test_entity_key_from_title_strips_wikipedia_suffix():
    from worldpgt.web_search.live_cache import entity_key_from_title

    assert entity_key_from_title("Sanae Takaichi - Wikipedia") == "sanae takaichi"


def test_render_web_answer_leads_with_direct_statement():
    text = render_web_answer("Who is the president of France?", [_result()])
    lines = text.splitlines()
    assert "Emmanuel Macron" in lines[1]
    assert "not Microworld memory" in text
    assert "Source: President of France" in text


def test_render_web_answer_lists_additional_sources():
    text = render_web_answer(
        "q",
        [_result(), _result(title="Macron - Wikipedia", url="https://example.com/macron")],
    )
    assert "Additional sources:" in text
    assert "https://example.com/macron" in text


def test_render_web_answer_cache_hit_shows_original_fetch_time():
    text = render_web_answer(
        "q", [_result()], fetched_at="2026-01-01T00:00:00+00:00", from_cache=True,
    )
    assert "2026-01-01T00:00:00+00:00" in text
    assert "(cached)" in text


def test_render_web_answer_empty_results_returns_empty_string():
    assert render_web_answer("q", []) == ""
