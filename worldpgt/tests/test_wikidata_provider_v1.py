"""Tests for the Wikidata structured-facts provider (no real network).

Covers time/quantity formatting and the full search() pipeline (entity
search -> claims -> referenced-entity label resolution -> sentence
rendering) against fake HTTP responses shaped like the real Wikidata API.
"""

from __future__ import annotations

import io
import json

from worldpgt.web_search.wikidata import (
    WikidataProvider,
    _format_quantity,
    _format_time,
    officeholder_entity_query,
)


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_format_time_day_precision():
    assert _format_time({"time": "+1982-03-02T00:00:00Z", "precision": 11}) == "March 02, 1982"


def test_format_time_month_precision():
    assert _format_time({"time": "+1982-03-00T00:00:00Z", "precision": 10}) == "March 1982"


def test_format_time_year_precision():
    assert _format_time({"time": "+1982-00-00T00:00:00Z", "precision": 9}) == "1982"


def test_format_time_too_coarse_returns_none():
    # precision 8 = decade, coarser than we bother rendering.
    assert _format_time({"time": "+1980-00-00T00:00:00Z", "precision": 8}) is None


def test_format_time_missing_or_malformed_returns_none():
    assert _format_time({}) is None
    assert _format_time({"time": "garbage", "precision": 11}) is None


def test_format_quantity_integer():
    assert _format_quantity({"amount": "+331002651"}) == "331,002,651"


def test_format_quantity_decimal():
    assert _format_quantity({"amount": "+2.5"}) == "2.50"


def test_format_quantity_missing_returns_none():
    assert _format_quantity({}) is None


def test_officeholder_entity_query_normalizes_us_president():
    assert (
        officeholder_entity_query("Who is the current president of the US?")
        == "President of the United States"
    )


def _search_response(qid: str, label: str) -> bytes:
    return json.dumps({"search": [{"id": qid, "label": label, "description": "a test entity"}]}).encode("utf-8")


def _claims_response(qid: str, claims: dict) -> bytes:
    return json.dumps({"entities": {qid: {"claims": claims}}}).encode("utf-8")


def _labels_response(labels: dict[str, str]) -> bytes:
    return json.dumps({
        "entities": {qid: {"labels": {"en": {"value": label}}} for qid, label in labels.items()}
    }).encode("utf-8")


def _sitelinks_response(titles: dict[str, str]) -> bytes:
    return json.dumps({
        "entities": {
            qid: {"labels": {}, "sitelinks": {"enwiki": {"title": title}}}
            for qid, title in titles.items()
        }
    }).encode("utf-8")


def _sparql_response(label: str | None, *, holder_qid: str = "Q2") -> bytes:
    bindings = []
    if label:
        bindings.append({
            "holder": {"type": "uri", "value": f"http://www.wikidata.org/entity/{holder_qid}"},
            "holderLabel": {"type": "literal", "value": label, "xml:lang": "en"},
        })
    return json.dumps({"results": {"bindings": bindings}}).encode("utf-8")


def test_search_end_to_end_renders_curated_facts_as_sentences():
    claims = {
        "P569": [{"mainsnak": {"datavalue": {
            "type": "time", "value": {"time": "+1985-06-15T00:00:00Z", "precision": 11},
        }}}],
        "P19": [{"mainsnak": {"datavalue": {
            "type": "wikibase-entityid", "value": {"id": "Q3"},
        }}}],
        "P26": [{"mainsnak": {"datavalue": {
            "type": "wikibase-entityid", "value": {"id": "Q2"},
        }}}],
    }

    def opener(req, timeout=None):
        url = req.full_url
        if "action=wbsearchentities" in url:
            return _FakeResponse(_search_response("Q1", "Test Person"))
        if "action=wbgetentities" in url and "labels" in url and "claims" not in url:
            return _FakeResponse(_labels_response({"Q2": "Test Spouse", "Q3": "Test City"}))
        if "action=wbgetentities" in url:
            return _FakeResponse(_claims_response("Q1", claims))
        raise AssertionError(f"unexpected url: {url}")

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    results = provider.search("who is test person?")

    assert len(results) == 1
    assert results[0].title == "Test Person - Wikidata"
    assert results[0].url == "https://www.wikidata.org/wiki/Q1"
    assert "Test Person was born on June 15, 1985." in results[0].snippet
    assert "Test Person was born in Test City." in results[0].snippet
    assert "Test Person was married to Test Spouse." in results[0].snippet


def test_search_renders_current_officeholder_claim():
    claims = {
        "P1308": [{"mainsnak": {"datavalue": {
            "type": "wikibase-entityid", "value": {"id": "Q2"},
        }}}],
    }
    searched_urls: list[str] = []

    def opener(req, timeout=None):
        url = req.full_url
        searched_urls.append(url)
        if "query.wikidata.org" in url:
            return _FakeResponse(_sparql_response(None))
        if "action=wbsearchentities" in url:
            return _FakeResponse(_search_response("Q1", "President of the United States"))
        if "action=wbgetentities" in url and "labels" in url and "claims" not in url:
            return _FakeResponse(_labels_response({"Q2": "Jane Example"}))
        if "action=wbgetentities" in url:
            return _FakeResponse(_claims_response("Q1", claims))
        raise AssertionError(f"unexpected url: {url}")

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    results = provider.search("Who is the current president of the US?")

    assert "search=President+of+the+United+States" in searched_urls[0]
    assert len(results) == 1
    assert "The officeholder of President of the United States is Jane Example." in results[0].snippet


def test_search_renders_current_officeholder_from_position_statement():
    searched_urls: list[str] = []

    def opener(req, timeout=None):
        url = req.full_url
        searched_urls.append(url)
        if "action=wbsearchentities" in url:
            return _FakeResponse(_search_response("Q1", "President of the United States"))
        if "query.wikidata.org" in url:
            return _FakeResponse(_sparql_response("Jane Example"))
        raise AssertionError(f"unexpected url: {url}")

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    results = provider.search("Who is the current president of the US?")

    assert any("query.wikidata.org" in url and "wd%3AQ1" in url for url in searched_urls)
    assert len(results) == 1
    assert results[0].snippet == "The officeholder of President of the United States is Jane Example."


def test_search_resolves_qid_when_sparql_label_service_returns_id():
    def opener(req, timeout=None):
        url = req.full_url
        if "action=wbsearchentities" in url:
            return _FakeResponse(_search_response("Q1", "President of the United States"))
        if "query.wikidata.org" in url:
            return _FakeResponse(_sparql_response("Q2", holder_qid="Q2"))
        if "action=wbgetentities" in url and "labels" in url:
            return _FakeResponse(_labels_response({"Q2": "Jane Example"}))
        raise AssertionError(f"unexpected url: {url}")

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    results = provider.search("Who is the current president of the US?")

    assert len(results) == 1
    assert results[0].snippet == "The officeholder of President of the United States is Jane Example."


def test_search_resolves_qid_from_enwiki_sitelink_when_label_missing():
    def opener(req, timeout=None):
        url = req.full_url
        if "action=wbsearchentities" in url:
            return _FakeResponse(_search_response("Q1", "President of the United States"))
        if "query.wikidata.org" in url:
            return _FakeResponse(_sparql_response("Q2", holder_qid="Q2"))
        if "action=wbgetentities" in url and "sitelinks" in url:
            return _FakeResponse(_sitelinks_response({"Q2": "Jane Example"}))
        raise AssertionError(f"unexpected url: {url}")

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    results = provider.search("Who is the current president of the US?")

    assert len(results) == 1
    assert results[0].snippet == "The officeholder of President of the United States is Jane Example."


def test_search_returns_empty_when_entity_not_found():
    def opener(req, timeout=None):
        return _FakeResponse(json.dumps({"search": []}).encode("utf-8"))

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    assert provider.search("who is nobody?") == []


def test_search_returns_empty_when_entity_has_no_curated_properties():
    def opener(req, timeout=None):
        url = req.full_url
        if "action=wbsearchentities" in url:
            return _FakeResponse(_search_response("Q1", "Obscure Entity"))
        return _FakeResponse(_claims_response("Q1", {"P999": [{"mainsnak": {"datavalue": {
            "type": "string", "value": "irrelevant, not in our template map",
        }}}]}))

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    assert provider.search("who is obscure entity?") == []


def test_search_skips_claims_whose_referenced_label_cannot_be_resolved():
    claims = {
        "P26": [{"mainsnak": {"datavalue": {
            "type": "wikibase-entityid", "value": {"id": "Q2"},
        }}}],
    }

    def opener(req, timeout=None):
        url = req.full_url
        if "action=wbsearchentities" in url:
            return _FakeResponse(_search_response("Q1", "Test Person"))
        if "labels" in url and "claims" not in url:
            return _FakeResponse(_labels_response({}))  # resolution fails
        return _FakeResponse(_claims_response("Q1", claims))

    provider = WikidataProvider(opener=opener, sleep=lambda s: None, min_interval_sec=0.0)
    assert provider.search("who did test person marry?") == []
