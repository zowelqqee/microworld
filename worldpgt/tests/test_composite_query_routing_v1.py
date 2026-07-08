"""Tests for which query string each provider receives from the composite.

Structural fix (2026-07-07): only WikipediaProvider gets the regex-reformed
``intent.search_query`` -- Wikidata and Claude both do their own real
understanding of the question (structured entity matching / actual language
understanding) and were previously handed our fragile regex output instead,
which imported bugs like "what capital of austria?" -> reformed into the
self-duplicating "capital of austria capital", or "who played on the
jeffersons?" -> the entity itself getting stripped by a relation-tail
heuristic that assumes verb-after-entity order, producing an empty
bare-entity and a garbled fallback query that matched the wrong Wikipedia
article entirely (a basketball player instead of the TV show).
"""

from __future__ import annotations

from worldpgt.web_search.claude_search import ClaudeWebSearchProvider
from worldpgt.web_search.composite import CompositeSearchProvider
from worldpgt.web_search.wikidata import WikidataProvider
from worldpgt.web_search.wikipedia import WikipediaProvider


class _RecordingProvider:
    def __init__(self):
        self.received_queries: list[str] = []

    def search(self, query, *, max_results=3):
        self.received_queries.append(query)
        return []


class _RecordingWikipediaProvider(_RecordingProvider, WikipediaProvider):
    def __init__(self):
        _RecordingProvider.__init__(self)


class _RecordingWikidataProvider(_RecordingProvider, WikidataProvider):
    def __init__(self):
        _RecordingProvider.__init__(self)


class _RecordingClaudeProvider(_RecordingProvider, ClaudeWebSearchProvider):
    def __init__(self):
        _RecordingProvider.__init__(self)


def test_wikipedia_gets_the_regex_reformed_query_but_wikidata_and_claude_get_the_raw_question() -> None:
    wikipedia = _RecordingWikipediaProvider()
    wikidata = _RecordingWikidataProvider()
    claude = _RecordingClaudeProvider()
    comp = CompositeSearchProvider([wikipedia, wikidata, claude], deadline_sec=2.0)

    question = "what capital of austria?"
    comp.search(question)

    assert wikipedia.received_queries == ["capital of austria"]
    assert wikidata.received_queries == [question]
    assert claude.received_queries == [question]  # NOT the garbled reformed query
