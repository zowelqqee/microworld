"""Tests for deterministic web-search query intent and relevance guards."""

from __future__ import annotations

from worldpgt.assistant_surface.web_search import WebSearchResult
from worldpgt.web_search.query_intent import (
    build_web_query_intent,
    filter_and_rank_results,
    is_relevant_result,
)


def test_language_question_rewrites_to_country_language_query() -> None:
    intent = build_web_query_intent("what does jamaican people speak?")

    assert intent.subject_query == "jamaican"
    assert intent.search_query == "jamaican languages spoken"
    assert intent.relation_terms == ("language",)


def test_timezone_question_rejects_unrelated_dallas_tv_result() -> None:
    intent = build_web_query_intent("what is my timezone in louisiana?")
    bad = WebSearchResult(
        title="Dallas (TV series) - Wikipedia",
        snippet="Dallas is an American prime time television soap opera.",
        url="https://en.wikipedia.org/wiki/Dallas_(TV_series)",
    )

    assert intent.search_query == "louisiana time zone"
    assert not is_relevant_result(intent, bad)


def test_filter_and_rank_keeps_subject_relevant_result() -> None:
    bad = WebSearchResult(
        title="Dallas (TV series) - Wikipedia",
        snippet="Dallas is an American prime time television soap opera.",
        url="https://en.wikipedia.org/wiki/Dallas_(TV_series)",
    )
    good = WebSearchResult(
        title="Louisiana - Wikipedia",
        snippet="Louisiana is in the Central Time Zone.",
        url="https://en.wikipedia.org/wiki/Louisiana",
    )

    _intent, results = filter_and_rank_results("what is my timezone in louisiana?", [bad, good])

    assert results == [good]


def test_filter_and_rank_does_not_reject_a_nickname_near_miss() -> None:
    """"Ben Franklin" vs "Benjamin Franklin" is a real, correct match that the
    strict token-overlap score alone puts just under threshold. Rejecting it
    outright turned "found the right page" into "found nothing" — a large,
    real answer-rate regression (99.6% -> ~30% on the WebQuestions sample)
    traced to exactly this. It must survive as a plausible result instead.
    """

    result = WebSearchResult(
        title="Benjamin Franklin - Wikipedia",
        snippet="Benjamin Franklin was an American polymath and Founding Father.",
        url="https://en.wikipedia.org/wiki/Benjamin_Franklin",
    )

    _intent, results = filter_and_rank_results("what else did ben franklin invent?", [result])

    assert results == [result]


def test_filter_and_rank_still_rejects_pure_zero_overlap_result() -> None:
    """A result that never names the subject at all — only a generic,
    coincidentally-matching relation word ("time" inside "prime time
    television") — must not be treated as plausible just because the raw
    score is nonzero; it has zero real signal about the actual subject.
    """

    bad = WebSearchResult(
        title="Dallas (TV series) - Wikipedia",
        snippet="Dallas is an American prime time television soap opera.",
        url="https://en.wikipedia.org/wiki/Dallas_(TV_series)",
    )

    _intent, results = filter_and_rank_results("what is my timezone in louisiana?", [bad])

    assert results == []
