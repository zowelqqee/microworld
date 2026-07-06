"""Tests for follow-up pronoun resolution after a live web-search answer.

Reproduces the bug found in manual testing: "Who is the current president
of Japan?" (answered via web search, since the overlay has no such fact)
followed by "Where was she born?" used to fail with
`unresolved_reference: could not determine what 'she' refers to`, because
the static entity index has no type for a person named only in a live
search result. worldpgt/api/server.py:_web_search_entity_hint extracts a
(name, type) hint from the rendered web-search answer text, and
ConversationTurn.entity_types carries it as a session-only fallback for the
coreference resolver — never written to the overlay.
"""

from __future__ import annotations

from worldpgt.api import server
from worldpgt.dialogue.conversation_context import ConversationContext
from worldpgt.dialogue.coreference_resolver import resolve_coreferences
from worldpgt.assistant_surface.types import AssistantAnswer
from worldpgt.assistant_surface.web_search import WebSearchResult, render_web_answer
from worldpgt.entity_qa.types import SemanticQuery


_JAPAN_PRESIDENT_ANSWER_TEXT = render_web_answer(
    "Who is the current president of Japan?",
    [
        WebSearchResult(
            title="Sanae Takaichi - Wikipedia",
            snippet="Sanae Takaichi is a Japanese politician who has been Prime "
                    "Minister of Japan since October 2025. She is the first woman "
                    "to hold the position.",
            url="https://en.wikipedia.org/wiki/Sanae_Takaichi",
        )
    ],
)


def test_web_search_entity_hint_extracts_person_and_strips_wikipedia_suffix():
    hint = server._web_search_entity_hint(_JAPAN_PRESIDENT_ANSWER_TEXT)
    assert hint == ("Sanae Takaichi", "person")


def test_web_search_entity_hint_returns_topic_without_person_pronoun():
    text = render_web_answer(
        "Where is Harvard University?",
        [WebSearchResult(
            title="Harvard University",
            snippet="Harvard University is located in Cambridge, Massachusetts.",
            url="https://example.com/harvard",
        )],
    )
    assert server._web_search_entity_hint(text) == ("Harvard University", "topic")


def test_web_search_entity_hint_none_without_source_line():
    assert server._web_search_entity_hint("No relevant information found.") is None


def test_record_turn_then_pronoun_resolves_to_web_search_person(tmp_path):
    server._startup("pump-dry-run")
    context = ConversationContext()

    answer = AssistantAnswer(
        question="Who is the current president of Japan?",
        decision="answer",
        route="current_live_request",
        answer_text=_JAPAN_PRESIDENT_ANSWER_TEXT,
        overlay_mode="pump-dry-run",
        supported_by_context=True,
        support_kind="web_search_result",
        risk_flags=["current_live", "web_search_live", "source_qualified_volatile"],
        source_system="web_search",
    )
    semantic_query = SemanticQuery(
        entity_a=None, entity_b=None, relation_intent=None,
        unknown_position="subject", query_type="lookup", confidence=0.9,
    )
    server._record_turn(
        context,
        question="Who is the current president of Japan?",
        semantic_query=semantic_query,
        answer=answer,
    )

    assert context.turns[-1].primary_entity == "Sanae Takaichi"
    assert context.turns[-1].entity_types == {"Sanae Takaichi": "person"}

    resolution = resolve_coreferences("Where was she born?", context, server._surface_index)

    assert resolution.unresolved_reference is None
    assert resolution.resolved_question == "Where was Sanae Takaichi born?"
