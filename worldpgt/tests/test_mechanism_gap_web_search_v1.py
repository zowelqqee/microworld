"""Tests for the mechanism-gap reasoning + targeted web-search augmentation.

For "how does X work?" questions where the overlay has a supported profile
but no mechanism evidence, the orchestrator used to narrate unrelated
supported facts and never admit the gap. It now detects the missing
mechanism role via the same reasoning trace used elsewhere
(``reason_over_plan``), tries a scoped live web search for it, and only
falls back to an honest admission when no web result is found — it must
never silently drop the already-supported facts already in the answer.
"""

from __future__ import annotations

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.web_search import WebSearchResult
from worldpgt.tests.test_assistant_surface_v1 import _FakeWebSearchProvider


def _starlink_answer(provider=None, *, web_search_enabled=None):
    orchestrator = AnswerOrchestrator(
        "pump-dry-run",
        web_search_provider=provider,
        web_search_enabled=web_search_enabled,
    )
    return orchestrator.answer("How does Starlink work?", web_search_enabled=web_search_enabled)


def test_mechanism_gap_is_admitted_without_web_search() -> None:
    answer = _starlink_answer(web_search_enabled=False)

    assert answer.decision == "answer"
    assert "Starlink" in answer.answer_text
    assert "the parts and steps that make it work" in answer.answer_text
    assert "operating mechanism is still missing" not in answer.answer_text
    # The already-supported profile facts must still be present, not dropped.
    assert "SpaceX" in answer.answer_text


def test_mechanism_gap_is_filled_by_targeted_web_search() -> None:
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Starlink - Wikipedia",
            snippet=(
                "Starlink is a satellite internet constellation. The network "
                "consists of satellites in low Earth orbit that communicate "
                "with user terminals and ground stations."
            ),
            url="https://en.wikipedia.org/wiki/Starlink",
        )
    ])

    answer = _starlink_answer(provider, web_search_enabled=True)

    assert answer.decision == "answer"
    assert "live web search" in answer.answer_text
    assert "low earth orbit" in answer.answer_text.lower()
    assert "ground stations" in answer.answer_text.lower()
    assert "unverified" in answer.answer_text.lower()
    assert provider.queries, "the mechanism-gap path must actually call the provider"


def test_mechanism_gap_augmentation_only_fires_for_how_it_works_questions() -> None:
    orchestrator = AnswerOrchestrator("pump-dry-run", web_search_enabled=False)
    answer = orchestrator.answer("Tell me about Starlink.")

    assert "operating mechanism is still missing" not in answer.answer_text
    assert "live web search" not in answer.answer_text


def test_mechanism_gap_web_search_only_triggers_when_a_role_is_actually_missing() -> None:
    """The provider must not be called at all when there is no detected gap."""

    provider = _FakeWebSearchProvider([
        WebSearchResult(title="Unrelated", snippet="Unrelated snippet.", url="https://example.com")
    ])
    orchestrator = AnswerOrchestrator(
        "pump-dry-run", web_search_provider=provider, web_search_enabled=True
    )
    orchestrator.answer("Tell me about Starlink.")

    assert provider.queries == []
