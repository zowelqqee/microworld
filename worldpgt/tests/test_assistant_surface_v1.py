"""Deterministic tests for Microworld Assistant Surface v1.

Covers routing, context support, safety audits, overlay modes, the CLI, the
benchmark runner, and the safety contract (no memory/overlay writes, no
neural/GPT/training/embedding imports, nanogpt untouched).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.answer_style import resolve_answer_style
from worldpgt.assistant_surface.assistant_renderer import render
from worldpgt.assistant_surface.context_selector import (
    ACCEPTED_OVERLAY_PATH,
    PROMOTED_OVERLAY_PATH,
    SNAPSHOT_DRY_RUN_OVERLAY_PATH,
    ContextSelector,
)
from worldpgt.assistant_surface.question_router import route
from worldpgt.assistant_surface.surface_validator import validate_answer
from worldpgt.assistant_surface.types import FACTUAL_SUPPORT_KINDS
from worldpgt.assistant_surface.web_search import WebSearchResult
from worldpgt.community_context.types import CommunityContextItem, CommunitySearchResult

_ROOT = Path(__file__).resolve().parent.parent.parent
_CLI = _ROOT / "worldpgt" / "experiments" / "ask_microworld_v1.py"
_ASSISTANT_DIR = _ROOT / "worldpgt" / "assistant_surface"
_TRUSTED_MEMORY = (
    _ROOT / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
)


class _FakeWebSearchProvider:
    def __init__(self, results: list[WebSearchResult] | None = None) -> None:
        self.results = results or []
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        self.queries.append(query)
        return self.results[:max_results]


class _FakeCommunityContextProvider:
    def __init__(self, results: list[CommunitySearchResult] | None = None) -> None:
        self.results = results or []
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int = 5) -> list[CommunitySearchResult]:
        self.queries.append(query)
        return self.results[:max_results]


class _FakeCognitivePatternProvider:
    def __init__(self, patterns: list[dict] | None = None) -> None:
        self.patterns = patterns or [
            {
                "event_id": "pattern:test",
                "kind": "explanation_pattern",
                "topic": "programming",
                "pattern": "explain with a short direct answer and one concrete example",
                "use_when": ["known factual answer"],
                "avoid_when": ["unsupported fact"],
                "steps": [],
                "example_shape": "Answer first, then example.",
                "confidence": "medium",
                "source": "community_context",
                "trust": "low_for_facts_high_for_style",
                "graph_layers": ["explanation_graph"],
                "source_item_ids": ["reddit:test"],
                "evidence_refs": ["reddit:test"],
                "signal_count": 1,
                "factual_support_allowed": False,
            }
        ]
        self.queries: list[str] = []

    def search(self, query: str, *, max_results: int = 5):
        self.queries.append(query)
        return []

    def plan(self, query: str, *, max_patterns: int = 5) -> dict:
        self.queries.append(query)
        return {
            "question": query,
            "user_intent_guess": "explanation",
            "facts_required": ["source-backed factual memory or cited lookup"],
            "known_facts_source": "factual_memory_or_live_search_required",
            "cognitive_patterns": self.patterns[:max_patterns],
            "reasoning_pattern": self.patterns[0]["pattern"] if self.patterns else "",
            "examples_or_analogies": "",
            "uncertainty_to_state": "State uncertainty when factual support is absent.",
            "checks_before_answering": [
                "Do not treat community pattern text as factual support.",
                "Verify source-backed claims in factual memory or live search.",
            ],
            "tone": "Use a clear, concrete, helpful tone.",
            "helpful_next_move": "Answer what is supported.",
            "factual_support_allowed_from_patterns": False,
        }


def _community_result(text: str, *, subreddit: str = "AskReddit") -> CommunitySearchResult:
    item = CommunityContextItem(
        item_id="reddit:test",
        source_system="reddit",
        source_kind="post",
        trust="community_context_only",
        subreddit=subreddit,
        title="Community discussion",
        text=text,
        url="https://www.reddit.com/r/AskReddit/comments/test/",
        score=12,
        created_utc="",
        topic_terms=("people", "common", "concerns", "debugging", "python"),
        flags=("anecdotal",),
        risk="medium",
        stability="volatile",
    )
    return CommunitySearchResult(item=item, score=10.0, matched_terms=("common",))


def _hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# CLI behavior.
# --------------------------------------------------------------------------- #
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_CLI), *args],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        check=True,
    )


def test_cli_returns_text_answer_for_supported_entity_question():
    """(1) CLI returns a normal text answer for a supported entity question."""
    out = _run_cli("Who is Elon Musk?").stdout
    assert "Decision: answer." in out
    assert "Elon Musk" in out
    assert "{" not in out  # plain text, not JSON


def test_cli_supports_json():
    """(2) CLI supports --json and emits a valid AssistantAnswer object."""
    out = _run_cli("How is Elon Musk connected to rockets?", "--json").stdout
    obj = json.loads(out)
    assert obj["decision"] == "answer"
    assert obj["route"] == "connection_path"
    assert obj["support_kind"] == "explicit_connection_path"
    assert obj["safe_for_general_runtime"] is False


def test_cli_accepts_cognitive_patterns_path(tmp_path):
    patterns = tmp_path / "cognitive_pattern_events.json"
    patterns.write_text(
        json.dumps(
            [
                {
                    "event_id": "pattern:cli",
                    "kind": "explanation_pattern",
                    "topic": "spacex",
                    "pattern": "answer founded questions with a short direct answer and one concrete example",
                    "source": "community_context",
                    "trust": "low_for_facts_high_for_style",
                    "factual_support_allowed": False,
                }
            ]
        ),
        encoding="utf-8",
    )

    out = _run_cli(
        "Who founded SpaceX?",
        "--cognitive-patterns",
        str(patterns),
        "--json",
    ).stdout
    obj = json.loads(out)

    assert obj["decision"] == "answer"
    assert obj["support_kind"] in FACTUAL_SUPPORT_KINDS
    assert obj["source_system"] == "entity_qa"
    assert "Short answer:" in obj["answer_text"]
    assert "cognitive_pattern_surface" in obj["risk_flags"]
    assert obj["trace"]["cognitive_plan"]["factual_support_allowed_from_patterns"] is False


# --------------------------------------------------------------------------- #
# Router intents (3-9).
# --------------------------------------------------------------------------- #
def test_router_entity_definition():
    assert route("Who is Elon Musk?").intent == "entity_definition"


@pytest.mark.parametrize(
    "question",
    [
        "How does Starlink work?",
        "What do you know about Blue Origin?",
        "Explain SpaceX in simple terms.",
        "Describe Tesla in plain English.",
        "Summarize Starlink briefly.",
    ],
)
def test_router_open_synthesis_to_entity_definition(question):
    r = route(question)

    assert r.intent == "entity_definition"
    assert r.notes == "semantic open synthesis query"


def test_router_entity_relation():
    assert route("What does SpaceX develop?").intent == "entity_relation"


def test_router_unsupported_relation_tail_is_not_definition():
    r = route("Who was Richard Nixon married to?")

    assert r.intent == "unknown_or_unsupported"
    assert r.notes == "unsupported relation lookup"


def test_router_connection_path():
    assert route("How is Elon Musk connected to rockets?").intent == "connection_path"


def test_router_current_live_request():
    r = route("What is Tesla's current stock price?")
    assert r.intent == "current_live_request"
    assert r.is_hard_safety is True


def test_router_private_sensitive_request():
    r = route("What is Elon Musk's private email?")
    assert r.intent == "private_sensitive_request"
    assert r.is_hard_safety is True


def test_router_relation_inversion():
    r = route("Did SpaceX found Elon Musk?")
    assert r.intent == "relation_inversion"
    assert r.is_hard_safety is True


def test_router_unsupported_universal():
    r = route("If SpaceX makes rockets, are all rockets SpaceX products?")
    assert r.intent == "unsupported_universal"
    assert r.is_hard_safety is True


# --------------------------------------------------------------------------- #
# Context pack + support (10-14).
# --------------------------------------------------------------------------- #
def test_context_pack_built_for_every_request():
    """(10) A context pack is built (and validated) for every request."""
    selector = ContextSelector("promoted")
    for q in [
        "Who is Elon Musk?",
        "What is Tesla's current stock price?",
        "Did SpaceX found Elon Musk?",
        "asldkfj qwoeiru?",
    ]:
        pack, summary = selector.select(q)
        assert pack is not None
        assert summary.pack_valid is True


def test_supported_answer_has_context_support():
    """(11) A supported answer has context support and a factual support kind."""
    a = AnswerOrchestrator("promoted").answer("How is Elon Musk connected to rockets?")
    assert a.decision == "answer"
    assert a.supported_by_context is True
    assert a.support_kind in FACTUAL_SUPPORT_KINDS


def test_supported_negative_answer_has_explicit_support():
    a = AnswerOrchestrator("promoted").answer("Is Elon Musk an organization?")

    assert a.decision == "no"
    assert a.supported_by_context is True
    assert a.support_kind == "explicit_type_contradiction"
    assert a.support_kind in FACTUAL_SUPPORT_KINDS
    assert "Decision: no." in render(a)
    assert validate_answer(a) == []


def test_pump_dry_run_open_synthesis_answers_with_definition_support():
    a = AnswerOrchestrator("pump-dry-run").answer("What do you know about Blue Origin?")

    assert a.decision == "answer"
    assert a.support_kind == "stable_definition"
    assert "Blue Origin" in (a.answer_text or "")


def test_pump_dry_run_open_synthesis_uses_alias_definition():
    a = AnswerOrchestrator("pump-dry-run").answer("Tell me about Ray Kroc.")

    assert a.decision == "answer"
    assert a.support_kind == "stable_definition"
    assert "Ray Kroc is an American businessman" in (a.answer_text or "")
    assert "an other" not in (a.answer_text or "")


def test_answer_style_normalizes_russian_brief_about_prompt():
    result = resolve_answer_style("коротко про SpaceX")

    assert result.question == "Tell me about SpaceX."
    assert result.answer_style == "brief"


def test_answer_style_shortens_open_synthesis():
    normal = AnswerOrchestrator("pump-dry-run").answer("Tell me about SpaceX.")
    brief = AnswerOrchestrator("pump-dry-run").answer("коротко про SpaceX")

    assert brief.decision == "answer"
    assert len(brief.answer_text or "") < len(normal.answer_text or "")
    assert "SpaceX is an aerospace manufacturer" in (brief.answer_text or "")


def test_cognitive_surface_shapes_universal_explain_entity_prompt():
    cognitive_patterns = _ROOT / "worldpgt" / "experiments" / "community_context_v1" / "cognitive_pattern_events.json"
    a = AnswerOrchestrator(
        "pump-dry-run",
        cognitive_patterns_path=str(cognitive_patterns),
        cognitive_patterns_enabled=True,
    ).answer("Explain SpaceX in simple terms.")

    assert a.decision == "answer"
    assert a.route == "entity_definition"
    assert a.support_kind in FACTUAL_SUPPORT_KINDS
    assert a.source_system == "entity_qa"
    assert "Short answer:" in a.answer_text
    assert "SpaceX" in a.answer_text
    assert "The factual claim above still comes from Microworld memory." in a.answer_text
    assert "The cognitive pattern only shapes the explanation; it is not extra evidence." in a.answer_text
    assert "cognitive_pattern_surface" in a.risk_flags
    assert a.trace is not None
    assert a.trace.cognitive_plan is not None
    assert a.trace.cognitive_plan["factual_support_allowed_from_patterns"] is False
    assert validate_answer(a) == []


def test_answer_style_simple_keeps_how_question_parseable():
    answer = AnswerOrchestrator("pump-dry-run").answer(
        "простыми словами How does Starlink work?"
    )

    assert answer.decision == "answer"
    assert "Starlink is" in (answer.answer_text or "")


def test_audit_request_does_not_answer_as_fact():
    """(12) An audit request is not rendered as a stable fact."""
    a = AnswerOrchestrator("promoted").answer("What is Tesla's current stock price?")
    assert a.decision == "audit"
    assert a.supported_by_context is False
    assert "Decision: audit." in render(a)


def test_current_live_request_can_use_explicit_web_search_provider():
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Tesla Inc. stock quote",
            snippet="Tesla stock was trading at 123.45 USD in the latest market quote.",
            url="https://example.com/tesla-stock",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("What is Tesla's current stock price?")

    assert a.decision == "answer"
    assert a.route == "current_live_request"
    assert a.support_kind == "web_search_result"
    assert a.source_system == "web_search"
    assert a.supported_by_context is True
    assert "not Microworld memory" in a.answer_text
    assert "https://example.com/tesla-stock" in a.answer_text
    assert "web_search_live" in a.risk_flags
    assert provider.queries == ["What is Tesla's current stock price?"]
    assert validate_answer(a) == []


def test_current_office_request_can_use_explicit_web_search_provider():
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="President of France",
            snippet="The current president of France is Emmanuel Macron.",
            url="https://example.com/france-president",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("Who is the current president of France?")

    assert a.decision == "answer"
    assert a.route == "current_live_request"
    assert a.support_kind == "web_search_result"
    assert "Emmanuel Macron" in a.answer_text
    assert validate_answer(a) == []


def test_current_office_request_rejects_generic_office_page():
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="President of the United States - Wikipedia",
            snippet=(
                "The president of the United States is the head of state and "
                "head of government of the United States."
            ),
            url="https://example.com/president-office",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("Who is the current president of the US?")

    assert a.decision == "audit"
    assert a.route == "current_live_request"
    assert "generic" not in a.answer_text.lower()
    assert validate_answer(a) == []


def test_private_request_never_reaches_web_search_provider():
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Bad",
            snippet="Should not be used.",
            url="https://example.com/private",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("What is Elon Musk's phone number?")

    assert a.decision == "audit"
    assert a.route == "private_sensitive_request"
    assert a.source_system == "safety_gate"
    assert provider.queries == []
    assert validate_answer(a) == []


def test_web_search_second_identical_call_uses_cache(tmp_path):
    from worldpgt.web_search.live_cache import LiveSearchCache

    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="President of France",
            snippet="The current president of France is Emmanuel Macron.",
            url="https://example.com/france-president",
        )
    ])
    cache = LiveSearchCache(tmp_path / "live_cache.json")
    orch = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
        live_cache=cache,
    )

    first = orch.answer("Who is the current president of France?")
    second = orch.answer("Who is the current president of France?")

    assert provider.queries == ["Who is the current president of France?"]
    assert first.decision == "answer" and second.decision == "answer"
    assert "Emmanuel Macron" in first.answer_text
    assert "Emmanuel Macron" in second.answer_text
    assert "not Microworld memory" in second.answer_text
    assert "web_search_cached" not in first.risk_flags
    assert "web_search_cached" in second.risk_flags
    assert validate_answer(first) == []
    assert validate_answer(second) == []


def test_differently_worded_followup_reuses_learned_entity_without_network(tmp_path):
    """The core 'learn once, remember' contract: a SECOND question with
    completely different wording, but naming the same entity, must be
    answered from the cached article — never a second network call."""
    from worldpgt.web_search.live_cache import LiveSearchCache

    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Sanae Takaichi - Wikipedia",
            snippet="Sanae Takaichi is a Japanese politician, born 7 March 1961, "
                    "who has been Prime Minister of Japan since October 2025.",
            url="https://en.wikipedia.org/wiki/Sanae_Takaichi",
        )
    ])
    cache = LiveSearchCache(tmp_path / "live_cache.json")
    orch = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
        live_cache=cache,
    )

    first = orch.answer("Who is Sanae Takaichi?")
    second = orch.answer("When was Sanae Takaichi born?")  # different wording

    assert provider.queries == ["Who is Sanae Takaichi?"]  # no second network call
    assert first.decision == "answer" and second.decision == "answer"
    assert "Sanae Takaichi" in second.answer_text
    assert "web_search_cached" in second.risk_flags
    assert validate_answer(first) == []
    assert validate_answer(second) == []


def test_web_search_deadline_sec_is_passed_to_default_provider():
    """AnswerOrchestrator(web_search_deadline_sec=...) must reach the
    lazily-constructed default CompositeSearchProvider — this is the knob for
    the latency/coverage tradeoff (see composite.py's measured deadline
    comparison), so it must actually take effect, not just be accepted."""
    from unittest.mock import patch
    import worldpgt.web_search.composite as composite_mod

    captured_deadlines = []

    class _FakeComposite:
        def __init__(self, deadline_sec=None):
            captured_deadlines.append(deadline_sec)

        def search(self, query, *, max_results=3):
            return []

    with patch.object(composite_mod, "CompositeSearchProvider", _FakeComposite):
        orch = AnswerOrchestrator(
            "promoted", web_search_enabled=True, web_search_deadline_sec=9.5,
        )
        orch.answer("What is Tesla's current stock price?")

    assert captured_deadlines == [9.5]


def test_unknown_entity_definition_falls_back_to_web_search_when_enabled():
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Sanae Takaichi - Wikipedia",
            snippet="Sanae Takaichi is a Japanese politician who has been Prime "
                    "Minister of Japan since October 2025.",
            url="https://en.wikipedia.org/wiki/Sanae_Takaichi",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("Who is Sanae Takaichi?")

    assert a.decision == "answer"
    assert a.route == "entity_definition"
    assert a.support_kind == "web_search_result"
    assert a.source_system == "web_search"
    assert "Sanae Takaichi" in a.answer_text
    assert "not Microworld memory" in a.answer_text
    assert provider.queries == ["Who is Sanae Takaichi?"]
    assert validate_answer(a) == []


def test_unknown_entity_definition_still_audits_when_web_search_disabled():
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Sanae Takaichi - Wikipedia",
            snippet="Sanae Takaichi is a Japanese politician.",
            url="https://en.wikipedia.org/wiki/Sanae_Takaichi",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=False,
    ).answer("Who is Sanae Takaichi?")

    assert a.decision == "audit"
    assert a.route == "entity_definition"
    assert provider.queries == []
    assert validate_answer(a) == []


def test_missing_knowledge_can_use_community_context_without_fact_support():
    provider = _FakeCommunityContextProvider([
        _community_result(
            "People often describe learning Python through small projects, "
            "reading error messages, and asking focused debugging questions.",
            subreddit="learnpython",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        community_context_provider=provider,
        community_context_enabled=True,
    ).answer("What are common concerns when people learn python debugging?")

    assert a.decision == "answer"
    assert a.support_kind == "safe_policy_answer"
    assert a.source_system == "community_context"
    assert a.supported_by_context is True
    assert "A practical debugging rhythm" in a.answer_text
    assert "not factual support" in a.answer_text
    assert "Do not promote it to stable facts" in a.answer_text
    assert "community_context_only" in a.risk_flags
    assert provider.queries == ["What are common concerns when people learn python debugging?"]
    assert validate_answer(a) == []


def test_community_tone_applies_to_known_overlay_answers():
    provider = _FakeCommunityContextProvider([
        _community_result("People prefer direct answers with a plain-language follow-up.")
    ])

    a = AnswerOrchestrator(
        "promoted",
        community_context_provider=provider,
        community_context_enabled=True,
    ).answer("Who founded SpaceX?")

    assert a.decision == "answer"
    assert a.support_kind != "web_search_result"
    assert a.source_system == "entity_qa"
    assert "Short version:" in a.answer_text
    assert "SpaceX" in a.answer_text
    assert "Elon Musk" in a.answer_text
    assert "community_style_tone" in a.risk_flags
    assert "phrasing is community-shaped" in a.answer_text
    assert validate_answer(a) == []


def test_cognitive_patterns_plan_known_answer_without_becoming_fact_support():
    provider = _FakeCognitivePatternProvider()

    a = AnswerOrchestrator(
        "promoted",
        cognitive_pattern_provider=provider,
        cognitive_patterns_enabled=True,
    ).answer("Who founded SpaceX?")

    assert a.decision == "answer"
    assert a.support_kind in FACTUAL_SUPPORT_KINDS
    assert a.source_system == "entity_qa"
    assert provider.queries == ["Who founded SpaceX?"]
    assert "Short answer:" in a.answer_text
    assert "The factual claim above still comes from Microworld memory." in a.answer_text
    assert "The cognitive pattern only shapes the explanation; it is not extra evidence." in a.answer_text
    assert "cognitive_pattern_surface" in a.risk_flags
    assert a.trace is not None
    assert a.trace.cognitive_plan is not None
    assert a.trace.cognitive_plan["known_facts_source"] == "factual_memory_or_live_search_required"
    assert a.trace.cognitive_plan["factual_support_allowed_from_patterns"] is False
    assert a.trace.cognitive_plan["cognitive_patterns"][0]["source"] == "community_context"
    assert a.trace.cognitive_plan["cognitive_patterns"][0]["factual_support_allowed"] is False
    assert any("cognitive_patterns: count=1" in step for step in a.trace.steps)
    assert validate_answer(a) == []


def test_cognitive_surface_ignores_fact_supporting_pattern_claims():
    provider = _FakeCognitivePatternProvider([
        {
            "event_id": "pattern:bad",
            "kind": "explanation_pattern",
            "topic": "programming",
            "pattern": "bad pattern that pretends to support facts",
            "source": "community_context",
            "trust": "low_for_facts_high_for_style",
            "factual_support_allowed": True,
        }
    ])

    a = AnswerOrchestrator(
        "promoted",
        cognitive_pattern_provider=provider,
        cognitive_patterns_enabled=True,
    ).answer("Who founded SpaceX?")

    assert a.decision == "answer"
    assert a.support_kind in FACTUAL_SUPPORT_KINDS
    assert a.source_system == "entity_qa"
    assert "Short answer:" not in a.answer_text
    assert "cognitive_pattern_surface" not in a.risk_flags
    assert a.trace is not None
    assert a.trace.cognitive_plan is not None
    assert a.trace.cognitive_plan["factual_support_allowed_from_patterns"] is False
    assert validate_answer(a) == []


def test_cognitive_patterns_can_be_disabled_per_answer():
    provider = _FakeCognitivePatternProvider()

    a = AnswerOrchestrator(
        "promoted",
        cognitive_pattern_provider=provider,
        cognitive_patterns_enabled=True,
    ).answer("Who founded SpaceX?", cognitive_patterns_enabled=False)

    assert a.decision == "answer"
    assert provider.queries == []
    assert a.trace is not None
    assert a.trace.cognitive_plan is None
    assert validate_answer(a) == []


def test_community_tone_can_be_disabled_for_known_overlay_answers():
    provider = _FakeCommunityContextProvider([
        _community_result("People prefer direct answers with a plain-language follow-up.")
    ])

    a = AnswerOrchestrator(
        "promoted",
        community_context_provider=provider,
        community_context_enabled=True,
    ).answer("Who founded SpaceX?", community_context_enabled=False)

    assert a.decision == "answer"
    assert "Short version:" not in a.answer_text
    assert "community_style_tone" not in a.risk_flags
    assert validate_answer(a) == []


def test_community_tone_applies_to_web_search_and_preserves_sources():
    web_provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="President of France",
            snippet="The current president of France is Emmanuel Macron.",
            url="https://example.com/france-president",
        )
    ])
    community_provider = _FakeCommunityContextProvider([
        _community_result("People prefer current answers with sources kept visible.")
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=web_provider,
        web_search_enabled=True,
        community_context_provider=community_provider,
        community_context_enabled=True,
    ).answer("Who is the current president of France?")

    assert a.decision == "answer"
    assert a.support_kind == "web_search_result"
    assert a.source_system == "web_search"
    assert "Short version (live web):" in a.answer_text
    assert "Emmanuel Macron" in a.answer_text
    assert "https://example.com/france-president" in a.answer_text
    assert "not Microworld memory" in a.answer_text
    assert "community_style_tone" in a.risk_flags
    assert validate_answer(a) == []


def test_community_context_does_not_bypass_private_safety():
    provider = _FakeCommunityContextProvider([
        _community_result("This should not be used for private information.")
    ])

    a = AnswerOrchestrator(
        "promoted",
        community_context_provider=provider,
        community_context_enabled=True,
    ).answer("What is Elon Musk's private email?")

    assert a.decision == "audit"
    assert a.route == "private_sensitive_request"
    assert a.source_system == "safety_gate"
    assert provider.queries == []
    assert validate_answer(a) == []


def test_cognitive_patterns_do_not_bypass_private_safety():
    provider = _FakeCognitivePatternProvider()

    a = AnswerOrchestrator(
        "promoted",
        cognitive_pattern_provider=provider,
        cognitive_patterns_enabled=True,
    ).answer("What is Elon Musk's private email?")

    assert a.decision == "audit"
    assert a.route == "private_sensitive_request"
    assert a.source_system == "safety_gate"
    assert provider.queries == []
    assert a.trace is not None
    assert a.trace.cognitive_plan is None
    assert validate_answer(a) == []


def test_private_request_still_never_reaches_web_search_fallback():
    """Hard-safety audits (private/inversion/universal) must never be
    overridden by the new missing_knowledge -> web-search fallback."""
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Bad",
            snippet="Should not be used.",
            url="https://example.com/private",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("What is Elon Musk's phone number?")

    assert a.decision == "audit"
    assert a.route == "private_sensitive_request"
    assert provider.queries == []
    assert validate_answer(a) == []


def test_relation_inversion_still_never_reaches_web_search_fallback():
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Bad",
            snippet="Should not be used.",
            url="https://example.com/inversion",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("Did SpaceX found Elon Musk?")

    assert a.decision == "audit"
    assert a.route == "relation_inversion"
    assert provider.queries == []
    assert validate_answer(a) == []


def test_known_overlay_entity_answer_never_reaches_web_search_fallback():
    """A supported overlay answer must not even attempt the web-search
    fallback (it only triggers on decision == 'audit')."""
    provider = _FakeWebSearchProvider([
        WebSearchResult(
            title="Bad",
            snippet="Should not be used.",
            url="https://example.com/spacex",
        )
    ])

    a = AnswerOrchestrator(
        "promoted",
        web_search_provider=provider,
        web_search_enabled=True,
    ).answer("Who founded SpaceX?")

    assert a.decision == "answer"
    assert a.support_kind != "web_search_result"
    assert provider.queries == []


def test_absent_fact_does_not_become_negative_answer():
    a = AnswerOrchestrator("promoted").answer("Is SpaceX profitable?")

    assert a.decision == "audit"
    assert a.supported_by_context is False
    assert "Decision: audit." in render(a)


def test_weak_only_context_cannot_support_factual_answer():
    """(13) Weak-only context never supports a factual answer."""
    orch = AnswerOrchestrator("promoted")
    for q in [
        "How is Elon Musk connected to rockets?",
        "Who is Elon Musk?",
        "What does SpaceX develop?",
    ]:
        a = orch.answer(q)
        if a.decision == "answer" and a.support_kind in FACTUAL_SUPPORT_KINDS:
            ctx = a.trace.context_summary
            # A factual answer must rest on real support, never weak-only.
            assert ctx["has_weak_only"] is False


def test_source_qualified_volatile_fact_not_stable():
    """(14) A source-qualified volatile fact never becomes a stable answer."""
    a = AnswerOrchestrator("promoted").answer(
        "According to Forbes, what is Elon Musk's estimated net worth?"
    )
    assert a.decision == "answer"
    assert a.support_kind == "source_qualified_fact"
    assert a.support_kind not in {
        "stable_definition",
        "stable_relation",
        "explicit_connection_path",
    }
    assert "source_qualified_volatile" in a.risk_flags


# --------------------------------------------------------------------------- #
# Hard-safety audits (15-18).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question,expected_route",
    [
        ("What is Tesla's current stock price?", "current_live_request"),
        ("What is Elon Musk's private email?", "private_sensitive_request"),
        ("Did SpaceX found Elon Musk?", "relation_inversion"),
        ("If SpaceX makes rockets, are all rockets SpaceX products?", "unsupported_universal"),
    ],
)
def test_hard_safety_requests_audit(question, expected_route):
    a = AnswerOrchestrator("promoted").answer(question)
    assert a.route == expected_route
    assert a.decision == "audit"
    assert a.supported_by_context is False


# --------------------------------------------------------------------------- #
# Overlay modes (19).
# --------------------------------------------------------------------------- #
def test_snapshot_dry_run_marked_as_proposal():
    a = AnswerOrchestrator("snapshot-dry-run").answer("Who is Elon Musk?")
    assert a.overlay_mode == "snapshot-dry-run"
    rendered = render(a)
    assert "snapshot-dry-run proposal, not accepted memory" in rendered


# --------------------------------------------------------------------------- #
# Safety contract (20-25).
# --------------------------------------------------------------------------- #
def test_overlay_files_not_modified_by_answering():
    """(20) Answering does not modify accepted/promoted/snapshot overlays."""
    files = [ACCEPTED_OVERLAY_PATH, PROMOTED_OVERLAY_PATH, SNAPSHOT_DRY_RUN_OVERLAY_PATH]
    before = {str(f): _hash(f) for f in files}
    for mode in ("accepted", "promoted", "snapshot-dry-run"):
        orch = AnswerOrchestrator(mode)
        for q in ["Who is Elon Musk?", "Did SpaceX found Elon Musk?", "What is SpaceX?"]:
            orch.answer(q)
    after = {str(f): _hash(f) for f in files}
    assert before == after


def test_benchmark_summary_safety_flags(tmp_path):
    """(21-23) Summary asserts runtime/trusted-memory/network safety flags."""
    from worldpgt.experiments.run_assistant_surface_v1 import run

    summary = run(outdir=tmp_path)
    assert summary["runtime_behavior_modified"] is False
    assert summary["trusted_memory_modified"] is False
    assert summary["network_calls"] is False
    assert summary["safe_for_general_runtime"] is False


def test_static_ui_hides_internal_decision_and_support_labels():
    html = (_ROOT / "worldpgt" / "api" / "static" / "index.html").read_text()

    assert "badge" not in html
    assert "data.decision" not in html
    assert "data.support" not in html


def test_no_neural_or_network_imports():
    """(24) No neural/GPT/training/embedding/network imports in the package."""
    banned = ["torch", "tensorflow", "openai", "transformers", "numpy.random",
              "backprop", "embedding", "requests", "urllib.request", "socket", "httpx"]
    for py in _ASSISTANT_DIR.glob("*.py"):
        text = py.read_text(encoding="utf-8").lower()
        for token in banned:
            assert f"import {token}" not in text, f"{py.name} imports {token}"
            assert f"from {token}" not in text, f"{py.name} imports from {token}"


def test_nanogpt_untouched():
    """(25) The assistant surface never references nanogpt."""
    for py in _ASSISTANT_DIR.glob("*.py"):
        assert "nanogpt" not in py.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Benchmark artifacts + critical pass (26-27).
# --------------------------------------------------------------------------- #
def test_benchmark_writes_outputs_and_report(tmp_path):
    from worldpgt.experiments.run_assistant_surface_v1 import run

    run(outdir=tmp_path)
    for name in (
        "assistant_surface_outputs.csv",
        "assistant_surface_outputs.json",
        "assistant_surface_summary.json",
        "assistant_surface_report.json",
    ):
        assert (tmp_path / name).exists(), f"missing artifact: {name}"


def test_benchmark_all_critical_passed(tmp_path):
    from worldpgt.experiments.run_assistant_surface_v1 import run

    s = run(outdir=tmp_path)
    assert s["unsafe_answer_count"] == 0
    assert s["answer_without_context_support_count"] == 0
    assert s["weak_link_false_support_count"] == 0
    assert s["volatile_false_stable_count"] == 0
    assert s["current_live_false_support_count"] == 0
    assert s["private_false_support_count"] == 0
    assert s["inversion_false_support_count"] == 0
    assert s["universal_false_support_count"] == 0
    assert s["all_critical_passed"] is True


def test_every_answer_passes_invariants():
    """All produced answers satisfy the surface safety invariants."""
    orch = AnswerOrchestrator("promoted")
    questions = [
        "Who is Elon Musk?",
        "What does SpaceX develop?",
        "How is Elon Musk connected to rockets?",
        "According to Forbes, what is Elon Musk's estimated net worth?",
        "Why is Forbes linked to Elon Musk?",
        "What does weak context link mean?",
        "What is Tesla's current stock price?",
        "What is Elon Musk's private email?",
        "Did SpaceX found Elon Musk?",
        "If SpaceX makes rockets, are all rockets SpaceX products?",
    ]
    for q in questions:
        a = orch.answer(q)
        assert validate_answer(a) == [], f"invariant failure for: {q}"
