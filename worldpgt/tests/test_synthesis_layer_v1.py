"""Tests for the synthesis layer (Entity QA layer 3).

The synthesis layer answers open questions ("Tell me about X") by gathering
every relevant fact about an entity from the overlay graph, grouped by type and
confidence tier (VERIFIED / SNAPSHOT / UNKNOWN). It never invents facts.

All tests are deterministic and offline. No network access. No modification of
sense_memory.py, accepted_knowledge_memory_v1.json, the overlay semantics, the
planner thresholds, or the validators.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from worldpgt.entity_qa.entity_answer_planner import EntityAnswerPlanner
from worldpgt.entity_qa.entity_answer_renderer import render
from worldpgt.entity_qa.entity_question_analyzer import analyze
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.entity_qa.synthesis_engine import synthesize
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_OVERLAY_JSON = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"


@pytest.fixture(scope="module")
def provider():
    return WikiMemoryOverlayProvider(_OVERLAY_JSON)


@pytest.fixture(scope="module")
def planner(provider):
    return EntityAnswerPlanner(provider=provider)


def _answer(planner, question: str):
    analyzed = analyze(question)
    plan = planner.plan(analyzed)
    return analyzed, plan, render(plan)


# ---------------------------------------------------------------------------
# 1. Parser routes open questions to open_synthesis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Tell me about SpaceX.",
        "Tell me about Elon Musk.",
        "What do you know about Blue Origin?",
        "How does Starlink work?",
        "What can you tell me about Tesla?",
        "Describe SpaceX.",
        "Give me an overview of Blue Origin.",
    ],
)
def test_parser_detects_open_synthesis(question):
    sem = parse_semantic_query(question)
    assert sem.query_type == "open_synthesis", question
    assert sem.entity_a, question


def test_analyzer_routes_open_synthesis():
    analyzed = analyze("Tell me about SpaceX.")
    assert analyzed.intent == "open_synthesis"
    assert analyzed.subject == "SpaceX"


# ---------------------------------------------------------------------------
# 2. Demo answers: every clause is backed by a real overlay fact
# ---------------------------------------------------------------------------


def test_spacex_synthesis(planner):
    _, plan, answer = _answer(planner, "Tell me about SpaceX.")
    assert plan.decision == "answer"
    low = answer.lower()
    assert "aerospace manufacturer and space transportation company" in low
    assert "develops rockets and spacecraft" in low
    assert "produces falcon rockets and dragon spacecraft" in low
    assert "founded by elon musk" in low
    assert "[verified" in low


def test_elon_musk_synthesis_has_snapshot_tier(planner):
    _, plan, answer = _answer(planner, "Tell me about Elon Musk.")
    assert plan.decision == "answer"
    low = answer.lower()
    assert "businessman" in low
    assert "founded spacex, neuralink, the boring company, and xai" in low
    # net worth is a dated, source-qualified estimate -> SNAPSHOT tier
    assert "according to forbes" in low
    assert "should be rechecked" in low
    assert "snapshot —" in low


def test_blue_origin_synthesis(planner):
    _, plan, answer = _answer(planner, "What do you know about Blue Origin?")
    assert plan.decision == "answer"
    low = answer.lower()
    assert "american aerospace company" in low
    assert "founded by jeff bezos" in low


def test_starlink_unknown_tier(planner):
    """'How does Starlink work?' has only a definition — process is UNKNOWN."""
    _, plan, answer = _answer(planner, "How does Starlink work?")
    assert plan.decision == "answer"
    low = answer.lower()
    assert "satellite internet constellation" in low
    assert "unknown —" in low
    assert "i don't have verified information about how it works" in low


# ---------------------------------------------------------------------------
# 3. No fabrication
# ---------------------------------------------------------------------------


def test_unknown_entity_audits(planner):
    _, plan, answer = _answer(planner, "Tell me about Atlantis the lost city.")
    assert plan.decision == "audit"
    assert "don't have verified information" in answer.lower()


def test_synthesize_never_fabricates_for_unknown(provider):
    result = synthesize(provider, "Totally Fictional Nonexistent Corp")
    assert result.matched is False
    assert result.definition is None
    assert not result.groups


def test_no_leadership_in_verified_facts(planner):
    """Leadership is current-sensitive; it must not appear as a VERIFIED fact."""
    _, _, answer = _answer(planner, "Tell me about SpaceX.")
    # Gwynne Shotwell is only a (volatile) leader_of edge, never founded SpaceX.
    assert "shotwell" not in answer.lower()


# ---------------------------------------------------------------------------
# 4. Tier accounting
# ---------------------------------------------------------------------------


def test_verified_count_matches_clauses(provider):
    result = synthesize(provider, "SpaceX", "Tell me about SpaceX.")
    # definition + develops(2) + produces(2) + founded_by(1) = 6
    assert result.verified_count == 6
    assert result.snapshot_count == 0


def test_snapshot_count_for_musk(provider):
    result = synthesize(provider, "Elon Musk", "Tell me about Elon Musk.")
    assert result.snapshot_count == 1


# ---------------------------------------------------------------------------
# 5. Adversarial yes/no questions naming an entity still audit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "Is Musk's net worth permanently US$1.1 trillion?",
        "Will Musk's net worth always be US$1.1 trillion?",
        "Since leadership is linked to Musk, is leadership his product?",
    ],
)
def test_yes_no_traps_do_not_synthesize(planner, question):
    analyzed, plan, _ = _answer(planner, question)
    assert analyzed.intent != "open_synthesis", question
    assert plan.decision == "audit", question


# ---------------------------------------------------------------------------
# 6. Existing "Tell me about X" define-style coverage still answers
# ---------------------------------------------------------------------------


def test_tell_me_about_tesla_still_answers(planner):
    _, plan, answer = _answer(planner, "Tell me about Tesla.")
    assert plan.decision == "answer"
    assert "electric vehicle" in answer.lower()
