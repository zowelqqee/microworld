"""Tests for the synthesis layer (Entity QA layer 3).

The synthesis layer answers open questions ("Tell me about X") by gathering
every relevant supported fact about an entity from the overlay, grouped by type and
confidence tier (VERIFIED / SNAPSHOT / UNKNOWN). It never invents facts.

All tests are deterministic and offline. No network access. No modification of
sense_memory.py, accepted_knowledge_memory_v1.json, the overlay semantics, the
planner thresholds, or the validators.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.cognition.mini_reasoner import build_mini_thought
from worldpgt.cognition.reasoning_engine import reason_over_plan
from worldpgt.cognition.self_correction import self_correct_answer
from worldpgt.cognition.support_guard import validate_conclusion_support
from worldpgt.cognition.surface_selection import select_answer_variant
from worldpgt.cognition.phrase_graph import (
    build_phrase_graph,
    generate as generate_phrase_graph,
)
from worldpgt.cognition.types import (
    ActionPlan,
    AllowedConclusion,
    AnswerVariant,
    EvidenceItem,
    EvidenceWorkspace,
)
from worldpgt.entity_qa.entity_answer_planner import EntityAnswerPlanner
from worldpgt.entity_qa.entity_answer_renderer import render
from worldpgt.entity_qa.entity_question_analyzer import analyze
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.entity_qa.semantic_speech_planner import build_speech_plan, render_speech_plan
from worldpgt.entity_qa.symbolic_text_generator import (
    generate_text_with_selection_trace,
    next_speech_unit_candidates,
)
from worldpgt.entity_qa.synthesis_engine import synthesize
from worldpgt.entity_qa.types import (
    EntityQAEvidence,
    EntityQAPlan,
    SynthesisAnswer,
    SynthesisFactGroup,
)
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_OVERLAY_JSON = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"


@pytest.fixture(scope="module")
def provider():
    return WikiMemoryOverlayProvider(_OVERLAY_JSON)


@pytest.fixture(scope="module")
def planner(provider):
    return EntityAnswerPlanner(provider=provider)


class _BulkOnlyProvider:
    """Provider adapter exposing only the public bulk overlay accessors."""

    def __init__(self, provider):
        self._provider = provider

    def all_entities(self):
        return self._provider.all_entities()

    def all_definitions(self):
        return self._provider.all_definitions()

    def all_relations(self):
        return self._provider.all_relations()

    def all_source_facts(self):
        return self._provider.all_source_facts()


class _StaticProvider:
    def __init__(
        self,
        *,
        entities: list[dict],
        definitions: list[dict] | None = None,
        relations: list[dict] | None = None,
        source_facts: list[dict] | None = None,
    ):
        self._entities = entities
        self._definitions = definitions or []
        self._relations = relations or []
        self._source_facts = source_facts or []

    def all_entities(self):
        return self._entities

    def all_definitions(self):
        return self._definitions

    def all_relations(self):
        return self._relations

    def all_source_facts(self):
        return self._source_facts


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
    assert "[verified" not in low


def test_elon_musk_synthesis_has_snapshot_tier(planner):
    _, plan, answer = _answer(planner, "Tell me about Elon Musk.")
    assert plan.decision == "answer"
    low = answer.lower()
    assert "businessman" in low
    assert "founded spacex, neuralink, the boring company, and xai" in low
    # net worth is a dated, source-qualified estimate -> SNAPSHOT tier
    # as_of 2026-06 is within the 30-day freshness window for estimated_net_worth
    # so the renderer uses present tense without the "may be outdated" caveat
    assert "estimated net worth is us$1.1 trillion (forbes)" in low
    assert "snapshot —" not in low


def test_open_synthesis_structured_rendering_orders_tiers():
    result = SynthesisAnswer(
        subject="ExampleCo",
        matched=True,
        match_kind="exact",
        definition="robotics company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="develops",
                objects=["robots", "drones", "sensors", "arms", "software"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="founded_by",
                objects=["Ada Stone"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="known_for",
                objects=["Robotics"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="snapshot",
                predicate="estimated_revenue",
                objects=["US$10 billion"],
                tier="SNAPSHOT",
                source_name="ExampleSource",
                as_of="2026-06",
            ),
            SynthesisFactGroup(
                kind="inferred_relation",
                predicate="competes_with",
                objects=["PeerCo"],
                tier="INFERRED",
                rule="competitor_detection_v1",
                confidence=0.7,
            ),
        ],
    )
    analyzed = analyze("Tell me about ExampleCo.")
    plan = EntityQAPlan(
        analyzed=analyzed,
        decision="answer",
        audit_reason=None,
        evidence=EntityQAEvidence(),
        render_template="open_synthesis",
        render_args={"synthesis": result, "question": analyzed.question},
        confidence=0.9,
    )

    answer = render(plan)

    # The learned relative-clause connector folds the first fact into the
    # definition: "X is a Y that develops ..." instead of a separate "It" clause.
    assert answer.startswith(
        "ExampleCo is a robotics company that develops robots, drones, sensors, arms, and 1 more."
    )
    assert "It was founded by Ada Stone." in answer
    assert answer.index("that develops") < answer.index("It was founded by Ada Stone.")
    assert answer.index("It was founded by Ada Stone.") < answer.index("It is known for Robotics.")
    assert "As of June 2026, its estimated revenue is US$10 billion (ExampleSource)" in answer
    # Inferred (derived, non-stated) facts are deliberately left out of the
    # answer — it states the directly supported profile, not a reasoning dump.
    assert "Based on reasoning" not in answer
    assert "PeerCo" not in answer


def test_open_synthesis_person_without_gender_uses_plural_pronoun():
    result = SynthesisAnswer(
        subject="Ada Stone",
        matched=True,
        match_kind="exact",
        definition="engineer",
        entity_type="person",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="known_for",
                objects=["Robotics"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="founded",
                objects=["ExampleCo"],
                tier="VERIFIED",
            ),
        ],
    )
    analyzed = analyze("Tell me about Ada Stone.")
    plan = EntityQAPlan(
        analyzed=analyzed,
        decision="answer",
        audit_reason=None,
        evidence=EntityQAEvidence(),
        render_template="open_synthesis",
        render_args={"synthesis": result, "question": analyzed.question},
        confidence=0.9,
    )

    answer = render(plan)

    assert "They are known for Robotics." in answer
    # First fact woven into the definition via the learned person subordinator.
    assert "Ada Stone is an engineer who founded ExampleCo." in answer


def test_phrase_graph_learns_fragments_and_transitions_from_overlay(tmp_path):
    overlay = tmp_path / "overlay.json"
    overlay.write_text(
        json.dumps(
            [
                {
                    "overlay_type": "overlay_entity",
                    "label": "ExampleCo",
                    "entity_type": "organization",
                    "source_page": "ExampleCo",
                },
                {
                    "overlay_type": "overlay_definition",
                    "subject": "ExampleCo",
                    "definition": "robotics company",
                    "source_page": "ExampleCo",
                    "evidence_text": "ExampleCo is a robotics company.",
                },
                {
                    "overlay_type": "overlay_relation",
                    "subject": "ExampleCo",
                    "predicate": "develops",
                    "object": "robots",
                    "source_page": "ExampleCo",
                    "evidence_text": "ExampleCo develops robots.",
                },
                {
                    "overlay_type": "overlay_relation",
                    "subject": "ExampleCo",
                    "predicate": "founded_by",
                    "object": "Ada Stone",
                    "source_page": "ExampleCo",
                    "evidence_text": "ExampleCo was founded by Ada Stone.",
                },
            ]
        ),
        encoding="utf-8",
    )

    graph = build_phrase_graph(
        overlay_paths=[overlay],
        artifact_paths=[],
        snapshot_dir=tmp_path / "missing",
        community_context_paths=[],
    )

    assert graph.best_fragment(("organization", "is_a")) == "is {definition}"
    assert graph.best_fragment(("organization", "develops")) == "develops {object_list}"
    assert graph.best_fragment(("organization", "founded_by")) == "was founded by {object_list}"
    assert graph.best_connector(
        ("organization", "is_a"),
        ("organization", "develops"),
    ) == ""


def test_phrase_graph_generates_deterministic_best_path_from_synthesis():
    graph = build_phrase_graph(
        overlay_paths=[],
        artifact_paths=[],
        snapshot_dir=Path("/missing"),
        community_context_paths=[],
    )
    graph.add_fragment("person", "is_a", "is {definition}")
    graph.add_fragment("person", "founded", "founded {object_list}")
    graph.add_fragment("person", "known_for", "is known for {object_list}")
    graph.add_transition(("person", "is_a"), ("person", "founded"), "")
    graph.add_transition(("person", "founded"), ("person", "known_for"), "")
    result = SynthesisAnswer(
        subject="Ada Stone",
        matched=True,
        match_kind="exact",
        definition="investor",
        entity_type="other",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="known_for",
                objects=["Robotics"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="founded",
                objects=["ExampleCo"],
                tier="VERIFIED",
            ),
        ],
    )

    answer = generate_phrase_graph(result, graph=graph)

    assert answer == (
        "Ada Stone is an investor. "
        "They founded ExampleCo. "
        "They are known for Robotics."
    )
    assert "It is known for Robotics." not in answer


def test_blue_origin_synthesis(planner):
    _, plan, answer = _answer(planner, "What do you know about Blue Origin?")
    assert plan.decision == "answer"
    low = answer.lower()
    assert "american aerospace company" in low
    assert "founded by jeff bezos" in low


def test_open_synthesis_groups_verified_facts_in_speech_plan(planner):
    _, plan, answer = _answer(planner, "What do you know about Blue Origin?")

    assert plan.decision == "answer"
    assert "that develops rockets and spacecraft" in answer
    assert "What I can verify" not in answer
    assert "Key supported details" not in answer
    assert answer.count("Blue Origin") < 5


def test_open_synthesis_speech_plan_is_deterministic(planner):
    _, _, answer1 = _answer(planner, "Tell me about Elon Musk.")
    _, _, answer2 = _answer(planner, "Tell me about Elon Musk.")

    assert answer1 == answer2


def test_open_synthesis_followup_style_keeps_answer_compact(planner):
    analyzed = analyze("Tell me about SpaceX.")
    plan = planner.plan(analyzed)
    plan.render_args["answer_style"] = "followup"

    answer = render(plan)

    assert answer.count(".") <= 2
    assert "SpaceX is an aerospace manufacturer" in answer


def test_open_synthesis_does_not_expose_internal_graph_word(planner):
    _, _, answer = _answer(planner, "How does Starlink work?")

    assert "graph" not in answer.lower()


def test_open_synthesis_prioritizes_explanatory_relations_for_how_questions():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="owned_by",
                objects=["SpaceX"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="uses",
                objects=["low Earth orbit satellites"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    answer = render_speech_plan(plan)

    assert "uses low Earth orbit satellites" in answer
    assert "provides satellite internet access" in answer
    assert answer.index("uses low Earth orbit satellites") < answer.index("is owned by SpaceX")


def test_symbolic_text_generation_exposes_next_allowed_speech_units():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="owned_by",
                objects=["SpaceX"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="uses",
                objects=["low Earth orbit satellites"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    candidates = next_speech_unit_candidates(plan)

    assert candidates[0].kind == "mechanism"
    assert "uses low Earth orbit satellites" in candidates[0].text
    assert all("graph" not in c.text.lower() for c in candidates)


def test_self_correction_rewrites_unsupported_mechanism_wording():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    trace = self_correct_answer(
        "Starlink works by providing satellite internet access.",
        plan,
    )

    assert trace.draft.text == "Starlink works by providing satellite internet access."
    assert [finding.code for finding in trace.findings] == [
        "unsupported_mechanism_wording",
    ]
    assert [repair.kind for repair in trace.repairs] == [
        "replace_unsupported_mechanism",
    ]
    assert trace.final_text == (
        "Starlink provides satellite internet access, but I do not have the "
        "mechanism evidence yet."
    )


def test_self_correction_leaves_supported_mechanism_wording_unchanged():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="works_by",
                objects=["using low Earth orbit satellites"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    trace = self_correct_answer(
        "Starlink works by using low Earth orbit satellites.",
        plan,
    )

    assert trace.findings == ()
    assert trace.repairs == ()
    assert trace.final_text == "Starlink works by using low Earth orbit satellites."


def test_surface_selection_prefers_clean_variant_over_repaired_variant():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    trace = select_answer_variant(
        (
            AnswerVariant(
                name="risky",
                text="Starlink works by providing satellite internet access.",
            ),
            AnswerVariant(
                name="clean",
                text=(
                    "Starlink provides satellite internet access, but I do not "
                    "have the mechanism evidence yet."
                ),
            ),
        ),
        plan,
    )

    assert trace.selected_name == "clean"
    assert trace.evaluation_for("risky").score > trace.evaluation_for("clean").score
    assert trace.final_text == (
        "Starlink provides satellite internet access, but I do not have the "
        "mechanism evidence yet."
    )


def test_symbolic_text_generation_exposes_surface_selection_trace():
    result = SynthesisAnswer(
        subject="SpaceX",
        matched=True,
        match_kind="exact",
        definition="aerospace manufacturer and space transportation company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="develops",
                objects=["rockets", "spacecraft", "launch vehicles"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="produces",
                objects=["Falcon rockets", "Dragon spacecraft"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="founded_by",
                objects=["Elon Musk"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "Tell me about SpaceX.")
    trace = generate_text_with_selection_trace(plan)

    assert trace.selected_name in {variant.name for variant in trace.variants}
    assert "baseline" in [variant.name for variant in trace.variants]
    assert len(trace.variants) >= 2
    assert trace.evaluation_for("baseline") is not None
    assert trace.final_text.startswith("SpaceX is an aerospace manufacturer")


def test_symbolic_text_generation_can_execute_clarification_action():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[],
    )
    plan = build_speech_plan(result, "How does Starlink work?")
    action = ActionPlan(
        next_action="ask_clarification",
        surface_goal="ask for the missing mechanism evidence",
        forbidden_claims=("unsupported answer",),
        next_questions=("What mechanism facts are known about Starlink?",),
    )

    trace = generate_text_with_selection_trace(plan, action=action)

    assert trace.selected_name == "clarification"
    assert trace.final_text == (
        "To answer that properly, I need one more piece: "
        "What mechanism facts are known about Starlink?"
    )
    assert trace.variants[0].text == trace.final_text


def test_symbolic_text_generation_humanizes_explicit_gap_action():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
        ],
    )
    plan = build_speech_plan(result, "How does Starlink work?")
    trace = reason_over_plan(plan)
    action = trace.thought_loop.decision.action

    selection = generate_text_with_selection_trace(plan, action=action)

    assert len(selection.variants) >= 4
    assert selection.selected_name in {variant.name for variant in selection.variants}
    assert (
        "Here is the honest version"
        in selection.final_text
        or "the parts and steps that make it work" in selection.final_text
    )
    assert "current evidence" not in selection.final_text
    assert "A useful next question" not in selection.final_text
    assert (
        "The missing piece is" in selection.final_text
        or "one more supported fact" in selection.final_text
    )


def test_symbolic_text_generation_humanizes_explicit_thin_profile_action():
    result = SynthesisAnswer(
        subject="Ray Kroc",
        matched=True,
        match_kind="exact",
        definition="American businessman",
        entity_type="person",
        groups=[],
    )
    plan = build_speech_plan(result, "Tell me about Ray Kroc.")
    trace = reason_over_plan(plan)
    action = trace.thought_loop.decision.action

    selection = generate_text_with_selection_trace(plan, action=action)

    assert len(selection.variants) >= 3
    assert selection.selected_name in {variant.name for variant in selection.variants}
    assert (
        "That is the reliable part I have for Ray Kroc right now."
        in selection.final_text
        or "Right now I can identify Ray Kroc"
        in selection.final_text
    )
    assert "I should not" not in selection.final_text
    assert "unsupported memory" not in selection.final_text


def test_surface_selection_can_prefer_less_internal_clean_variant():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[],
    )
    plan = build_speech_plan(result, "How does Starlink work?")

    selection = select_answer_variant(
        (
            AnswerVariant(
                name="baseline",
                text=(
                    "Starlink is a satellite internet constellation. The facts I "
                    "have identify it, but they do not explain the mechanism."
                ),
            ),
            AnswerVariant(
                name="human",
                text=(
                    "Starlink is a satellite internet constellation. I can identify "
                    "Starlink, but I do not yet have enough to explain how it works."
                ),
            ),
        ),
        plan,
    )

    assert selection.selected_name == "human"
    assert selection.evaluation_for("baseline").score > selection.evaluation_for("human").score


def test_symbolic_text_generation_can_leave_secondary_links_unsaid():
    result = SynthesisAnswer(
        subject="SpaceX",
        matched=True,
        match_kind="exact",
        definition="aerospace manufacturer and space transportation company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="develops",
                objects=["rockets"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="founded_by",
                objects=["Elon Musk"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="inverse_relation",
                predicate="owned_by",
                objects=["Falcon 9"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="located_in",
                objects=["Hawthorne, California"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "Tell me about SpaceX.")
    answer = render_speech_plan(plan)

    assert "develops rockets" in answer
    assert "founded by Elon Musk" in answer
    assert "owns Falcon 9" in answer
    assert "via located_in" not in answer


def test_mini_reasoning_marks_thin_profiles_without_adding_facts():
    result = SynthesisAnswer(
        subject="Ray Kroc",
        matched=True,
        match_kind="exact",
        definition="American businessman",
        entity_type="person",
        groups=[],
    )

    plan = build_speech_plan(result, "Tell me about Ray Kroc.")
    thought = build_mini_thought(plan)
    answer = render_speech_plan(plan)

    assert thought.confidence == "thin"
    assert thought.action.next_action == "answer_with_gap"
    assert (
        "That is the reliable part I have for Ray Kroc right now." in answer
        or "Right now I can identify Ray Kroc" in answer
    )
    assert "thin profile" not in answer
    assert "fuller biography" not in answer
    assert "McDonald's" not in answer


def test_mini_reasoning_names_mechanism_gap_for_how_questions():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="owned_by",
                objects=["SpaceX"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    thought = build_mini_thought(plan)
    answer = render_speech_plan(plan)

    assert thought.task.intent == "mechanism_explanation"
    assert thought.confidence == "gap_heavy"
    assert (
        "Here is the honest version" in answer
        or "the parts and steps that make it work" in answer
    )
    assert "mechanism" in answer
    assert "works by" not in answer


def test_mini_reasoning_adds_cautious_profile_summary_for_rich_profiles():
    result = SynthesisAnswer(
        subject="SpaceX",
        matched=True,
        match_kind="exact",
        definition="aerospace manufacturer and space transportation company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="develops",
                objects=["rockets", "spacecraft", "launch vehicles", "Falcon 9"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="produces",
                objects=["Falcon rockets", "Dragon spacecraft"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="inverse_relation",
                predicate="owned_by",
                objects=["Starlink"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "Tell me about SpaceX.")
    thought = build_mini_thought(plan)
    answer = render_speech_plan(plan)

    assert thought.confidence == "grounded"
    assert (
        "Here are the supported pieces I can put together for SpaceX" in answer
        or "For SpaceX, I have a usable basic profile" in answer
    )
    assert "Starlink" in answer
    assert "unsupported" not in answer


def test_mini_reasoning_action_plan_controls_detail_selection():
    result = SynthesisAnswer(
        subject="SpaceX",
        matched=True,
        match_kind="exact",
        definition="aerospace manufacturer and space transportation company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="develops",
                objects=["rockets", "spacecraft", "launch vehicles", "Falcon 9"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="founded_by",
                objects=["Elon Musk"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="inverse_relation",
                predicate="owned_by",
                objects=["Starlink"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="known_for",
                objects=["rockets"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "Tell me about SpaceX.")
    thought = build_mini_thought(plan)
    answer = render_speech_plan(plan)

    assert thought.action.detail_unit_limit == 2
    assert thought.action.preferred_buckets[:2] == ("activity", "origin")
    assert (
        "Here are the supported pieces I can put together for SpaceX" in answer
        or "For SpaceX, I have a usable basic profile" in answer
    )
    assert "develops rockets" in answer
    assert "founded by Elon Musk" in answer
    assert "owns Starlink" not in answer


def test_reasoning_engine_builds_inspectable_trace_for_profile_summary():
    result = SynthesisAnswer(
        subject="SpaceX",
        matched=True,
        match_kind="exact",
        definition="aerospace manufacturer and space transportation company",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="develops",
                objects=["rockets", "spacecraft", "launch vehicles", "Falcon 9"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="inverse_relation",
                predicate="owned_by",
                objects=["Starlink"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "Tell me about SpaceX.")
    trace = reason_over_plan(plan)

    assert trace.task.intent == "entity_profile"
    assert trace.workspace.subject == "SpaceX"
    assert trace.workspace.texts_for("activity")
    assert trace.workspace.texts_for("ownership")
    assert trace.program.operations == (
        "observe",
        "decompose",
        "cluster",
        "choose_action",
        "support_guard",
    )
    assert [step.kind for step in trace.steps] == (
        ["observe", "decompose", "cluster", "choose_action", "support_guard"]
    )
    assert trace.primary_conclusion is not None
    assert trace.primary_conclusion.kind == "profile_summary"
    assert trace.action.surface_goal == "summarize the profile before listing details"
    assert trace.action_candidates
    assert trace.action_candidates[0].action.next_action == "answer"
    assert trace.action_candidates[0].priority == 70
    assert trace.support_checks
    assert trace.support_checks[0].ok
    subgoal_status = {goal.name: goal.status for goal in trace.subgoals}
    assert subgoal_status["identify subject"] == "satisfied"
    assert subgoal_status["summarize salient profile"] == "satisfied"
    assert subgoal_status["guard unsupported expansion"] == "satisfied"
    assert trace.working_memory.active_subgoals == ()
    assert trace.working_memory.missing_roles == ()


def test_reasoning_engine_does_not_convert_profile_evidence_into_mechanism():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="owned_by",
                objects=["SpaceX"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    trace = reason_over_plan(plan)

    assert trace.task.intent == "mechanism_explanation"
    assert trace.program.operations == (
        "observe",
        "decompose",
        "gap_check",
        "choose_action",
        "support_guard",
    )
    assert [step.kind for step in trace.steps] == (
        ["observe", "decompose", "gap_check", "choose_action", "support_guard"]
    )
    assert trace.primary_conclusion is not None
    assert trace.primary_conclusion.kind == "mechanism_gap"
    assert trace.confidence == "gap_heavy"
    assert trace.action_candidates
    assert trace.action_candidates[0].action.next_action == "answer_with_gap"
    assert trace.action_candidates[0].priority == 90
    assert trace.support_checks
    assert trace.support_checks[0].ok
    assert "mechanism" in trace.action.forbidden_claims[0]
    assert not trace.workspace.texts_for("mechanism")
    assert trace.missing_evidence
    assert trace.missing_evidence[0].role == "mechanism"
    subgoal_status = {goal.name: goal.status for goal in trace.subgoals}
    assert subgoal_status["identify subject"] == "satisfied"
    assert subgoal_status["explain mechanism"] == "missing"
    assert subgoal_status["fallback profile"] == "satisfied"
    assert trace.working_memory.active_subgoals == ("explain mechanism",)
    assert trace.working_memory.missing_roles == ("mechanism",)


def test_reasoning_engine_suggests_next_question_for_missing_mechanism():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    trace = reason_over_plan(plan)
    answer = render_speech_plan(plan)

    assert trace.action.next_questions
    assert trace.missing_evidence[0].next_questions == trace.action.next_questions
    assert trace.action.next_questions[0] == (
        "What does Starlink use to provide that service?"
    )
    assert "The missing piece is" in answer
    assert trace.action.next_questions[0] in answer


def test_reasoning_program_uses_scope_guard_for_thin_profiles():
    result = SynthesisAnswer(
        subject="Ray Kroc",
        matched=True,
        match_kind="exact",
        definition="American businessman",
        entity_type="person",
        groups=[],
    )

    plan = build_speech_plan(result, "Tell me about Ray Kroc.")
    trace = reason_over_plan(plan)

    assert trace.program.operations == (
        "observe",
        "decompose",
        "scope_guard",
        "choose_action",
        "support_guard",
    )
    assert [step.kind for step in trace.steps] == (
        ["observe", "decompose", "scope_guard", "choose_action", "support_guard"]
    )
    assert trace.primary_conclusion is not None
    assert trace.primary_conclusion.kind == "thin_profile"
    assert trace.action_candidates
    assert trace.action_candidates[0].action.next_action == "answer_with_gap"
    assert trace.action_candidates[0].priority == 85
    assert trace.working_memory.active_subgoals == ("summarize salient profile",)
    assert trace.working_memory.missing_roles[:2] == ("activity", "origin")


def test_reasoning_program_operations_are_executed_in_order():
    cases = [
        (
            "How does Starlink work?",
            SynthesisAnswer(
                subject="Starlink",
                matched=True,
                match_kind="exact",
                definition="satellite internet constellation operated by SpaceX",
                entity_type="organization",
                groups=[
                    SynthesisFactGroup(
                        kind="forward_relation",
                        predicate="provides",
                        objects=["satellite internet access"],
                        tier="VERIFIED",
                    )
                ],
            ),
        ),
        (
            "Tell me about SpaceX.",
            SynthesisAnswer(
                subject="SpaceX",
                matched=True,
                match_kind="exact",
                definition="aerospace manufacturer and space transportation company",
                entity_type="organization",
                groups=[
                    SynthesisFactGroup(
                        kind="forward_relation",
                        predicate="develops",
                        objects=["rockets", "spacecraft", "launch vehicles", "Falcon 9"],
                        tier="VERIFIED",
                    ),
                    SynthesisFactGroup(
                        kind="inverse_relation",
                        predicate="owned_by",
                        objects=["Starlink"],
                        tier="VERIFIED",
                    ),
                ],
            ),
        ),
        (
            "Tell me about Ray Kroc.",
            SynthesisAnswer(
                subject="Ray Kroc",
                matched=True,
                match_kind="exact",
                definition="American businessman",
                entity_type="person",
                groups=[],
            ),
        ),
    ]

    for question, result in cases:
        plan = build_speech_plan(result, question)
        trace = reason_over_plan(plan)

        assert tuple(step.kind for step in trace.steps) == trace.program.operations
        assert trace.support_checks
        assert all(check.ok for check in trace.support_checks)


def test_reasoning_action_selection_records_candidate_reasons():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[
            SynthesisFactGroup(
                kind="forward_relation",
                predicate="provides",
                objects=["satellite internet access"],
                tier="VERIFIED",
            ),
        ],
    )

    plan = build_speech_plan(result, "How does Starlink work?")
    trace = reason_over_plan(plan)
    choose_step = [step for step in trace.steps if step.kind == "choose_action"][0]

    assert trace.action.next_action == "answer_with_gap"
    assert choose_step.input == (
        "mechanism subgoal is missing but fallback profile evidence exists",
    )
    assert choose_step.output == "describe supported profile and name the mechanism gap"


def test_support_guard_blocks_unsupported_profile_summary_terms():
    workspace = EvidenceWorkspace(
        subject="SpaceX",
        items=(
            EvidenceItem(role="activity", text="develops rockets and spacecraft"),
            EvidenceItem(role="ownership", text="owns Starlink"),
        ),
    )

    ok = validate_conclusion_support(
        AllowedConclusion(
            kind="profile_summary",
            text="The facts I have mostly frame it through rockets, spacecraft, and Starlink.",
            evidence_roles=("activity", "ownership"),
        ),
        workspace,
    )
    bad = validate_conclusion_support(
        AllowedConclusion(
            kind="profile_summary",
            text="The facts I have mostly frame it through rockets and Mars bases.",
            evidence_roles=("activity",),
        ),
        workspace,
    )

    assert ok.ok
    assert not bad.ok
    assert bad.issues == ("unsupported profile term: Mars bases",)


def test_starlink_unknown_tier(planner):
    """'How does Starlink work?' has only a definition — process is UNKNOWN."""
    _, plan, answer = _answer(planner, "How does Starlink work?")
    assert plan.decision == "answer"
    low = answer.lower()
    assert "satellite internet constellation" in low
    assert "unknown —" not in low
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


def test_synthesize_uses_bulk_provider_api(provider):
    result = synthesize(
        _BulkOnlyProvider(provider),
        "Elon Musk",
        "Tell me about Elon Musk.",
    )

    assert result.matched is True
    assert result.definition
    assert result.snapshot_count == 1


def test_synthesize_filters_noisy_forward_relations_for_people():
    provider = _StaticProvider(
        entities=[
            {
                "label": "Elon Musk",
                "entity_type": "person",
                "aliases": [],
                "source_page": "Elon Musk",
            }
        ],
        definitions=[
            {
                "subject": "Elon Musk",
                "definition": "businessman and entrepreneur",
            }
        ],
        relations=[
            {
                "overlay_type": "overlay_relation",
                "subject": "Elon Musk",
                "predicate": "founded",
                "object": "SpaceX",
                "risk": "low",
                "stability": "stable",
            },
            {
                "overlay_type": "overlay_relation",
                "subject": "Elon Musk",
                "predicate": "provides",
                "object": "Starlink",
                "risk": "low",
                "stability": "stable",
            },
            {
                "overlay_type": "overlay_relation",
                "subject": "Elon Musk",
                "predicate": "enables",
                "object": "Starlink",
                "risk": "low",
                "stability": "stable",
            },
            {
                "overlay_type": "overlay_relation",
                "subject": "Elon Musk",
                "predicate": "uses",
                "object": "Tesla",
                "risk": "low",
                "stability": "stable",
            },
            {
                "overlay_type": "overlay_relation",
                "subject": "Elon Musk",
                "predicate": "merged_with",
                "object": "March",
                "risk": "low",
                "stability": "stable",
            },
        ],
    )

    result = synthesize(provider, "Elon Musk", "Tell me about Elon Musk.")
    forward = [
        (group.predicate, tuple(group.objects))
        for group in result.groups
        if group.kind == "forward_relation"
    ]

    assert forward == [("founded", ("SpaceX",))]


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
