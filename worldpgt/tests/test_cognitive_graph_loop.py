"""Tests for the graph-native cognitive loop.

The loop must select cognitive moves from graph state, activation, and
evidence boundaries — not from canned answer templates or a fixed sequence.
"""

from __future__ import annotations

import json

from worldpgt.cognition import (
    action_plan_from_cognitive_loop,
    reason_over_plan,
    render_plan_addendum,
    run_cognitive_graph_loop,
)
from worldpgt.community_context.types import CognitivePatternEvent
from worldpgt.entity_qa.semantic_speech_planner import build_speech_plan
from worldpgt.entity_qa.types import SynthesisAnswer, SynthesisFactGroup


def _starlink_mechanism_gap_trace():
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
    return reason_over_plan(build_speech_plan(result, "How does Starlink work?"))


def _spacex_profile_trace():
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
                objects=["rockets", "spacecraft"],
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
    return reason_over_plan(build_speech_plan(result, "Tell me about SpaceX."))


def _debugging_pattern() -> CognitivePatternEvent:
    return CognitivePatternEvent(
        event_id="debug-1",
        kind="debugging_pattern",
        topic="programming",
        pattern="reduce the problem to a minimal reproducible example",
        steps=(
            "state expected behavior",
            "state actual behavior",
            "show the smallest reproduction",
        ),
        confidence="high",
        trust="behavioral_pattern",
        factual_support_allowed=False,
    )


def test_missing_mechanism_selects_evidence_boundary_moves() -> None:
    loop = run_cognitive_graph_loop(
        _starlink_mechanism_gap_trace(),
        question="How does Starlink work?",
    )

    applied = set(loop.applied_kinds)
    assert {"check_missing_evidence", "separate_fact_from_interpretation", "ask_missing_constraint"} <= applied
    # Gap activation dominates the first selection.
    assert loop.applied_moves[0].kind == "check_missing_evidence"
    assert any(node_id.startswith("gap:") for node_id in loop.applied_moves[0].activated_by)

    plan = loop.answer_plan
    assert plan.facts_to_say, "supported profile facts must remain sayable"
    assert any("mechanism" in claim for claim in plan.facts_not_allowed)
    assert plan.uncertainty_to_state
    assert loop.factual_support_allowed_from_patterns is False


def test_supported_profile_selects_explanation_moves() -> None:
    loop = run_cognitive_graph_loop(
        _spacex_profile_trace(),
        question="Explain SpaceX in simple terms.",
    )

    applied = set(loop.applied_kinds)
    assert {"decompose_question", "ground_in_supported_example", "choose_explanation_depth"} <= applied
    assert "check_missing_evidence" not in applied

    # The gap move must be rejected for a graph-state reason, not silently skipped.
    first_rejections = {move.kind: move.reason for move in loop.iterations[0].rejected}
    assert "check_missing_evidence" in first_rejections
    assert "gap" in first_rejections["check_missing_evidence"]

    plan = loop.answer_plan
    assert plan.verified is True
    assert loop.stop_reason == "answer_plan_ready"
    assert plan.explanation_depth is not None
    assert all(kind == "grounded" for kind, _text in plan.examples)


def test_debugging_pattern_becomes_graph_move_not_text() -> None:
    trace = _spacex_profile_trace()
    question = "Debug this programming error."

    with_pattern = run_cognitive_graph_loop(
        trace,
        question=question,
        cognitive_patterns=[_debugging_pattern()],
    )
    without_pattern = run_cognitive_graph_loop(trace, question=question)

    assert "reduce_to_minimal_repro" in with_pattern.applied_kinds
    repro = next(
        move for move in with_pattern.applied_moves if move.kind == "reduce_to_minimal_repro"
    )
    assert "pattern:debug-1" in repro.activated_by

    assert "reduce_to_minimal_repro" not in without_pattern.applied_kinds
    rejections = {
        move.kind: move.reason for move in without_pattern.iterations[0].rejected
    }
    assert "pattern" in rejections["reduce_to_minimal_repro"]


def test_unsafe_pattern_is_ignored_entirely() -> None:
    unsafe = CognitivePatternEvent(
        event_id="bad-fact",
        kind="explanation_pattern",
        topic="SpaceX",
        pattern="bad pattern that tries to act as factual support",
        factual_support_allowed=True,
    )
    trace = _spacex_profile_trace()
    loop = run_cognitive_graph_loop(
        trace,
        question="Explain SpaceX.",
        cognitive_patterns=[unsafe],
    )

    assert "pattern:bad-fact" not in {node.node_id for node in loop.nodes}
    assert loop.factual_support_allowed_from_patterns is False
    admitted = {item.text for item in trace.workspace.items}
    assert all(text in admitted for _role, text in loop.answer_plan.facts_to_say)


def test_changing_evidence_changes_selected_moves() -> None:
    gap_loop = run_cognitive_graph_loop(
        _starlink_mechanism_gap_trace(),
        question="How does Starlink work?",
    )
    profile_loop = run_cognitive_graph_loop(
        _spacex_profile_trace(),
        question="Explain SpaceX in simple terms.",
    )

    assert gap_loop.applied_moves[0].kind != profile_loop.applied_moves[0].kind
    assert set(gap_loop.applied_kinds) != set(profile_loop.applied_kinds)
    assert gap_loop.missing_roles and not profile_loop.missing_roles


def test_changing_pattern_graph_changes_selected_moves() -> None:
    trace = _spacex_profile_trace()
    question = "Debug this programming error."

    base = run_cognitive_graph_loop(trace, question=question)
    with_pattern = run_cognitive_graph_loop(
        trace,
        question=question,
        cognitive_patterns=[_debugging_pattern()],
    )

    assert set(base.applied_kinds) != set(with_pattern.applied_kinds)


def test_grounded_concepts_activate_comparison_move() -> None:
    trace = _spacex_profile_trace()
    loop = run_cognitive_graph_loop(
        trace,
        question="Is SpaceX closer to rockets or spacecraft?",
    )

    assert "compare_concepts" in loop.applied_kinds
    compare = next(move for move in loop.applied_moves if move.kind == "compare_concepts")
    assert set(compare.activated_by) >= {"concept:rockets", "concept:spacecraft"}


def test_loop_trace_is_iterative_and_inspectable() -> None:
    loop = run_cognitive_graph_loop(
        _starlink_mechanism_gap_trace(),
        question="How does Starlink work?",
    )

    assert len(loop.iterations) >= 3, "the loop must iterate, not rank once"
    for iteration in loop.iterations:
        if iteration.selected is None:
            continue
        assert iteration.candidates
        assert iteration.selected.reason
        assert iteration.selected.effects
    assert all(move.reason for move in loop.iterations[0].rejected)
    assert loop.stop_reason in (
        "answer_plan_ready",
        "blocked_unsupported",
        "no_applicable_moves",
        "max_iterations",
    )

    payload = json.dumps(loop.to_dict())
    assert "answer_plan" in payload


def test_rendering_bridge_forces_clarification_when_unsupported() -> None:
    from worldpgt.cognition.answer_session import _primary_action

    result = SynthesisAnswer(
        subject="Mystery",
        matched=True,
        match_kind="exact",
        definition="",
        entity_type="organization",
        groups=[],
    )
    question = "How does Mystery work?"
    trace = reason_over_plan(build_speech_plan(result, question))
    loop = run_cognitive_graph_loop(trace, question=question)
    fallback = _primary_action(trace)

    bridged = action_plan_from_cognitive_loop(loop, fallback=fallback)

    if loop.stop_reason == "blocked_unsupported":
        assert bridged.next_action == "ask_clarification"
    assert set(bridged.forbidden_claims) >= set(loop.answer_plan.facts_not_allowed)


def test_rendering_bridge_never_loosens_bucket_suppression() -> None:
    from worldpgt.cognition.answer_session import _primary_action

    trace = _starlink_mechanism_gap_trace()
    question = "How does Starlink work?"
    loop = run_cognitive_graph_loop(trace, question=question)
    fallback = _primary_action(trace)

    bridged = action_plan_from_cognitive_loop(loop, fallback=fallback)

    assert set(fallback.suppressed_buckets) <= set(bridged.suppressed_buckets)
    assert "mechanism" in bridged.suppressed_buckets
    # The tuned per-style bucket ordering is never reshuffled by the loop.
    assert bridged.preferred_buckets == fallback.preferred_buckets


def test_repro_checklist_and_mistake_check_render_as_labeled_addendum() -> None:
    trace = _spacex_profile_trace()
    question = "I have a programming bug, compare recursion mistakes to SpaceX engineering."
    patterns = [
        CognitivePatternEvent(
            event_id="debug-1",
            kind="debugging_pattern",
            topic="programming",
            pattern="reduce the problem to a minimal reproducible example",
            steps=("state expected behavior", "state actual behavior", "show the smallest reproduction"),
            confidence="high",
            trust="behavioral_pattern",
            factual_support_allowed=False,
        ),
        CognitivePatternEvent(
            event_id="mistake-1",
            kind="mistake_pattern",
            topic="programming",
            pattern="forgetting the base case causes infinite recursion",
            confidence="high",
            factual_support_allowed=False,
        ),
        CognitivePatternEvent(
            event_id="analogy-1",
            kind="analogy_pattern",
            topic="programming",
            pattern="use an analogy only as a bridge, then return to the precise concept",
            example_shape="Recursion is like nesting dolls.",
            confidence="high",
            factual_support_allowed=False,
        ),
    ]

    loop = run_cognitive_graph_loop(trace, question=question, cognitive_patterns=patterns)
    addendum = render_plan_addendum(loop.answer_plan)

    assert any("reduce_to_minimal_repro" == kind for kind in loop.applied_kinds)
    assert any("detect_likely_mistake" == kind for kind in loop.applied_kinds)
    assert any("state expected behavior" in line for line in addendum)
    assert any("forgetting the base case" in line for line in addendum)
    assert any("As an analogy only" in line and "nesting dolls" in line for line in addendum)
    # No duplicate lines even though the analogy pattern feeds two moves.
    assert len(addendum) == len(set(addendum))
    # Addendum content must never be treated as part of facts_to_say.
    admitted = {text for _role, text in loop.answer_plan.facts_to_say}
    assert not any(line in admitted for line in addendum)


def test_addendum_is_empty_without_patterns() -> None:
    loop = run_cognitive_graph_loop(_spacex_profile_trace(), question="Explain SpaceX in simple terms.")
    assert render_plan_addendum(loop.answer_plan) == ()


def test_stops_when_nothing_is_supported() -> None:
    result = SynthesisAnswer(
        subject="Mystery",
        matched=True,
        match_kind="exact",
        definition="",
        entity_type="organization",
        groups=[],
    )
    trace = reason_over_plan(build_speech_plan(result, "How does Mystery work?"))
    loop = run_cognitive_graph_loop(trace, question="How does Mystery work?")

    if not loop.answer_plan.facts_to_say:
        assert loop.stop_reason == "blocked_unsupported"
        assert "stop_when_unsupported" in loop.applied_kinds
    assert loop.answer_plan.facts_to_say == () or loop.answer_plan.verified
