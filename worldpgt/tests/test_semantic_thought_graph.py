"""Tests for graph-based cognitive move selection."""

from __future__ import annotations

from worldpgt.cognition import reason_over_plan, run_semantic_thought_graph
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


def test_semantic_thought_graph_selects_gap_move_from_missing_evidence() -> None:
    trace = _starlink_mechanism_gap_trace()

    graph = run_semantic_thought_graph(trace, question="How does Starlink work?")

    assert graph.selected_moves
    assert graph.selected_moves[0].kind == "check_missing_evidence"
    assert "gap:mechanism" in {node.node_id for node in graph.nodes}
    assert graph.factual_support_allowed_from_patterns is False
    assert all(move.factual_support_allowed is False for move in graph.selected_moves)


def test_semantic_thought_graph_changes_moves_when_evidence_changes() -> None:
    gap_graph = run_semantic_thought_graph(
        _starlink_mechanism_gap_trace(),
        question="How does Starlink work?",
    )
    profile_graph = run_semantic_thought_graph(
        _spacex_profile_trace(),
        question="Explain SpaceX in simple terms.",
    )

    assert gap_graph.selected_moves[0].kind == "check_missing_evidence"
    assert profile_graph.selected_moves[0].kind != "check_missing_evidence"
    assert {move.kind for move in profile_graph.selected_moves} >= {
        "ground_in_supported_example",
    }


def test_semantic_thought_graph_uses_patterns_as_moves_not_facts() -> None:
    trace = _spacex_profile_trace()
    patterns = [
        CognitivePatternEvent(
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
        ),
        CognitivePatternEvent(
            event_id="bad-fact",
            kind="explanation_pattern",
            topic="SpaceX",
            pattern="bad pattern that tries to act as factual support",
            factual_support_allowed=True,
        ),
    ]

    graph = run_semantic_thought_graph(
        trace,
        question="Debug this programming error.",
        cognitive_patterns=patterns,
    )

    assert graph.selected_moves[0].kind == "reduce_to_minimal_repro"
    assert "pattern:debug-1" in {node.node_id for node in graph.nodes}
    assert "pattern:bad-fact" not in {node.node_id for node in graph.nodes}
    assert graph.factual_support_allowed_from_patterns is False
