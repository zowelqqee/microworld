"""Tests for transient cognitive session planning."""

from __future__ import annotations

from worldpgt.cognition import (
    CognitiveAnswerSession,
    CognitiveSession,
    build_task_plan,
    reason_over_plan,
    run_thought_loop,
)
from worldpgt.assistant_surface.types import AssistantAnswer
from worldpgt.entity_qa.semantic_speech_planner import build_speech_plan
from worldpgt.entity_qa.types import (
    EntityQAEvidence,
    EntityQAPlan,
    SynthesisAnswer,
    SynthesisFactGroup,
)


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
                kind="forward_relation",
                predicate="founded_by",
                objects=["Elon Musk"],
                tier="VERIFIED",
            ),
        ],
    )
    return reason_over_plan(build_speech_plan(result, "Tell me about SpaceX."))


class _DefinitionOnlyPlanner:
    def plan(self, analyzed):
        synthesis = SynthesisAnswer(
            subject="Starlink",
            matched=True,
            match_kind="exact",
            definition="satellite internet constellation operated by SpaceX",
            entity_type="organization",
            groups=[],
        )
        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=EntityQAEvidence(),
            render_template="open_synthesis",
            render_args={"synthesis": synthesis},
            confidence=1.0,
        )


class _DefinitionOnlyOrchestrator:
    def __init__(self):
        self._entity_planner = _DefinitionOnlyPlanner()

    def answer(self, question: str, answer_style: str = "normal"):
        return AssistantAnswer(
            question=question,
            decision="answer",
            route="entity_definition",
            answer_text="Old non-cognitive surface text.",
            overlay_mode="pump-dry-run",
            supported_by_context=True,
            support_kind="stable_definition",
            source_system="fake_orchestrator",
        )


def test_cognitive_session_explains_missing_evidence_from_last_trace():
    session = CognitiveSession()
    session.record_trace("How does Starlink work?", _starlink_mechanism_gap_trace())

    plan = session.continue_turn("что нужно знать?")

    assert plan.intent == "explain_missing_evidence"
    assert plan.subject == "Starlink"
    assert plan.active_subgoals == ("explain mechanism",)
    assert [missing.role for missing in plan.missing_evidence] == ["mechanism"]
    assert "missing evidence is mechanism" in plan.answer_text
    assert "What does Starlink use to provide that service?" in plan.answer_text


def test_cognitive_session_suggests_next_question_from_trace():
    session = CognitiveSession()
    session.record_trace("How does Starlink work?", _starlink_mechanism_gap_trace())

    plan = session.continue_turn("что дальше?")

    assert plan.intent == "suggest_next_question"
    assert plan.subject == "Starlink"
    assert plan.next_questions[0] == "What does Starlink use to provide that service?"
    assert "What process connects Starlink to its users?" in plan.next_questions
    assert plan.answer_text.endswith("What does Starlink use to provide that service?")


def test_task_planner_turns_mechanism_gap_into_blocked_plan():
    trace = _starlink_mechanism_gap_trace()

    task_plan = build_task_plan(trace)

    assert task_plan.subject == "Starlink"
    assert task_plan.goal == "explain how Starlink works"
    assert task_plan.blocked is True
    assert [step.title for step in task_plan.steps if step.status == "satisfied"] == [
        "identify subject",
        "fallback profile",
    ]
    blocked = [step for step in task_plan.steps if step.status == "blocked"]
    assert len(blocked) == 1
    assert blocked[0].title == "explain mechanism"
    assert blocked[0].evidence_role == "mechanism"
    assert blocked[0].next_question == "What does Starlink use to provide that service?"
    assert "Blocked: explain mechanism" in task_plan.answer_text


def test_deliberation_rejects_unsupported_direct_mechanism_answer():
    trace = _starlink_mechanism_gap_trace()

    assert trace.deliberation is not None
    rejected = trace.deliberation.evaluation_for("direct_mechanism_explanation")

    assert rejected is not None
    assert rejected.decision == "rejected"
    assert rejected.missing_roles == ("mechanism",)
    assert "missing required evidence: mechanism" in rejected.reason
    assert trace.deliberation.selected_name == "action_candidate_1_answer-with-gap"


def test_deliberation_accepts_supported_profile_summary_candidate():
    trace = _spacex_profile_trace()

    assert trace.deliberation is not None
    assert trace.deliberation.selected_name == "action_candidate_1_answer"
    selected = trace.deliberation.evaluation_for(trace.deliberation.selected_name)

    assert selected is not None
    assert selected.decision == "accepted"
    assert selected.missing_roles == ()


def test_thought_loop_rejects_direct_mechanism_then_accepts_gap_fallback():
    trace = _starlink_mechanism_gap_trace()

    loop = run_thought_loop(trace)

    assert loop.subject == "Starlink"
    assert loop.goal == "explain how Starlink works"
    assert [step.operation for step in loop.steps] == [
        "inspect_evidence",
        "form_hypothesis",
        "test_hypothesis",
        "test_hypothesis",
    ]
    direct = [h for h in loop.hypotheses if h.name == "direct_mechanism"][0]
    fallback = [h for h in loop.hypotheses if h.name == "supported_profile_with_gap"][0]
    assert direct.status == "rejected"
    assert direct.reason == "missing evidence: mechanism"
    assert fallback.status == "accepted"
    assert loop.accepted_hypothesis == fallback
    assert loop.stop_reason == "supported_conclusion"


def test_reasoning_trace_attaches_thought_loop_gap_decision():
    trace = _starlink_mechanism_gap_trace()

    assert trace.thought_loop is not None
    assert trace.thought_loop.accepted_hypothesis is not None
    assert trace.thought_loop.accepted_hypothesis.name == "supported_profile_with_gap"
    assert trace.thought_loop.decision is not None
    assert trace.thought_loop.decision.selected_hypothesis == "supported_profile_with_gap"
    assert trace.thought_loop.decision.action.next_action == "answer_with_gap"
    assert trace.thought_loop.decision.action.forbidden_claims == ("unsupported mechanism",)
    assert trace.thought_loop.decision.action.next_questions[0] == (
        "What does Starlink use to provide that service?"
    )


def test_thought_loop_identifies_gap_when_no_fallback_is_supported():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[],
    )
    trace = reason_over_plan(build_speech_plan(result, "How does Starlink work?"))

    loop = run_thought_loop(trace)

    assert loop.stop_reason == "blocked_missing_evidence"
    assert [h.status for h in loop.hypotheses] == ["rejected", "rejected"]
    assert loop.next_questions == (
        "What mechanism facts are known about Starlink?",
        "What does Starlink use or do to work?",
    )
    assert loop.steps[-1].operation == "identify_gap"


def test_thought_loop_decision_asks_for_missing_evidence_when_blocked():
    result = SynthesisAnswer(
        subject="Starlink",
        matched=True,
        match_kind="exact",
        definition="satellite internet constellation operated by SpaceX",
        entity_type="organization",
        groups=[],
    )
    trace = reason_over_plan(build_speech_plan(result, "How does Starlink work?"))

    loop = trace.thought_loop

    assert loop is not None
    assert loop.stop_reason == "blocked_missing_evidence"
    assert loop.decision is not None
    assert loop.decision.selected_hypothesis is None
    assert loop.decision.action.next_action == "ask_clarification"
    assert loop.decision.action.next_questions == (
        "What mechanism facts are known about Starlink?",
        "What does Starlink use or do to work?",
    )


def test_thought_loop_accepts_supported_profile_conclusion():
    trace = _spacex_profile_trace()

    loop = run_thought_loop(trace)

    assert loop.subject == "SpaceX"
    assert loop.stop_reason == "supported_conclusion"
    assert loop.accepted_hypothesis is not None
    assert loop.accepted_hypothesis.name == "supported_conclusion"
    assert loop.rejected_hypotheses == ()


def test_thought_loop_profile_decision_answers_supported_conclusion():
    trace = _spacex_profile_trace()

    loop = trace.thought_loop

    assert loop is not None
    assert loop.decision is not None
    assert loop.decision.selected_hypothesis == "supported_conclusion"
    assert loop.decision.action.next_action == "answer"
    assert "unsupported expansion" in loop.decision.action.forbidden_claims


def test_cognitive_session_explains_task_plan_from_last_trace():
    session = CognitiveSession()
    session.record_trace("How does Starlink work?", _starlink_mechanism_gap_trace())

    plan = session.continue_turn("какой план?")

    assert plan.intent == "explain_task_plan"
    assert plan.subject == "Starlink"
    assert plan.task_plan is not None
    assert plan.task_plan.blocked is True
    assert plan.next_questions[0] == "What does Starlink use to provide that service?"
    assert "current goal is to explain how Starlink works" in plan.answer_text


def test_cognitive_session_marks_new_task_without_prior_trace():
    session = CognitiveSession()

    plan = session.continue_turn("Tell me about SpaceX.")

    assert plan.intent == "new_task"
    assert plan.needs_new_reasoning is True
    assert plan.subject is None


def test_cognitive_session_routes_explicit_new_task_after_trace():
    session = CognitiveSession()
    session.record_trace("How does Starlink work?", _starlink_mechanism_gap_trace())

    plan = session.continue_turn("Tell me about SpaceX.")

    assert plan.intent == "new_task"
    assert plan.needs_new_reasoning is True
    assert plan.subject is None


def test_cognitive_session_does_not_guess_unclear_followup_without_active_subgoal():
    session = CognitiveSession()
    session.record_trace("Tell me about SpaceX.", _spacex_profile_trace())

    plan = session.continue_turn("а оно?")

    assert plan.intent == "unresolved_followup"
    assert plan.subject == "SpaceX"
    assert plan.needs_new_reasoning is False
    assert "cannot safely continue" in plan.answer_text


def test_cognitive_answer_session_records_trace_then_explains_missing_evidence():
    session = CognitiveAnswerSession("pump-dry-run")

    first = session.ask("How does Starlink work?")
    second = session.ask("что нужно знать?")

    assert first.decision == "answer"
    assert first.reasoning_trace is not None
    assert first.reasoning_trace.task.subject == "Starlink"
    assert second.decision == "answer"
    assert second.source_system == "cognitive_session"
    assert second.session_plan is not None
    assert second.session_plan.intent == "explain_missing_evidence"
    assert "missing evidence is mechanism" in second.answer_text
    assert "What does Starlink use to provide that service?" in second.answer_text
    assert len(session.dialogue_context.turns) == 1
    assert second.task_memory_snapshot is not None
    assert second.task_memory_snapshot.events[-1].kind == "session_plan_used"


def test_cognitive_answer_session_uses_thought_loop_action_for_surface_text():
    session = CognitiveAnswerSession("pump-dry-run")

    result = session.ask("How does Starlink work?")

    assert result.reasoning_trace is not None
    assert result.reasoning_trace.thought_loop is not None
    assert result.reasoning_trace.thought_loop.decision is not None
    assert (
        result.reasoning_trace.thought_loop.decision.action.next_action
        == "answer_with_gap"
    )
    assert result.surface_selection_trace is not None
    assert result.answer_text == result.surface_selection_trace.final_text
    assert (
        "I can identify Starlink and describe the service it provides"
        in result.answer_text
        or "I can say what Starlink is and what it provides" in result.answer_text
    )
    assert (
        "To answer that more fully" in result.answer_text
        or "The next useful question" in result.answer_text
    )
    assert "A useful next question would be" not in result.answer_text


def test_cognitive_answer_session_executes_loop_clarification_action():
    session = CognitiveAnswerSession(
        "pump-dry-run",
        orchestrator=_DefinitionOnlyOrchestrator(),
    )

    result = session.ask("How does Starlink work?")

    assert result.reasoning_trace is not None
    assert result.reasoning_trace.thought_loop is not None
    assert result.reasoning_trace.thought_loop.decision is not None
    assert (
        result.reasoning_trace.thought_loop.decision.action.next_action
        == "ask_clarification"
    )
    assert result.surface_selection_trace is not None
    assert result.surface_selection_trace.selected_name == "clarification"
    assert result.answer_text == (
        "To answer that properly, I need one more piece: "
        "What mechanism facts are known about Starlink?"
    )
    assert result.answer_text == result.surface_selection_trace.final_text


def test_cognitive_answer_session_humanizes_thin_profile_answer():
    session = CognitiveAnswerSession("pump-dry-run")

    result = session.ask("Tell me about Ray Kroc.")

    assert result.reasoning_trace is not None
    assert result.reasoning_trace.confidence == "thin"
    assert (
        "That is the basic identification I have for Ray Kroc right now."
        in result.answer_text
        or "Right now I only know the basic identification for Ray Kroc."
        in result.answer_text
    )
    assert "I should not" not in result.answer_text
    assert "unsupported memory" not in result.answer_text


def test_cognitive_answer_session_suggests_next_question_without_rerunning_qa():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    followup = session.ask("что дальше?")

    assert followup.decision == "answer"
    assert followup.answer is None
    assert followup.source_system == "cognitive_session"
    assert followup.session_plan is not None
    assert followup.session_plan.intent == "suggest_next_question"
    assert followup.answer_text.endswith("What does Starlink use to provide that service?")


def test_cognitive_answer_session_returns_task_plan_without_rerunning_qa():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    followup = session.ask("какой план?")

    assert followup.decision == "answer"
    assert followup.answer is None
    assert followup.source_system == "cognitive_session"
    assert followup.session_plan is not None
    assert followup.session_plan.intent == "explain_task_plan"
    assert followup.session_plan.task_plan is not None
    assert followup.session_plan.task_plan.blocked is True
    assert "Blocked: explain mechanism" in followup.answer_text


def test_cognitive_answer_session_records_active_working_task_from_reasoning():
    session = CognitiveAnswerSession("pump-dry-run")

    result = session.ask("How does Starlink work?")

    snapshot = result.task_memory_snapshot
    assert snapshot is not None
    assert snapshot.active_task is not None
    assert snapshot.active_task.subject == "Starlink"
    assert snapshot.active_task.goal == "explain how Starlink works"
    assert snapshot.active_task.active_subgoals == ("explain mechanism",)
    assert snapshot.active_task.missing_roles == ("mechanism",)
    assert snapshot.active_task.next_questions[0] == (
        "What does Starlink use to provide that service?"
    )
    assert snapshot.active_task.selected_candidate == "action_candidate_1_answer-with-gap"
    assert "direct_mechanism_explanation" in snapshot.active_task.rejected_candidates
    assert snapshot.active_task.selected_surface_variant in {
        "baseline",
        "decision_sample_1",
        "combined_sample",
    }
    assert snapshot.events[-1].kind == "active_task_updated"


def test_cognitive_answer_session_parks_previous_working_task_on_subject_switch():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    result = session.ask("Tell me about Blue Origin.")

    snapshot = result.task_memory_snapshot
    assert snapshot is not None
    assert snapshot.active_task is not None
    assert snapshot.active_task.subject == "Blue Origin"
    assert [task.subject for task in snapshot.parked_tasks] == ["Starlink"]
    assert snapshot.parked_tasks[0].status == "parked"
    assert snapshot.parked_tasks[0].missing_roles == ("mechanism",)
    assert [event.kind for event in snapshot.events][-2:] == [
        "task_parked",
        "active_task_updated",
    ]


def test_cognitive_answer_session_lists_open_working_tasks():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    session.ask("Tell me about Blue Origin.")
    result = session.ask("какие задачи открыты?")

    assert result.source_system == "cognitive_task_memory"
    assert "Active: Blue Origin" in result.answer_text
    assert "Parked: Starlink" in result.answer_text
    assert result.task_memory_snapshot is not None
    assert result.task_memory_snapshot.events[-1].kind == "memory_summarized"


def test_cognitive_answer_session_inspects_parked_task_without_resuming_it():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    session.ask("Tell me about Blue Origin.")
    result = session.ask("что было по Starlink?")

    assert result.source_system == "cognitive_task_memory"
    assert "Starlink (parked)" in result.answer_text
    assert "missing: mechanism" in result.answer_text
    assert result.task_memory_snapshot is not None
    assert result.task_memory_snapshot.active_task.subject == "Blue Origin"
    assert [task.subject for task in result.task_memory_snapshot.parked_tasks] == [
        "Starlink"
    ]
    assert result.task_memory_snapshot.events[-1].kind == "task_inspected"


def test_cognitive_answer_session_resumes_parked_task_and_restores_followup_trace():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    session.ask("Tell me about Blue Origin.")
    resumed = session.ask("вернемся к Starlink")
    followup = session.ask("что дальше?")

    assert resumed.source_system == "cognitive_task_memory"
    assert "Resumed: Starlink" in resumed.answer_text
    assert resumed.task_memory_snapshot is not None
    assert resumed.task_memory_snapshot.active_task.subject == "Starlink"
    assert [task.subject for task in resumed.task_memory_snapshot.parked_tasks] == [
        "Blue Origin"
    ]
    assert resumed.task_memory_snapshot.events[-1].kind == "task_resumed"
    assert followup.source_system == "cognitive_session"
    assert followup.session_plan.subject == "Starlink"
    assert "What does Starlink use to provide that service?" in followup.answer_text


def test_cognitive_answer_session_summarizes_active_task_from_memory():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    result = session.ask("резюмируй текущую задачу")

    assert result.source_system == "cognitive_task_memory"
    assert "Active task: Starlink (active)" in result.answer_text
    assert "explain how Starlink works" in result.answer_text
    assert "active subgoals: explain mechanism" in result.answer_text
    assert "Last supported answer: Starlink is a satellite internet constellation" in (
        result.answer_text
    )
    assert result.task_memory_snapshot is not None
    assert result.task_memory_snapshot.events[-1].detail == "active_task"


def test_cognitive_answer_session_explains_known_supported_from_active_memory():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    result = session.ask("объясни что уже знаем")

    assert result.source_system == "cognitive_task_memory"
    assert "For Starlink, the supported answer so far is:" in result.answer_text
    assert "Starlink is a satellite internet constellation operated by SpaceX" in (
        result.answer_text
    )
    assert "Still missing: mechanism." in result.answer_text
    assert "Next useful question: What does Starlink use to provide that service?" in (
        result.answer_text
    )
    assert result.answer_text.count("A useful next question would be") == 0
    assert result.task_memory_snapshot is not None
    assert result.task_memory_snapshot.events[-1].detail == "known_supported"


def test_cognitive_answer_session_explains_known_supported_after_resume():
    session = CognitiveAnswerSession("pump-dry-run")

    session.ask("How does Starlink work?")
    session.ask("Tell me about Blue Origin.")
    session.ask("вернемся к Starlink")
    result = session.ask("расскажи только известное")

    assert result.source_system == "cognitive_task_memory"
    assert "For Starlink, the supported answer so far is:" in result.answer_text
    assert "Still missing: mechanism." in result.answer_text
    assert result.task_memory_snapshot is not None
    assert result.task_memory_snapshot.active_task.subject == "Starlink"


def test_cognitive_answer_session_keeps_existing_dialogue_rewrites():
    session = CognitiveAnswerSession("pump-dry-run")

    first = session.ask("Tell me about Blue Origin.")
    followup = session.ask("а кто основал?")

    assert first.decision == "answer"
    assert followup.effective_question == "Who founded Blue Origin?"
    assert followup.followup_rewrite is not None
    assert followup.followup_rewrite.reason == "omitted_subject_founding"
    assert followup.decision == "answer"
    assert "Jeff Bezos" in followup.answer_text


def test_cognitive_answer_session_does_not_guess_unclear_followup():
    session = CognitiveAnswerSession("pump-dry-run")

    first = session.ask("Tell me about SpaceX.")
    unclear = session.ask("а оно?")

    assert first.reasoning_trace is not None
    assert unclear.source_system == "cognitive_session"
    assert unclear.session_plan is not None
    assert unclear.session_plan.intent == "unresolved_followup"
    assert "cannot safely continue" in unclear.answer_text
    assert len(session.dialogue_context.turns) == 1
