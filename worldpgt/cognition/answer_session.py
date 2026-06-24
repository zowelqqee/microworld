"""Programmatic cognitive answer session.

This module composes the existing dialogue context with the transient cognitive
session. It is deliberately a wrapper: the stable single-shot
``AnswerOrchestrator.answer`` contract stays unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import re

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.context_selector import (
    ACCEPTED_OVERLAY_PATH,
    SNAPSHOT_DRY_RUN_OVERLAY_PATH,
    resolve_overlay,
)
from worldpgt.assistant_surface.types import (
    AssistantAnswer,
    OVERLAY_MODE_CUSTOM_PATH,
)
from worldpgt.cognition.reasoning_engine import reason_over_plan
from worldpgt.cognition.session_engine import CognitiveSession
from worldpgt.cognition.types import (
    ActionPlan,
    AnswerSelectionTrace,
    ReasoningTrace,
    SessionTurnPlan,
    TaskMemorySnapshot,
)
from worldpgt.cognition.working_memory import TaskMemory
from worldpgt.dialogue.conversation_context import (
    ConversationContext,
    ConversationTurn,
)
from worldpgt.dialogue.coreference_resolver import (
    CoreferenceResolution,
    resolve_coreferences,
)
from worldpgt.dialogue.followup_rewriter import FollowupRewrite, rewrite_followup
from worldpgt.entity_qa.entity_question_analyzer import analyze as analyze_entity
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.entity_qa.semantic_speech_planner import build_speech_plan
from worldpgt.entity_qa.symbolic_text_generator import generate_text_with_selection_trace
from worldpgt.entity_qa.types import SemanticQuery
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex


@dataclass(frozen=True)
class CognitiveAnswerResult:
    """One turn through the cognitive answer session."""

    question: str
    effective_question: str
    decision: str
    answer_text: str
    source_system: str
    support_kind: str
    answer: AssistantAnswer | None = None
    session_plan: SessionTurnPlan | None = None
    reasoning_trace: ReasoningTrace | None = None
    surface_selection_trace: AnswerSelectionTrace | None = None
    task_memory_snapshot: TaskMemorySnapshot | None = None
    resolved_references: tuple[str, ...] = ()
    followup_rewrite: FollowupRewrite | None = None
    needs_new_reasoning: bool = False


class CognitiveAnswerSession:
    """Stateful, in-memory dialogue + reasoning wrapper."""

    def __init__(
        self,
        overlay_mode: str = "promoted",
        overlay_path: str | None = None,
        ontology_layer_path: str | None = None,
        *,
        orchestrator: AnswerOrchestrator | None = None,
        surface_index: EntitySurfaceIndex | None = None,
    ) -> None:
        self.overlay_mode = (
            OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode
        )
        self.orchestrator = orchestrator or AnswerOrchestrator(
            overlay_mode,
            overlay_path=overlay_path,
            ontology_layer_path=ontology_layer_path,
        )
        self.dialogue_context = ConversationContext()
        self.cognitive_session = CognitiveSession()
        self.task_memory = TaskMemory()
        self._surface_index = surface_index or _surface_index_for_session(
            overlay_mode,
            overlay_path,
        )

    def ask(self, question: str) -> CognitiveAnswerResult:
        """Answer one turn, using transient session state when appropriate."""

        memory_result = self._answer_task_memory_command(question)
        if memory_result is not None:
            return memory_result

        resolution = resolve_coreferences(
            question,
            self.dialogue_context,
            self._surface_index,
        )
        followup = rewrite_followup(
            resolution.resolved_question,
            self.dialogue_context,
            self._surface_index,
        )
        effective_question = followup.resolved_question
        semantic_query = parse_semantic_query(effective_question, self._surface_index)

        session_plan = self.cognitive_session.continue_turn(effective_question)
        if _can_answer_from_session(session_plan):
            self.task_memory.record_session_plan(
                plan=session_plan,
                answer_text=session_plan.answer_text,
            )
            return CognitiveAnswerResult(
                question=question,
                effective_question=effective_question,
                decision="answer",
                answer_text=session_plan.answer_text,
                source_system="cognitive_session",
                support_kind="safe_policy_answer",
                session_plan=session_plan,
                task_memory_snapshot=self.task_memory.snapshot(),
                resolved_references=_resolved_references(resolution),
                followup_rewrite=followup,
                needs_new_reasoning=False,
            )

        if _unresolved_reference_without_subject(resolution, semantic_query):
            answer = _unresolved_reference_answer(
                question,
                self.overlay_mode,
                resolution.unresolved_reference or "",
            )
        else:
            answer = self.orchestrator.answer(
                effective_question,
                answer_style=followup.answer_style,
            )

        trace, selection_trace = self._record_reasoning_artifacts(
            question=effective_question,
            answer=answer,
            answer_style=followup.answer_style,
        )
        if selection_trace is not None and selection_trace.final_text:
            answer = replace(answer, answer_text=selection_trace.final_text)

        self._record_dialogue_turn(
            question=question,
            semantic_query=semantic_query,
            answer=answer,
        )
        self.task_memory.record_reasoned_answer(
            question=effective_question,
            answer_text=answer.answer_text,
            trace=trace,
            selection_trace=selection_trace,
        )

        return CognitiveAnswerResult(
            question=question,
            effective_question=effective_question,
            decision=answer.decision,
            answer_text=answer.answer_text,
            source_system=answer.source_system,
            support_kind=answer.support_kind,
            answer=answer,
            session_plan=session_plan,
            reasoning_trace=trace,
            surface_selection_trace=selection_trace,
            task_memory_snapshot=self.task_memory.snapshot(),
            resolved_references=_resolved_references(resolution),
            followup_rewrite=followup,
            needs_new_reasoning=session_plan.needs_new_reasoning,
        )

    def _answer_task_memory_command(self, question: str) -> CognitiveAnswerResult | None:
        normalized = _normalize_prompt(question)
        if _is_known_supported_request(normalized):
            answer_text = self.task_memory.known_facts_summary()
            return CognitiveAnswerResult(
                question=question,
                effective_question=question,
                decision="answer",
                answer_text=answer_text,
                source_system="cognitive_task_memory",
                support_kind="safe_policy_answer",
                task_memory_snapshot=self.task_memory.snapshot(),
            )

        if _is_active_summary_request(normalized):
            answer_text = self.task_memory.summarize_active()
            return CognitiveAnswerResult(
                question=question,
                effective_question=question,
                decision="answer",
                answer_text=answer_text,
                source_system="cognitive_task_memory",
                support_kind="safe_policy_answer",
                task_memory_snapshot=self.task_memory.snapshot(),
            )

        if _is_open_tasks_request(normalized):
            answer_text = self.task_memory.describe_open_tasks()
            return CognitiveAnswerResult(
                question=question,
                effective_question=question,
                decision="answer",
                answer_text=answer_text,
                source_system="cognitive_task_memory",
                support_kind="safe_policy_answer",
                task_memory_snapshot=self.task_memory.snapshot(),
            )

        subject = _mentioned_memory_subject(normalized, self.task_memory.known_subjects())
        if subject is None:
            return None

        if _is_resume_request(normalized):
            task = self.task_memory.resume(subject)
            if task is None:
                return None
            self._restore_cognitive_trace(task.subject)
            answer_text = f"Resumed: {_describe_memory_task(task)}"
            return CognitiveAnswerResult(
                question=question,
                effective_question=question,
                decision="answer",
                answer_text=answer_text,
                source_system="cognitive_task_memory",
                support_kind="safe_policy_answer",
                task_memory_snapshot=self.task_memory.snapshot(),
            )

        if _is_task_inspection_request(normalized):
            answer_text = self.task_memory.describe_task(subject)
            if answer_text is None:
                return None
            return CognitiveAnswerResult(
                question=question,
                effective_question=question,
                decision="answer",
                answer_text=answer_text,
                source_system="cognitive_task_memory",
                support_kind="safe_policy_answer",
                task_memory_snapshot=self.task_memory.snapshot(),
            )

        return None

    def _restore_cognitive_trace(self, subject: str) -> None:
        for record in reversed(self.cognitive_session.traces):
            if record.trace.task.subject == subject:
                self.cognitive_session.record_trace(f"[resumed] {subject}", record.trace)
                return

    def _record_reasoning_artifacts(
        self,
        *,
        question: str,
        answer: AssistantAnswer,
        answer_style: str,
    ) -> tuple[ReasoningTrace | None, AnswerSelectionTrace | None]:
        if answer.decision != "answer":
            return None, None

        analyzed = analyze_entity(question)
        plan = self.orchestrator._entity_planner.plan(analyzed)
        if plan.decision != "answer" or plan.render_template != "open_synthesis":
            return None, None
        synthesis = plan.render_args.get("synthesis")
        if synthesis is None or not getattr(synthesis, "matched", False):
            return None, None

        speech_plan = build_speech_plan(
            synthesis,
            question,
            answer_style=answer_style,
        )
        trace = reason_over_plan(speech_plan)
        selection_trace = generate_text_with_selection_trace(
            speech_plan,
            action=_primary_action(trace),
        )
        self.cognitive_session.record_trace(question, trace)
        return trace, selection_trace

    def _record_dialogue_turn(
        self,
        *,
        question: str,
        semantic_query: SemanticQuery,
        answer: AssistantAnswer,
    ) -> None:
        if answer.decision == "audit":
            mentioned: list[str] = []
            primary = None
        else:
            text_entities = [
                canonical
                for _surface, canonical, _start, _end in self._surface_index.find_in_text(
                    answer.answer_text
                )
            ]
            ctx_entities = (
                answer.trace.context_summary.get("matched_entities", [])
                if answer.trace and answer.trace.context_summary
                else []
            )
            mentioned = _dedupe(
                text_entities
                + ctx_entities
                + [semantic_query.entity_a or "", semantic_query.entity_b or ""]
            )
            primary = self._primary_from_answer(semantic_query.entity_a, text_entities)
            if primary is None:
                primary = mentioned[0] if mentioned else None

        if answer.decision == "no" and primary:
            mentioned = _dedupe([primary] + mentioned)

        self.dialogue_context.append_turn(
            ConversationTurn(
                question=question,
                semantic_query=semantic_query,
                decision=answer.decision,
                primary_entity=primary,
                mentioned_entities=mentioned,
                relation_type=semantic_query.relation_intent,
            )
        )

    def _primary_from_answer(
        self,
        entity_a: str | None,
        text_entities: list[str],
    ) -> str | None:
        if not entity_a:
            return text_entities[0] if text_entities else None
        ea_type = self._surface_index.entity_type(entity_a)
        if ea_type not in ("organization", "vehicle", "program"):
            return entity_a
        persons = [
            entity
            for entity in text_entities
            if self._surface_index.entity_type(entity) == "person"
        ]
        if len(persons) == 1:
            return persons[0]
        return entity_a


def _can_answer_from_session(plan: SessionTurnPlan) -> bool:
    if plan.needs_new_reasoning:
        return False
    if plan.intent == "new_task":
        return False
    return bool(plan.answer_text)


def _primary_action(trace: ReasoningTrace) -> ActionPlan:
    if trace.thought_loop is not None and trace.thought_loop.decision is not None:
        return trace.thought_loop.decision.action
    return trace.action


def _surface_index_for_session(
    overlay_mode: str,
    overlay_path: str | None,
) -> EntitySurfaceIndex:
    resolved_path = overlay_path or resolve_overlay(overlay_mode)[0]
    return EntitySurfaceIndex(
        accepted_overlay_path=ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=Path(resolved_path),
        snapshot_overlay_path=SNAPSHOT_DRY_RUN_OVERLAY_PATH,
    )


def _unresolved_reference_without_subject(
    resolution: CoreferenceResolution,
    semantic_query: SemanticQuery,
) -> bool:
    return (
        resolution.unresolved_reference is not None
        and semantic_query.entity_a is None
        and semantic_query.entity_b is None
    )


def _unresolved_reference_answer(
    question: str,
    overlay_mode: str,
    reference: str,
) -> AssistantAnswer:
    return AssistantAnswer(
        question=question,
        decision="audit",
        route="unknown_or_unsupported",
        answer_text=f"unresolved_reference: could not determine what {reference!r} refers to",
        overlay_mode=overlay_mode,
        supported_by_context=False,
        support_kind="missing_knowledge",
        source_system="dialogue_context",
        safe_for_general_runtime=False,
    )


def _resolved_references(resolution: CoreferenceResolution) -> tuple[str, ...]:
    return tuple(
        f"[{replacement.reference} -> {replacement.display_target}]"
        for replacement in resolution.replacements
    )


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _is_open_tasks_request(normalized: str) -> bool:
    return any(
        pattern in normalized
        for pattern in (
            "open tasks",
            "what tasks",
            "what is open",
            "какие задачи",
            "что открыто",
            "что в памяти",
            "задачи открыты",
            "открытые задачи",
        )
    )


def _is_active_summary_request(normalized: str) -> bool:
    return any(
        pattern in normalized
        for pattern in (
            "summarize current task",
            "summarize active task",
            "current task summary",
            "what is the current task",
            "резюмируй текущую задачу",
            "суммаризируй текущую задачу",
            "что за текущая задача",
            "текущая задача",
        )
    )


def _is_known_supported_request(normalized: str) -> bool:
    return any(
        pattern in normalized
        for pattern in (
            "what do we know so far",
            "what is known so far",
            "explain what we know",
            "explain known facts",
            "known supported",
            "summarize known",
            "only known",
            "что уже знаем",
            "что мы уже знаем",
            "объясни что уже знаем",
            "расскажи что уже знаем",
            "расскажи только известное",
            "только известное",
            "известные факты",
        )
    )


def _is_resume_request(normalized: str) -> bool:
    return any(
        pattern in normalized
        for pattern in (
            "resume",
            "return to",
            "back to",
            "вернемся",
            "вернёмся",
            "вернуться",
            "возобнови",
            "продолжим",
        )
    )


def _is_task_inspection_request(normalized: str) -> bool:
    return any(
        pattern in normalized
        for pattern in (
            "what was about",
            "what did we have on",
            "what do we have on",
            "show task",
            "что было по",
            "что у нас было по",
            "что по",
            "покажи по",
        )
    )


def _mentioned_memory_subject(
    normalized: str,
    subjects: tuple[str, ...],
) -> str | None:
    matches = [
        subject
        for subject in subjects
        if re.search(rf"(?<!\w){re.escape(subject.lower())}(?!\w)", normalized)
    ]
    if len(matches) == 1:
        return matches[0]
    return None


def _describe_memory_task(task) -> str:
    text = f"{task.subject} — {task.goal}"
    if task.missing_roles:
        text += f"; missing: {_join_list([str(role) for role in task.missing_roles])}"
    if task.next_questions:
        text += f"; next: {task.next_questions[0]}"
    return _finish_sentence(text)


def _join_list(items: list[str]) -> str:
    items = [item for item in dict.fromkeys(items) if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def _finish_sentence(text: str) -> str:
    return text if text.endswith((".", "?", "!")) else f"{text}."
