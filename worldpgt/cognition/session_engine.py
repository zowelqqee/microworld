"""In-process cognitive session state.

The session engine is intentionally small and transient. It stores recent
reasoning traces so later turns can continue an unfinished task, explain
missing evidence, or suggest the next useful question without mutating memory
or guessing new facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from worldpgt.cognition.task_planner import build_task_plan
from worldpgt.cognition.types import (
    MissingEvidence,
    ReasoningTrace,
    SessionTraceRecord,
    SessionTurnPlan,
)


_MISSING_EVIDENCE_PATTERNS = (
    "what is missing",
    "what's missing",
    "what do you need",
    "what should we know",
    "what do we need to know",
    "why can't you explain",
    "why can you not explain",
    "чего не хватает",
    "что не хватает",
    "что нужно знать",
    "что надо знать",
    "почему не можешь",
    "почему нельзя объяснить",
)

_NEXT_QUESTION_PATTERNS = (
    "what next",
    "what should i ask",
    "what should we ask",
    "next question",
    "useful next question",
    "что дальше",
    "куда дальше",
    "что спросить",
    "следующий вопрос",
)

_TASK_PLAN_PATTERNS = (
    "what is the plan",
    "what's the plan",
    "how do we continue",
    "how should we continue",
    "what can you do next",
    "plan the next step",
    "какой план",
    "план",
    "как продолжить",
    "что делаем дальше",
    "что можно сделать дальше",
)

_NEW_TASK_PATTERNS = (
    "tell me about",
    "what do you know about",
    "how does",
    "who founded",
    "what is ",
    "who is ",
    "расскажи",
    "кто основал",
)


@dataclass
class CognitiveSession:
    """Transient task-level state for bounded reasoning sessions."""

    traces: list[SessionTraceRecord] = field(default_factory=list)
    turn_plans: list[SessionTurnPlan] = field(default_factory=list)

    def record_trace(self, question: str, trace: ReasoningTrace) -> SessionTraceRecord:
        record = SessionTraceRecord(question=question, trace=trace)
        self.traces.append(record)
        return record

    def last_trace(self) -> ReasoningTrace | None:
        return self.traces[-1].trace if self.traces else None

    def continue_turn(self, question: str) -> SessionTurnPlan:
        trace = self.last_trace()
        normalized = _normalize(question)
        if trace is None:
            plan = SessionTurnPlan(
                question=question,
                intent="new_task",
                needs_new_reasoning=True,
            )
            self.turn_plans.append(plan)
            return plan

        if _matches(normalized, _MISSING_EVIDENCE_PATTERNS):
            plan = _missing_evidence_plan(question, trace)
        elif _matches(normalized, _TASK_PLAN_PATTERNS):
            plan = _task_plan(question, trace)
        elif _matches(normalized, _NEXT_QUESTION_PATTERNS):
            plan = _next_question_plan(question, trace)
        elif _matches(normalized, _NEW_TASK_PATTERNS):
            plan = SessionTurnPlan(
                question=question,
                intent="new_task",
                subject=None,
                needs_new_reasoning=True,
            )
        elif trace.working_memory.active_subgoals:
            plan = _continue_task_plan(question, trace)
        else:
            plan = SessionTurnPlan(
                question=question,
                intent="unresolved_followup",
                subject=trace.task.subject,
                answer_text=(
                    "I cannot safely continue this as the same task without a "
                    "clear reference or a new question."
                ),
            )

        self.turn_plans.append(plan)
        return plan


def _missing_evidence_plan(question: str, trace: ReasoningTrace) -> SessionTurnPlan:
    missing = trace.missing_evidence
    next_questions = _next_questions(trace)
    if missing:
        answer = _missing_evidence_text(trace.task.subject, missing, next_questions)
    else:
        answer = f"I do not have an active missing-evidence subgoal for {trace.task.subject}."
    return SessionTurnPlan(
        question=question,
        intent="explain_missing_evidence",
        subject=trace.task.subject,
        active_subgoals=trace.working_memory.active_subgoals,
        missing_evidence=missing,
        next_questions=next_questions,
        answer_text=answer,
    )


def _next_question_plan(question: str, trace: ReasoningTrace) -> SessionTurnPlan:
    next_questions = _next_questions(trace)
    if next_questions:
        answer = f"A useful next question would be: {next_questions[0]}"
    else:
        answer = f"I do not have a pending next question for {trace.task.subject}."
    return SessionTurnPlan(
        question=question,
        intent="suggest_next_question",
        subject=trace.task.subject,
        active_subgoals=trace.working_memory.active_subgoals,
        missing_evidence=trace.missing_evidence,
        next_questions=next_questions,
        answer_text=answer,
    )


def _task_plan(question: str, trace: ReasoningTrace) -> SessionTurnPlan:
    task_plan = build_task_plan(trace)
    return SessionTurnPlan(
        question=question,
        intent="explain_task_plan",
        subject=trace.task.subject,
        active_subgoals=trace.working_memory.active_subgoals,
        missing_evidence=trace.missing_evidence,
        next_questions=task_plan.next_questions,
        task_plan=task_plan,
        answer_text=task_plan.answer_text,
    )


def _continue_task_plan(question: str, trace: ReasoningTrace) -> SessionTurnPlan:
    missing = trace.missing_evidence
    next_questions = _next_questions(trace)
    task_plan = build_task_plan(trace)
    return SessionTurnPlan(
        question=question,
        intent="continue_task",
        subject=trace.task.subject,
        active_subgoals=trace.working_memory.active_subgoals,
        missing_evidence=missing,
        next_questions=next_questions,
        task_plan=task_plan,
        answer_text=_missing_evidence_text(trace.task.subject, missing, next_questions),
    )


def _missing_evidence_text(
    subject: str,
    missing: tuple[MissingEvidence, ...],
    next_questions: tuple[str, ...],
) -> str:
    if not missing:
        return f"There is no active missing-evidence item for {subject}."
    roles = _join_list([m.role for m in missing])
    reasons = _join_list([m.reason for m in missing])
    text = f"To continue the {subject} task, the missing evidence is {roles}: {reasons}."
    if next_questions:
        text += f" A useful next question would be: {next_questions[0]}"
    return text


def _next_questions(trace: ReasoningTrace) -> tuple[str, ...]:
    if trace.action.next_questions:
        return trace.action.next_questions
    out: list[str] = []
    for missing in trace.missing_evidence:
        out.extend(missing.next_questions)
    return tuple(dict.fromkeys(out))


def _normalize(question: str) -> str:
    text = re.sub(r"[?.!]+$", "", question.strip().lower())
    return re.sub(r"\s+", " ", text)


def _matches(normalized: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in normalized for pattern in patterns)


def _join_list(items: list[str]) -> str:
    items = [item for item in dict.fromkeys(items) if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
