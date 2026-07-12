"""Task planning over explicit reasoning traces.

The planner does not look up new facts. It turns the current reasoning trace
into a bounded continuation plan: satisfied steps, blocked steps, and useful
next questions.
"""

from __future__ import annotations

from worldpgt.cognition.types import (
    CognitiveTaskPlan,
    MissingEvidence,
    ReasoningTrace,
    TaskPlanStep,
    TaskSubgoal,
)


def build_task_plan(trace: ReasoningTrace) -> CognitiveTaskPlan:
    """Build an inspectable task plan from one reasoning trace."""

    steps: list[TaskPlanStep] = []
    for subgoal in trace.subgoals:
        steps.append(_step_for_subgoal(subgoal, trace.missing_evidence))

    next_questions = _next_questions(trace)
    blocked = any(step.status == "blocked" for step in steps)
    goal = _goal_for_trace(trace)
    answer_text = _render_task_plan(
        subject=trace.task.subject,
        goal=goal,
        steps=tuple(steps),
        blocked=blocked,
        next_questions=next_questions,
    )
    return CognitiveTaskPlan(
        subject=trace.task.subject,
        goal=goal,
        steps=tuple(steps),
        blocked=blocked,
        next_questions=next_questions,
        answer_text=answer_text,
    )


def _step_for_subgoal(
    subgoal: TaskSubgoal,
    missing_evidence: tuple[MissingEvidence, ...],
) -> TaskPlanStep:
    role = subgoal.missing_roles[0] if subgoal.missing_roles else None
    missing = _missing_for_role(role, missing_evidence) if role else None
    if subgoal.status == "satisfied":
        return TaskPlanStep(
            title=subgoal.name,
            status="satisfied",
            evidence_role=role,
            reason="supported by current evidence",
        )
    if missing is not None:
        return TaskPlanStep(
            title=subgoal.name,
            status="blocked",
            evidence_role=missing.role,
            reason=missing.reason,
            next_question=missing.next_questions[0] if missing.next_questions else None,
        )
    return TaskPlanStep(
        title=subgoal.name,
        status="actionable",
        evidence_role=role,
        reason="partially supported; continue using available evidence",
    )


def _missing_for_role(
    role,
    missing_evidence: tuple[MissingEvidence, ...],
) -> MissingEvidence | None:
    for missing in missing_evidence:
        if missing.role == role:
            return missing
    return None


def _goal_for_trace(trace: ReasoningTrace) -> str:
    if trace.task.intent == "mechanism_explanation":
        return f"explain how {trace.task.subject} works"
    if trace.task.intent == "followup_answer":
        return f"continue the current answer about {trace.task.subject}"
    return f"build a supported profile of {trace.task.subject}"


def _next_questions(trace: ReasoningTrace) -> tuple[str, ...]:
    if trace.action.next_questions:
        return trace.action.next_questions
    out: list[str] = []
    for missing in trace.missing_evidence:
        out.extend(missing.next_questions)
    return tuple(dict.fromkeys(out))


def _render_task_plan(
    *,
    subject: str,
    goal: str,
    steps: tuple[TaskPlanStep, ...],
    blocked: bool,
    next_questions: tuple[str, ...],
) -> str:
    satisfied = [step.title for step in steps if step.status == "satisfied"]
    blocked_steps = [step for step in steps if step.status == "blocked"]
    actionable = [step.title for step in steps if step.status == "actionable"]

    parts = [f"For {subject}, the current goal is to {goal}."]
    if satisfied:
        parts.append(f"Already grounded: {_join_list(satisfied)}.")
    if actionable:
        parts.append(f"Can continue with: {_join_list(actionable)}.")
    if blocked_steps:
        reasons = _join_list(
            [
                f"{step.title} ({step.reason})"
                for step in blocked_steps
                if step.reason
            ]
        )
        parts.append(f"Blocked: {reasons}.")
    elif not blocked:
        parts.append("No blocked step is active.")
    if next_questions:
        parts.append(f"Next useful question: {next_questions[0]}")
    return " ".join(part for part in parts if part).strip()


def _join_list(items: list[str]) -> str:
    items = [item for item in dict.fromkeys(items) if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
