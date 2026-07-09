"""Iterative thought loop over bounded reasoning traces.

The loop is deliberately small: it does not fetch facts and does not mutate
memory. It repeatedly proposes and tests hypotheses against the evidence already
admitted into a ``ReasoningTrace``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from worldpgt.cognition.types import (
    ActionPlan,
    EvidenceRole,
    MissingEvidence,
    ReasoningTrace,
    ThoughtHypothesis,
    ThoughtLoopDecision,
    ThoughtLoopStep,
    ThoughtLoopStopReason,
    ThoughtLoopTrace,
)


@dataclass
class _LoopState:
    trace: ReasoningTrace
    steps: list[ThoughtLoopStep] = field(default_factory=list)
    hypotheses: list[ThoughtHypothesis] = field(default_factory=list)
    stop_reason: ThoughtLoopStopReason | None = None
    next_questions: tuple[str, ...] = ()
    max_steps: int = 8


def run_thought_loop(
    trace: ReasoningTrace,
    *,
    max_steps: int = 8,
) -> ThoughtLoopTrace:
    """Run an iterative thinking loop over a reasoning trace."""

    state = _LoopState(trace=trace, max_steps=max_steps)
    while state.stop_reason is None and len(state.steps) < state.max_steps:
        operation = _next_operation(state)
        _execute(operation, state)

    if state.stop_reason is None:
        state.stop_reason = "max_steps"
        _add_step(state, "stop", "stopped after max steps")

    accepted = next(
        (hypothesis for hypothesis in state.hypotheses if hypothesis.status == "accepted"),
        None,
    )
    rejected = tuple(
        hypothesis for hypothesis in state.hypotheses if hypothesis.status == "rejected"
    )
    decision = _decision_for_state(state, accepted)
    return ThoughtLoopTrace(
        subject=trace.task.subject,
        goal=_goal_for_trace(trace),
        steps=tuple(state.steps),
        hypotheses=tuple(state.hypotheses),
        accepted_hypothesis=accepted,
        rejected_hypotheses=rejected,
        stop_reason=state.stop_reason,
        next_questions=state.next_questions,
        decision=decision,
    )


def _next_operation(state: _LoopState) -> str:
    operations = [step.operation for step in state.steps]
    if "inspect_evidence" not in operations:
        return "inspect_evidence"
    if not state.hypotheses:
        return "form_hypothesis"
    if any(hypothesis.status == "pending" for hypothesis in state.hypotheses):
        return "test_hypothesis"
    if (
        state.trace.missing_evidence
        and not any(step.operation == "identify_gap" for step in state.steps)
    ):
        return "identify_gap"
    return "stop"


def _execute(operation: str, state: _LoopState) -> None:
    if operation == "inspect_evidence":
        roles = ", ".join(str(role) for role in state.trace.workspace.roles())
        _add_step(state, "inspect_evidence", f"available evidence roles: {roles}")
    elif operation == "form_hypothesis":
        _form_hypotheses(state)
    elif operation == "test_hypothesis":
        _test_next_hypothesis(state)
    elif operation == "identify_gap":
        _identify_gap(state)
    else:
        _stop(state)


def _form_hypotheses(state: _LoopState) -> None:
    if state.trace.task.intent == "mechanism_explanation":
        state.hypotheses.append(
            ThoughtHypothesis(
                name="direct_mechanism",
                text=f"{state.trace.task.subject} can be explained by a mechanism.",
                required_roles=("mechanism",),
            )
        )
        state.hypotheses.append(
            ThoughtHypothesis(
                name="supported_profile_with_gap",
                text="Answer the supported profile and name the mechanism gap.",
                required_roles=_fallback_roles(state.trace),
            )
        )
        _add_step(
            state,
            "form_hypothesis",
            "formed direct mechanism and supported profile hypotheses",
        )
        return

    if state.trace.primary_conclusion is not None:
        state.hypotheses.append(
            ThoughtHypothesis(
                name="supported_conclusion",
                text=state.trace.primary_conclusion.text,
                required_roles=state.trace.primary_conclusion.evidence_roles,
            )
        )
        _add_step(state, "form_hypothesis", "formed supported conclusion hypothesis")
        return

    roles = tuple(role for role in state.trace.workspace.roles() if role != "definition")
    state.hypotheses.append(
        ThoughtHypothesis(
            name="direct_supported_answer",
            text="Use the available supported facts directly.",
            required_roles=roles,
        )
    )
    _add_step(state, "form_hypothesis", "formed direct supported answer hypothesis")


def _test_next_hypothesis(state: _LoopState) -> None:
    hypothesis = next(
        hypothesis for hypothesis in state.hypotheses if hypothesis.status == "pending"
    )
    available = set(state.trace.workspace.roles())
    missing = tuple(role for role in hypothesis.required_roles if role not in available)
    if missing:
        _replace_hypothesis(
            state,
            hypothesis,
            ThoughtHypothesis(
                name=hypothesis.name,
                text=hypothesis.text,
                required_roles=hypothesis.required_roles,
                status="rejected",
                reason=f"missing evidence: {_join_roles(missing)}",
            ),
        )
        _add_step(
            state,
            "test_hypothesis",
            f"rejected {hypothesis.name}: missing {_join_roles(missing)}",
        )
        return

    _replace_hypothesis(
        state,
        hypothesis,
        ThoughtHypothesis(
            name=hypothesis.name,
            text=hypothesis.text,
            required_roles=hypothesis.required_roles,
            status="accepted",
            reason="supported by current evidence",
        ),
    )
    _add_step(state, "test_hypothesis", f"accepted {hypothesis.name}")
    state.stop_reason = "supported_conclusion"


def _identify_gap(state: _LoopState) -> None:
    questions: list[str] = []
    roles: list[str] = []
    for missing in state.trace.missing_evidence:
        roles.append(str(missing.role))
        questions.extend(missing.next_questions)
    state.next_questions = tuple(dict.fromkeys(questions))
    state.stop_reason = "blocked_missing_evidence"
    _add_step(
        state,
        "identify_gap",
        f"blocked by missing {_join_list(roles)} evidence",
    )


def _stop(state: _LoopState) -> None:
    if any(hypothesis.status == "accepted" for hypothesis in state.hypotheses):
        state.stop_reason = "supported_conclusion"
        _add_step(state, "stop", "accepted supported conclusion")
    elif state.trace.missing_evidence:
        _identify_gap(state)
    else:
        state.stop_reason = "max_steps"
        _add_step(state, "stop", "no further operation selected")


def _decision_for_state(
    state: _LoopState,
    accepted: ThoughtHypothesis | None,
) -> ThoughtLoopDecision:
    if accepted is not None:
        return _accepted_decision(state.trace, accepted, state.next_questions)
    if state.stop_reason == "blocked_missing_evidence":
        return ThoughtLoopDecision(
            action=ActionPlan(
                next_action="ask_clarification",
                surface_goal="ask for the missing evidence needed to continue",
                forbidden_claims=("unsupported answer",),
                next_questions=state.next_questions or _questions_from_missing(state.trace.missing_evidence),
            ),
            reason="no hypothesis is supported by the current evidence",
        )
    return ThoughtLoopDecision(
        action=ActionPlan(
            next_action="ask_clarification",
            surface_goal="ask for a narrower question before continuing",
            forbidden_claims=("unsupported answer",),
        ),
        reason="the loop stopped without selecting a supported hypothesis",
    )


def _accepted_decision(
    trace: ReasoningTrace,
    accepted: ThoughtHypothesis,
    next_questions: tuple[str, ...],
) -> ThoughtLoopDecision:
    if accepted.name == "supported_profile_with_gap":
        questions = next_questions or _questions_from_missing(trace.missing_evidence)
        return ThoughtLoopDecision(
            action=ActionPlan(
                next_action="answer_with_gap",
                surface_goal="answer the supported profile and name the mechanism gap",
                forbidden_claims=("unsupported mechanism",),
                preferred_buckets=("purpose", "classification", "ownership", "activity"),
                suppressed_buckets=("other",),
                detail_unit_limit=2,
                next_questions=questions,
            ),
            selected_hypothesis=accepted.name,
            reason="fallback profile evidence is supported but mechanism evidence is missing",
        )

    if accepted.name == "supported_conclusion" and trace.primary_conclusion is not None:
        return ThoughtLoopDecision(
            action=ActionPlan(
                next_action="answer",
                surface_goal="state the supported conclusion",
                forbidden_claims=("unsupported expansion",),
                preferred_buckets=tuple(str(role) for role in trace.primary_conclusion.evidence_roles),
                suppressed_buckets=("other",),
            ),
            selected_hypothesis=accepted.name,
            reason="the selected hypothesis is fully supported by current evidence",
        )

    return ThoughtLoopDecision(
        action=ActionPlan(
            next_action="answer",
            surface_goal="state supported facts directly",
            forbidden_claims=("unsupported expansion",),
            preferred_buckets=tuple(str(role) for role in accepted.required_roles),
            suppressed_buckets=("other",),
        ),
        selected_hypothesis=accepted.name,
        reason="the selected hypothesis is supported by current evidence",
    )


def _questions_from_missing(
    missing_evidence: tuple[MissingEvidence, ...],
) -> tuple[str, ...]:
    questions: list[str] = []
    for missing in missing_evidence:
        questions.extend(missing.next_questions)
    return tuple(dict.fromkeys(questions))


def _replace_hypothesis(
    state: _LoopState,
    old: ThoughtHypothesis,
    new: ThoughtHypothesis,
) -> None:
    idx = state.hypotheses.index(old)
    state.hypotheses[idx] = new


def _fallback_roles(trace: ReasoningTrace) -> tuple[EvidenceRole, ...]:
    roles: list[EvidenceRole] = []
    for role in ("purpose", "classification", "ownership", "activity"):
        if trace.workspace.texts_for(role):
            roles.append(role)
    return tuple(roles) if roles else ("purpose",)


def _goal_for_trace(trace: ReasoningTrace) -> str:
    if trace.task.intent == "mechanism_explanation":
        return f"explain how {trace.task.subject} works"
    if trace.task.intent == "followup_answer":
        return f"continue the current answer about {trace.task.subject}"
    return f"build a supported profile of {trace.task.subject}"


def _add_step(state: _LoopState, operation: str, output: str) -> None:
    state.steps.append(
        ThoughtLoopStep(
            index=len(state.steps),
            operation=operation,
            output=output,
        )
    )


def _join_roles(roles: tuple[EvidenceRole, ...]) -> str:
    return _join_list([str(role) for role in roles])


def _join_list(items: list[str]) -> str:
    items = [item for item in dict.fromkeys(items) if item]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
