"""Deliberation over possible reasoning moves.

This layer does not add facts. It creates a small candidate set, rejects moves
that require missing evidence, and selects the highest-priority accepted move.
"""

from __future__ import annotations

from worldpgt.cognition.types import (
    ActionCandidate,
    ActionPlan,
    CandidateEvaluation,
    DeliberationTrace,
    EvidenceRole,
    EvidenceWorkspace,
    MissingEvidence,
    TaskFrame,
    TaskSubgoal,
    ThoughtCandidate,
)


def deliberate(
    *,
    task: TaskFrame,
    workspace: EvidenceWorkspace,
    subgoals: tuple[TaskSubgoal, ...],
    action_candidates: tuple[ActionCandidate, ...],
    missing_evidence: tuple[MissingEvidence, ...],
) -> DeliberationTrace:
    """Consider possible next moves and select the best supported one."""

    candidates = _candidate_moves(
        task=task,
        action_candidates=action_candidates,
        missing_evidence=missing_evidence,
    )
    evaluations = tuple(
        _evaluate_candidate(candidate, workspace, subgoals)
        for candidate in candidates
    )
    selected = _select_candidate(candidates, evaluations)
    return DeliberationTrace(
        candidates=candidates,
        evaluations=evaluations,
        selected_name=selected.name if selected is not None else None,
    )


def _candidate_moves(
    *,
    task: TaskFrame,
    action_candidates: tuple[ActionCandidate, ...],
    missing_evidence: tuple[MissingEvidence, ...],
) -> tuple[ThoughtCandidate, ...]:
    candidates: list[ThoughtCandidate] = []
    if task.intent == "mechanism_explanation":
        candidates.append(
            ThoughtCandidate(
                name="direct_mechanism_explanation",
                action=ActionPlan(
                    next_action="answer",
                    surface_goal=f"explain how {task.subject} works directly",
                    preferred_buckets=("mechanism",),
                    detail_unit_limit=1,
                ),
                claim="state the operating mechanism",
                required_roles=("mechanism",),
                priority=100,
            )
        )

    for idx, candidate in enumerate(action_candidates, start=1):
        name = _candidate_name(candidate, idx)
        candidates.append(
            ThoughtCandidate(
                name=name,
                action=candidate.action,
                claim=candidate.action.surface_goal,
                required_roles=_required_roles_for_action(candidate.action),
                priority=candidate.priority,
            )
        )

    if missing_evidence:
        next_questions = tuple(
            dict.fromkeys(
                question
                for missing in missing_evidence
                for question in missing.next_questions
            )
        )
        candidates.append(
            ThoughtCandidate(
                name="ask_for_missing_evidence",
                action=ActionPlan(
                    next_action="ask_clarification",
                    surface_goal="ask for the missing evidence needed to continue",
                    next_questions=next_questions,
                ),
                claim="request missing evidence instead of answering unsupported content",
                priority=60,
            )
        )

    return tuple(candidates)


def _candidate_name(candidate: ActionCandidate, idx: int) -> str:
    label = candidate.action.next_action.replace("_", "-")
    return f"action_candidate_{idx}_{label}"


def _required_roles_for_action(action: ActionPlan) -> tuple[EvidenceRole, ...]:
    if action.next_action == "answer_with_gap":
        return ()
    if "mechanism" in action.surface_goal and "gap" not in action.surface_goal:
        return ("mechanism",)
    return ()


def _evaluate_candidate(
    candidate: ThoughtCandidate,
    workspace: EvidenceWorkspace,
    subgoals: tuple[TaskSubgoal, ...],
) -> CandidateEvaluation:
    available = set(workspace.roles())
    missing_roles = tuple(
        role for role in candidate.required_roles if role not in available
    )
    if missing_roles:
        return CandidateEvaluation(
            candidate_name=candidate.name,
            decision="rejected",
            reason=f"missing required evidence: {_join_roles(missing_roles)}",
            missing_roles=missing_roles,
        )

    blocked_roles = tuple(
        role
        for subgoal in subgoals
        if subgoal.status == "missing"
        for role in subgoal.missing_roles
        if role in candidate.required_roles
    )
    if blocked_roles:
        return CandidateEvaluation(
            candidate_name=candidate.name,
            decision="rejected",
            reason=f"blocked by unsatisfied subgoal: {_join_roles(blocked_roles)}",
            missing_roles=blocked_roles,
        )

    return CandidateEvaluation(
        candidate_name=candidate.name,
        decision="accepted",
        reason="candidate is supported by current evidence boundaries",
    )


def _select_candidate(
    candidates: tuple[ThoughtCandidate, ...],
    evaluations: tuple[CandidateEvaluation, ...],
) -> ThoughtCandidate | None:
    accepted = {
        evaluation.candidate_name
        for evaluation in evaluations
        if evaluation.decision == "accepted"
    }
    selectable = [
        candidate for candidate in candidates if candidate.name in accepted
    ]
    if not selectable:
        return None
    return sorted(selectable, key=lambda candidate: candidate.priority, reverse=True)[0]


def _join_roles(roles: tuple[EvidenceRole, ...]) -> str:
    values = [str(role) for role in dict.fromkeys(roles)]
    if len(values) == 1:
        return values[0]
    if len(values) == 2:
        return f"{values[0]} and {values[1]}"
    return ", ".join(values[:-1]) + f", and {values[-1]}"
