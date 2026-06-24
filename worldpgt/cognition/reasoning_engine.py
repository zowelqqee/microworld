"""Explicit reasoning engine for bounded language tasks.

The engine operates over an already-built speech plan. It does not query raw
memory, does not invent facts, and does not decide truth. Its job is to turn
admitted evidence into an inspectable reasoning trace and an action plan that a
surface realizer can execute.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from worldpgt.cognition.deliberation_engine import deliberate
from worldpgt.cognition.support_guard import validate_trace_conclusions
from worldpgt.cognition.thought_loop import run_thought_loop
from worldpgt.cognition.types import (
    ActionCandidate,
    ActionPlan,
    AllowedConclusion,
    DeliberationTrace,
    EvidenceItem,
    EvidenceRole,
    EvidenceWorkspace,
    MissingEvidence,
    ReasoningProgram,
    ReasoningTrace,
    TaskFrame,
    TaskSubgoal,
    ThoughtStep,
    WorkingMemory,
)
from worldpgt.entity_qa.semantic_speech_planner import SpeechPlan, _join_list


@dataclass
class _ReasoningState:
    task: TaskFrame
    workspace: EvidenceWorkspace
    working_memory: WorkingMemory
    program: ReasoningProgram
    subgoals: tuple[TaskSubgoal, ...]
    plan: SpeechPlan
    steps: list[ThoughtStep] = field(default_factory=list)
    conclusions: list[AllowedConclusion] = field(default_factory=list)
    missing_evidence: list[MissingEvidence] = field(default_factory=list)
    action_candidates: list[ActionCandidate] = field(default_factory=list)
    action: ActionPlan | None = None
    deliberation: DeliberationTrace | None = None
    confidence: str = "grounded"
    support_checks: tuple = ()


def reason_over_plan(plan: SpeechPlan) -> ReasoningTrace:
    """Run small explicit reasoning operations over a speech plan."""

    task = _task_frame(plan)
    workspace = _workspace_from_plan(plan)
    subgoals = _subgoals_for_task(task, workspace)
    working_memory = _working_memory_for_plan(plan, workspace, subgoals)
    program = _program_for_task(task, subgoals)
    state = _ReasoningState(
        task=task,
        workspace=workspace,
        working_memory=working_memory,
        program=program,
        subgoals=subgoals,
        plan=plan,
    )

    for operation in program.operations:
        _execute_operation(operation, state)

    trace = ReasoningTrace(
        task=task,
        workspace=workspace,
        working_memory=working_memory,
        program=program,
        steps=tuple(state.steps),
        allowed_conclusions=tuple(state.conclusions),
        action=state.action or _direct_answer_action(),
        action_candidates=tuple(state.action_candidates),
        deliberation=state.deliberation,
        subgoals=subgoals,
        missing_evidence=tuple(state.missing_evidence),
        support_checks=state.support_checks,
        confidence=state.confidence,
    )
    return replace(trace, thought_loop=run_thought_loop(trace))


def _execute_operation(operation: str, state: _ReasoningState) -> None:
    if operation == "observe":
        state.steps.append(_observe(state.task, state.workspace))
    elif operation == "decompose":
        state.steps.append(_decompose(state.task, state.subgoals))
    elif operation == "gap_check":
        _run_gap_check(state)
    elif operation == "scope_guard":
        _run_scope_guard(state)
    elif operation == "cluster":
        _run_cluster(state)
    elif operation == "choose_action":
        _run_choose_action(state)
    elif operation == "support_guard":
        _run_support_guard(state)
        _run_deliberation(state)


def _run_gap_check(state: _ReasoningState) -> None:
    mechanism_gap = _mechanism_gap(
        state.task,
        state.workspace,
        state.plan,
        state.subgoals,
    )
    if not mechanism_gap:
        state.steps.append(
            ThoughtStep(
                kind="gap_check",
                input=("mechanism",),
                output="mechanism evidence is available",
            )
        )
        return

    missing = MissingEvidence(
        role="mechanism",
        reason="mechanism evidence is missing",
        next_questions=_next_questions_for_gap(state.task, state.workspace),
    )
    state.steps.append(
        ThoughtStep(
            kind="gap_check",
            input=("mechanism",),
            output=missing.reason,
            confidence="gap_heavy",
        )
    )
    state.missing_evidence.append(missing)
    state.conclusions.append(mechanism_gap)
    state.action_candidates.append(
        ActionCandidate(
            action=ActionPlan(
                next_action="answer_with_gap",
                surface_goal="describe supported profile and name the mechanism gap",
                forbidden_claims=("unsupported mechanism",),
                preferred_buckets=("purpose", "classification", "ownership", "activity"),
                suppressed_buckets=("other",),
                detail_unit_limit=2,
                next_questions=missing.next_questions,
            ),
            reason="mechanism subgoal is missing but fallback profile evidence exists",
            priority=90,
        )
    )
    state.confidence = "gap_heavy"


def _run_scope_guard(state: _ReasoningState) -> None:
    if not _is_thin_profile(state.task, state.workspace, state.subgoals):
        state.steps.append(
            ThoughtStep(
                kind="scope_guard",
                input=("profile evidence",),
                output="profile has enough evidence for synthesis",
            )
        )
        return

    missing = MissingEvidence(
        role="activity",
        reason="profile evidence is too thin for synthesis",
        next_questions=(f"What verified facts are known about {state.task.subject}?",),
    )
    state.steps.append(
        ThoughtStep(
            kind="scope_guard",
            input=("no relation buckets",),
            output=missing.reason,
            confidence="thin",
        )
    )
    state.missing_evidence.append(missing)
    state.conclusions.append(
        AllowedConclusion(
            kind="thin_profile",
            text=_thin_profile_sentence(state.plan),
            evidence_roles=("definition",),
            confidence="thin",
        )
    )
    state.action_candidates.append(
        ActionCandidate(
            action=ActionPlan(
                next_action="answer_with_gap",
                surface_goal="answer with definition only and avoid unsupported biography",
                forbidden_claims=("unsupported biography", "unsupported mechanism"),
                suppressed_buckets=("other",),
                detail_unit_limit=0,
                next_questions=missing.next_questions,
            ),
            reason="profile subgoal lacks relation evidence",
            priority=85,
        )
    )
    state.confidence = "thin"


def _run_cluster(state: _ReasoningState) -> None:
    cluster = _profile_cluster(state.workspace, state.plan)
    if not cluster:
        state.steps.append(
            ThoughtStep(
                kind="cluster",
                input=tuple(role for role in state.workspace.roles() if role != "definition"),
                output="no cautious profile summary supported",
                confidence="thin",
            )
        )
        return

    state.steps.append(
        ThoughtStep(
            kind="cluster",
            input=tuple(role for role in state.workspace.roles() if role != "definition"),
            output="built cautious profile summary",
        )
    )
    state.conclusions.append(cluster)
    state.action_candidates.append(
        ActionCandidate(
            action=ActionPlan(
                next_action="answer",
                surface_goal="summarize the profile before listing details",
                forbidden_claims=("unsupported category leap",),
                preferred_buckets=("activity", "origin", "ownership", "recognition"),
                suppressed_buckets=("other",),
                detail_unit_limit=2,
            ),
            reason="profile summary conclusion is available",
            priority=70,
        )
    )


def _run_choose_action(state: _ReasoningState) -> None:
    if state.action_candidates:
        selected = _select_action_candidate(state.action_candidates)
        state.action = selected.action
        state.steps.append(
            ThoughtStep(
                kind="choose_action",
                input=tuple(candidate.reason for candidate in state.action_candidates),
                output=selected.action.surface_goal,
                confidence=state.confidence,
            )
        )
        return

    state.conclusions.append(
        AllowedConclusion(
            kind="direct_answer",
            text="",
            evidence_roles=tuple(role for role in state.workspace.roles() if role != "definition"),
        )
    )
    state.action = _direct_answer_action()
    state.action_candidates.append(
        ActionCandidate(
            action=state.action,
            reason="no specialized action candidate was produced",
            priority=10,
        )
    )
    state.steps.append(
        ThoughtStep(
            kind="choose_action",
            input=tuple(role for role in state.workspace.roles() if role != "definition"),
            output="direct supported answer is enough",
        )
    )


def _select_action_candidate(candidates: list[ActionCandidate]) -> ActionCandidate:
    return sorted(candidates, key=lambda candidate: candidate.priority, reverse=True)[0]


def _run_support_guard(state: _ReasoningState) -> None:
    forbidden = state.action.forbidden_claims if state.action is not None else ()
    checks = validate_trace_conclusions(
        tuple(state.conclusions),
        state.workspace,
        forbidden_claims=forbidden,
    )
    state.support_checks = checks
    blocked = tuple(check for check in checks if not check.ok)
    if blocked:
        state.conclusions = [
            conclusion
            for conclusion, check in zip(state.conclusions, checks, strict=False)
            if check.ok
        ]
        if not state.conclusions:
            state.action = _direct_answer_action()
            state.confidence = "thin"
        state.steps.append(
            ThoughtStep(
                kind="support_guard",
                input=tuple(str(issue) for check in blocked for issue in check.issues),
                output="blocked unsupported conclusion",
                confidence="thin",
            )
        )
        return

    state.steps.append(
        ThoughtStep(
            kind="support_guard",
            input=tuple(check.conclusion_kind for check in checks),
            output="all conclusions supported",
            confidence=state.confidence,
        )
    )


def _run_deliberation(state: _ReasoningState) -> None:
    state.deliberation = deliberate(
        task=state.task,
        workspace=state.workspace,
        subgoals=state.subgoals,
        action_candidates=tuple(state.action_candidates),
        missing_evidence=tuple(state.missing_evidence),
    )


def _direct_answer_action() -> ActionPlan:
    return ActionPlan(
        next_action="answer",
        surface_goal="state supported facts",
        suppressed_buckets=("other",),
    )


def _task_frame(plan: SpeechPlan) -> TaskFrame:
    if plan.answer_style == "followup":
        intent = "followup_answer"
    elif plan.style == "how":
        intent = "mechanism_explanation"
    else:
        intent = "entity_profile"
    return TaskFrame(
        intent=intent,
        subject=plan.subject,
        question_style=plan.style,
        answer_style=plan.answer_style,
    )


def _working_memory_for_plan(
    plan: SpeechPlan,
    workspace: EvidenceWorkspace,
    subgoals: tuple[TaskSubgoal, ...],
) -> WorkingMemory:
    return WorkingMemory(
        current_subject=plan.subject,
        evidence_buckets=tuple(role for role in workspace.roles() if role != "definition"),
        gaps=workspace.gaps,
        active_subgoals=tuple(goal.name for goal in subgoals if goal.status != "satisfied"),
        missing_roles=tuple(
            role
            for goal in subgoals
            for role in goal.missing_roles
        ),
    )


def _workspace_from_plan(plan: SpeechPlan) -> EvidenceWorkspace:
    items: list[EvidenceItem] = []
    if plan.intro:
        items.append(EvidenceItem(role="definition", text=plan.intro))
    for role in (
        "activity",
        "origin",
        "ownership",
        "classification",
        "recognition",
        "mechanism",
        "purpose",
        "other",
    ):
        for value in getattr(plan, role):
            items.append(EvidenceItem(role=role, text=value))
    for value in plan.snapshots:
        items.append(EvidenceItem(role="snapshot", text=value))
    return EvidenceWorkspace(subject=plan.subject, items=tuple(items), gaps=tuple(plan.gaps))


def _observe(task: TaskFrame, workspace: EvidenceWorkspace) -> ThoughtStep:
    return ThoughtStep(
        kind="observe",
        input=(task.subject, task.question_style, task.answer_style),
        output=f"{len(workspace.roles())} evidence role(s) available",
    )


def _decompose(task: TaskFrame, subgoals: tuple[TaskSubgoal, ...]) -> ThoughtStep:
    summary = ", ".join(f"{goal.name}:{goal.status}" for goal in subgoals)
    if task.intent == "mechanism_explanation":
        output = f"mechanism task subgoals -> {summary}"
    elif task.intent == "followup_answer":
        output = f"followup task subgoals -> {summary}"
    else:
        output = f"profile task subgoals -> {summary}"
    return ThoughtStep(kind="decompose", input=(task.intent,), output=output)


def _subgoals_for_task(
    task: TaskFrame,
    workspace: EvidenceWorkspace,
) -> tuple[TaskSubgoal, ...]:
    if task.intent == "mechanism_explanation":
        return (
            _subgoal("identify subject", workspace, ("definition",)),
            _subgoal("explain mechanism", workspace, ("mechanism",)),
            _subgoal(
                "fallback profile",
                workspace,
                ("purpose", "classification", "ownership", "activity"),
                any_role=True,
            ),
        )
    if task.intent == "followup_answer":
        return (
            _subgoal(
                "continue current subject",
                workspace,
                ("definition", "activity", "origin", "ownership", "classification", "recognition"),
                any_role=True,
            ),
        )
    return (
        _subgoal("identify subject", workspace, ("definition",)),
        _subgoal(
            "summarize salient profile",
            workspace,
            ("activity", "origin", "ownership", "classification", "recognition", "purpose", "mechanism"),
            any_role=True,
        ),
        TaskSubgoal(
            name="guard unsupported expansion",
            status="satisfied",
        ),
    )


def _program_for_task(
    task: TaskFrame,
    subgoals: tuple[TaskSubgoal, ...],
) -> ReasoningProgram:
    operations: list[str] = ["observe", "decompose"]
    if task.intent == "mechanism_explanation":
        mechanism = _subgoal_by_name(subgoals, "explain mechanism")
        if mechanism and mechanism.status == "missing":
            operations.append("gap_check")
        else:
            operations.append("cluster")
    elif task.intent == "entity_profile":
        profile = _subgoal_by_name(subgoals, "summarize salient profile")
        if profile and profile.status == "missing":
            operations.append("scope_guard")
        else:
            operations.append("cluster")
    else:
        operations.append("choose_action")
    operations.extend(("choose_action", "support_guard"))
    return ReasoningProgram(operations=tuple(operations))


def _subgoal(
    name: str,
    workspace: EvidenceWorkspace,
    required_roles: tuple[EvidenceRole, ...],
    *,
    any_role: bool = False,
) -> TaskSubgoal:
    present = tuple(role for role in required_roles if workspace.texts_for(role))
    if any_role:
        status = "satisfied" if present else "missing"
        missing = () if present else required_roles
    else:
        missing = tuple(role for role in required_roles if role not in present)
        if not missing:
            status = "satisfied"
        elif present:
            status = "partial"
        else:
            status = "missing"
    return TaskSubgoal(
        name=name,
        required_roles=required_roles,
        satisfied_roles=present,
        missing_roles=missing,
        status=status,
    )


def _mechanism_gap(
    task: TaskFrame,
    workspace: EvidenceWorkspace,
    plan: SpeechPlan,
    subgoals: tuple[TaskSubgoal, ...],
) -> AllowedConclusion | None:
    mechanism_goal = _subgoal_by_name(subgoals, "explain mechanism")
    if (
        task.intent != "mechanism_explanation"
        or mechanism_goal is None
        or mechanism_goal.status != "missing"
        or workspace.texts_for("mechanism")
    ):
        return None
    if workspace.texts_for("purpose"):
        text = (
            "The facts I have describe what it provides, but they still do not "
            "explain the operating mechanism."
        )
        roles: tuple[EvidenceRole, ...] = ("purpose",)
    elif any(workspace.texts_for(role) for role in ("classification", "ownership", "activity")):
        text = (
            "The facts I have identify the profile, but they still do not explain "
            "the operating mechanism."
        )
        roles = tuple(
            role
            for role in ("classification", "ownership", "activity")
            if workspace.texts_for(role)
        )
    else:
        text = "That is enough for a basic profile, but not for a how-it-works explanation."
        roles = ("definition",)
    return AllowedConclusion(
        kind="mechanism_gap",
        text=text,
        evidence_roles=roles,
        confidence="gap_heavy",
    )


def _next_questions_for_gap(
    task: TaskFrame,
    workspace: EvidenceWorkspace,
) -> tuple[str, ...]:
    if task.intent == "mechanism_explanation":
        if workspace.texts_for("purpose"):
            return (
                f"What does {task.subject} use to provide that service?",
                f"What process connects {task.subject} to its users?",
            )
        return (
            f"What mechanism facts are known about {task.subject}?",
            f"What does {task.subject} use or do to work?",
        )
    return ()


def _is_thin_profile(
    task: TaskFrame,
    workspace: EvidenceWorkspace,
    subgoals: tuple[TaskSubgoal, ...],
) -> bool:
    if task.intent == "entity_profile":
        profile_goal = _subgoal_by_name(subgoals, "summarize salient profile")
        if profile_goal is not None:
            return profile_goal.status == "missing"
    meaningful_roles = [
        role
        for role in workspace.roles()
        if role not in {"definition", "snapshot", "gap", "other"}
    ]
    return not meaningful_roles


def _subgoal_by_name(
    subgoals: tuple[TaskSubgoal, ...],
    name: str,
) -> TaskSubgoal | None:
    for subgoal in subgoals:
        if subgoal.name == name:
            return subgoal
    return None


def _profile_cluster(
    workspace: EvidenceWorkspace,
    plan: SpeechPlan,
) -> AllowedConclusion | None:
    if plan.answer_style in {"brief", "followup"}:
        return None
    roles = tuple(
        role
        for role in ("activity", "ownership", "classification", "recognition", "mechanism", "purpose")
        if workspace.texts_for(role)
    )
    if len(roles) < 2:
        return None
    terms = _profile_terms(workspace, plan)
    if len(terms) < 4:
        return None
    return AllowedConclusion(
        kind="profile_summary",
        text=f"The facts I have mostly frame it through {_join_list(terms[:6])}.",
        evidence_roles=roles,
    )


def _profile_terms(workspace: EvidenceWorkspace, plan: SpeechPlan) -> list[str]:
    if plan.style == "how":
        roles: tuple[EvidenceRole, ...] = (
            "mechanism",
            "purpose",
            "classification",
            "ownership",
            "activity",
            "recognition",
        )
    else:
        roles = (
            "activity",
            "ownership",
            "recognition",
            "classification",
            "purpose",
            "mechanism",
        )
    terms: list[str] = []
    for role in roles:
        for value in workspace.texts_for(role):
            terms.extend(_extract_terms(value))
    return _dedupe_terms(terms)


def _extract_terms(text: str) -> list[str]:
    cleaned = str(text)
    for prefix in (
        "develops ",
        "produces ",
        "operates ",
        "publishes ",
        "provides ",
        "enables ",
        "uses ",
        "works by ",
        "is used for ",
        "is owned by ",
        "owns ",
        "is classified as ",
        "is known for ",
    ):
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):]
            break
    cleaned = cleaned.replace(", including ", ", ")
    parts = [p.strip() for p in cleaned.replace(" and ", ", ").split(",")]
    return [p for p in parts if p and not p.endswith("other named items")]


def _dedupe_terms(terms: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        if not _is_profile_term(term):
            continue
        key = term.lower()
        if key not in seen:
            seen.add(key)
            out.append(term)
    return out


def _is_profile_term(term: str) -> bool:
    text = term.strip()
    low = text.lower()
    if not text:
        return False
    if low in {"line"}:
        return False
    if low.startswith(("the ", "a ", "an ")):
        return False
    if len(text) < 3:
        return False
    return True


def _thin_profile_sentence(plan: SpeechPlan) -> str:
    subject = plan.subject or "this entity"
    return (
        f"I only have a thin profile for {subject} here, so I should not add "
        "a fuller biography from unsupported memory."
    )
