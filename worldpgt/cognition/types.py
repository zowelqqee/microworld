"""Small explicit cognition types.

These types are deliberately language-first. Code writing can later use the
same task-frame / thought-step / action-plan shape, but the initial runtime
surface is bounded QA text.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


TaskIntent = Literal[
    "entity_profile",
    "mechanism_explanation",
    "followup_answer",
    "clarification",
]

SessionIntent = Literal[
    "new_task",
    "explain_missing_evidence",
    "explain_task_plan",
    "suggest_next_question",
    "continue_task",
    "unresolved_followup",
]

ThoughtKind = Literal[
    "observe",
    "decompose",
    "gap_check",
    "cluster",
    "support_guard",
    "scope_guard",
    "choose_action",
]

Confidence = Literal["grounded", "thin", "gap_heavy"]

NextAction = Literal["answer", "answer_with_gap", "ask_clarification"]

SubgoalStatus = Literal["satisfied", "partial", "missing"]

TaskPlanStepStatus = Literal["satisfied", "actionable", "blocked"]

DeliberationDecision = Literal["accepted", "rejected"]

CritiqueSeverity = Literal["info", "warning", "error"]

RepairKind = Literal[
    "none",
    "replace_unsupported_mechanism",
    "dedupe_sentence",
]

WorkingTaskStatus = Literal["active", "parked"]

MemoryEventKind = Literal[
    "active_task_updated",
    "task_parked",
    "task_resumed",
    "task_inspected",
    "memory_summarized",
    "session_plan_used",
    "answer_without_trace",
]

HypothesisStatus = Literal["pending", "accepted", "rejected"]

ThoughtLoopOperationKind = Literal[
    "inspect_evidence",
    "form_hypothesis",
    "test_hypothesis",
    "identify_gap",
    "select_next_operation",
    "stop",
]

ThoughtLoopStopReason = Literal[
    "supported_conclusion",
    "blocked_missing_evidence",
    "max_steps",
]

EvidenceRole = Literal[
    "definition",
    "activity",
    "origin",
    "ownership",
    "classification",
    "recognition",
    "mechanism",
    "purpose",
    "other",
    "snapshot",
    "gap",
]

ConclusionKind = Literal[
    "profile_summary",
    "mechanism_gap",
    "thin_profile",
    "direct_answer",
]


@dataclass(frozen=True)
class TaskFrame:
    """What the user appears to be asking for in the current turn."""

    intent: TaskIntent
    subject: str
    question_style: str
    answer_style: str = "normal"


@dataclass(frozen=True)
class WorkingMemory:
    """Minimal explicit state available to a small reasoning step."""

    current_subject: str
    spoken_units: tuple[str, ...] = ()
    evidence_buckets: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    active_subgoals: tuple[str, ...] = ()
    missing_roles: tuple[EvidenceRole, ...] = ()


@dataclass(frozen=True)
class EvidenceItem:
    """One explicit piece of evidence admitted into reasoning."""

    role: EvidenceRole
    text: str
    source: Literal["speech_plan"] = "speech_plan"


@dataclass(frozen=True)
class EvidenceWorkspace:
    """The bounded evidence field available to reasoning operations."""

    subject: str
    items: tuple[EvidenceItem, ...] = ()
    gaps: tuple[str, ...] = ()

    def roles(self) -> tuple[EvidenceRole, ...]:
        seen: list[EvidenceRole] = []
        for item in self.items:
            if item.role not in seen:
                seen.append(item.role)
        return tuple(seen)

    def texts_for(self, role: EvidenceRole) -> tuple[str, ...]:
        return tuple(item.text for item in self.items if item.role == role)


@dataclass(frozen=True)
class ThoughtStep:
    """One inspectable step in bounded reasoning."""

    kind: ThoughtKind
    input: tuple[str, ...] = ()
    output: str = ""
    confidence: Confidence = "grounded"


@dataclass(frozen=True)
class ReasoningProgram:
    """The operation sequence selected for a bounded task."""

    operations: tuple[ThoughtKind, ...]


@dataclass(frozen=True)
class TaskSubgoal:
    """One explicit subgoal needed to complete a task."""

    name: str
    required_roles: tuple[EvidenceRole, ...] = ()
    satisfied_roles: tuple[EvidenceRole, ...] = ()
    missing_roles: tuple[EvidenceRole, ...] = ()
    status: SubgoalStatus = "missing"


@dataclass(frozen=True)
class ActionPlan:
    """The next language action chosen by the mini reasoning layer."""

    next_action: NextAction
    surface_goal: str
    forbidden_claims: tuple[str, ...] = ()
    preferred_buckets: tuple[str, ...] = ()
    suppressed_buckets: tuple[str, ...] = ()
    detail_unit_limit: int | None = None
    next_questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionCandidate:
    """A possible next action proposed by a reasoning operation."""

    action: ActionPlan
    reason: str
    priority: int = 0


@dataclass(frozen=True)
class ThoughtCandidate:
    """One possible reasoning move considered before selecting an action."""

    name: str
    action: ActionPlan
    claim: str = ""
    required_roles: tuple[EvidenceRole, ...] = ()
    priority: int = 0


@dataclass(frozen=True)
class CandidateEvaluation:
    """Evaluation of one thought candidate against available evidence."""

    candidate_name: str
    decision: DeliberationDecision
    reason: str
    missing_roles: tuple[EvidenceRole, ...] = ()


@dataclass(frozen=True)
class DeliberationTrace:
    """The accepted and rejected thought candidates for one reasoning pass."""

    candidates: tuple[ThoughtCandidate, ...] = ()
    evaluations: tuple[CandidateEvaluation, ...] = ()
    selected_name: str | None = None

    def evaluation_for(self, candidate_name: str) -> CandidateEvaluation | None:
        for evaluation in self.evaluations:
            if evaluation.candidate_name == candidate_name:
                return evaluation
        return None


@dataclass(frozen=True)
class AllowedConclusion:
    """A cautious conclusion allowed by the evidence workspace."""

    kind: ConclusionKind
    text: str
    evidence_roles: tuple[EvidenceRole, ...] = ()
    confidence: Confidence = "grounded"


@dataclass(frozen=True)
class MissingEvidence:
    """Evidence the task needs but the current workspace does not contain."""

    role: EvidenceRole
    reason: str
    next_questions: tuple[str, ...] = ()


@dataclass(frozen=True)
class TaskPlanStep:
    """One explicit step in a continuation plan."""

    title: str
    status: TaskPlanStepStatus
    evidence_role: EvidenceRole | None = None
    reason: str = ""
    next_question: str | None = None


@dataclass(frozen=True)
class CognitiveTaskPlan:
    """A task-level plan derived from one reasoning trace."""

    subject: str
    goal: str
    steps: tuple[TaskPlanStep, ...] = ()
    blocked: bool = False
    next_questions: tuple[str, ...] = ()
    answer_text: str = ""


@dataclass(frozen=True)
class AnswerDraft:
    """One generated answer before self-correction."""

    text: str
    source: Literal["symbolic_text_generator"] = "symbolic_text_generator"


@dataclass(frozen=True)
class CritiqueFinding:
    """A critique finding over a generated answer draft."""

    code: str
    message: str
    severity: CritiqueSeverity = "warning"


@dataclass(frozen=True)
class RepairAction:
    """A surface repair applied after critique."""

    kind: RepairKind
    reason: str


@dataclass(frozen=True)
class SelfCorrectionTrace:
    """Draft, critique, repair, and final text for one answer."""

    draft: AnswerDraft
    findings: tuple[CritiqueFinding, ...] = ()
    repairs: tuple[RepairAction, ...] = ()
    final_text: str = ""


@dataclass(frozen=True)
class AnswerVariant:
    """One candidate surface form for the same supported answer."""

    name: str
    text: str


@dataclass(frozen=True)
class AnswerVariantEvaluation:
    """Self-correction and score for one answer variant."""

    variant_name: str
    correction: SelfCorrectionTrace
    score: int
    reason: str


@dataclass(frozen=True)
class AnswerSelectionTrace:
    """Candidate surface forms, their evaluations, and the selected answer."""

    variants: tuple[AnswerVariant, ...] = ()
    evaluations: tuple[AnswerVariantEvaluation, ...] = ()
    selected_name: str | None = None
    final_text: str = ""

    def evaluation_for(self, variant_name: str) -> AnswerVariantEvaluation | None:
        for evaluation in self.evaluations:
            if evaluation.variant_name == variant_name:
                return evaluation
        return None


@dataclass(frozen=True)
class WorkingTask:
    """One task remembered inside a transient cognitive answer session."""

    subject: str
    goal: str
    status: WorkingTaskStatus = "active"
    turn_count: int = 1
    active_subgoals: tuple[str, ...] = ()
    missing_roles: tuple[EvidenceRole, ...] = ()
    next_questions: tuple[str, ...] = ()
    selected_candidate: str | None = None
    rejected_candidates: tuple[str, ...] = ()
    selected_surface_variant: str | None = None
    repair_kinds: tuple[RepairKind, ...] = ()
    last_answer_text: str = ""


@dataclass(frozen=True)
class MemoryEvent:
    """One event in transient task working memory."""

    index: int
    kind: MemoryEventKind
    subject: str | None = None
    detail: str = ""


@dataclass(frozen=True)
class ThoughtHypothesis:
    """One hypothesis considered inside an iterative thought loop."""

    name: str
    text: str
    required_roles: tuple[EvidenceRole, ...] = ()
    status: HypothesisStatus = "pending"
    reason: str = ""


@dataclass(frozen=True)
class ThoughtLoopStep:
    """One operation executed by the iterative thought loop."""

    index: int
    operation: ThoughtLoopOperationKind
    output: str


@dataclass(frozen=True)
class ThoughtLoopDecision:
    """The next action selected by an iterative thought loop."""

    action: ActionPlan
    selected_hypothesis: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class ThoughtLoopTrace:
    """An iterative reasoning loop over an existing reasoning trace."""

    subject: str
    goal: str
    steps: tuple[ThoughtLoopStep, ...] = ()
    hypotheses: tuple[ThoughtHypothesis, ...] = ()
    accepted_hypothesis: ThoughtHypothesis | None = None
    rejected_hypotheses: tuple[ThoughtHypothesis, ...] = ()
    stop_reason: ThoughtLoopStopReason = "max_steps"
    next_questions: tuple[str, ...] = ()
    decision: ThoughtLoopDecision | None = None


@dataclass(frozen=True)
class TaskMemorySnapshot:
    """Read-only view of task working memory."""

    active_task: WorkingTask | None = None
    parked_tasks: tuple[WorkingTask, ...] = ()
    events: tuple[MemoryEvent, ...] = ()


@dataclass(frozen=True)
class ConclusionSupportCheck:
    """Verification result for one allowed conclusion."""

    conclusion_kind: ConclusionKind
    ok: bool
    issues: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReasoningTrace:
    """Full output of the explicit reasoning engine."""

    task: TaskFrame
    workspace: EvidenceWorkspace
    working_memory: WorkingMemory
    program: ReasoningProgram
    steps: tuple[ThoughtStep, ...]
    allowed_conclusions: tuple[AllowedConclusion, ...]
    action: ActionPlan
    action_candidates: tuple[ActionCandidate, ...] = ()
    deliberation: DeliberationTrace | None = None
    subgoals: tuple[TaskSubgoal, ...] = ()
    missing_evidence: tuple[MissingEvidence, ...] = ()
    support_checks: tuple[ConclusionSupportCheck, ...] = ()
    confidence: Confidence = "grounded"
    thought_loop: ThoughtLoopTrace | None = None

    @property
    def primary_conclusion(self) -> AllowedConclusion | None:
        return self.allowed_conclusions[0] if self.allowed_conclusions else None


@dataclass(frozen=True)
class MiniThought:
    """A small bounded interpretation over already selected evidence."""

    task: TaskFrame
    working_memory: WorkingMemory
    steps: tuple[ThoughtStep, ...] = field(default_factory=tuple)
    action: ActionPlan = field(
        default_factory=lambda: ActionPlan(
            next_action="answer",
            surface_goal="state supported facts",
        )
    )
    interpretation: str | None = None
    confidence: Confidence = "grounded"
    trace: ReasoningTrace | None = None

    @property
    def surface_sentence(self) -> str:
        return self.interpretation or ""


@dataclass(frozen=True)
class SessionTurnPlan:
    """A session-level interpretation of the next user turn."""

    question: str
    intent: SessionIntent
    subject: str | None = None
    active_subgoals: tuple[str, ...] = ()
    missing_evidence: tuple[MissingEvidence, ...] = ()
    next_questions: tuple[str, ...] = ()
    task_plan: CognitiveTaskPlan | None = None
    answer_text: str = ""
    needs_new_reasoning: bool = False


@dataclass(frozen=True)
class SessionTraceRecord:
    """One reasoning trace recorded in a cognitive session."""

    question: str
    trace: ReasoningTrace
