"""Explicit cognition primitives for bounded Microworld reasoning."""

from typing import TYPE_CHECKING

from worldpgt.cognition.types import (
    ActionPlan,
    ActionCandidate,
    AnswerDraft,
    AnswerSelectionTrace,
    AnswerVariant,
    AnswerVariantEvaluation,
    AllowedConclusion,
    CandidateEvaluation,
    ConclusionSupportCheck,
    CognitiveTaskPlan,
    CritiqueFinding,
    DeliberationTrace,
    EvidenceItem,
    EvidenceWorkspace,
    MemoryEvent,
    MissingEvidence,
    MiniThought,
    ReasoningProgram,
    ReasoningTrace,
    RepairAction,
    SessionTraceRecord,
    SessionTurnPlan,
    SelfCorrectionTrace,
    TaskFrame,
    TaskMemorySnapshot,
    TaskPlanStep,
    TaskSubgoal,
    ThoughtHypothesis,
    ThoughtCandidate,
    ThoughtLoopDecision,
    ThoughtLoopStep,
    ThoughtLoopTrace,
    ThoughtStep,
    WorkingTask,
    WorkingMemory,
)
from worldpgt.cognition.cognitive_graph_loop import (
    AnswerPlan,
    CognitiveGraphEdge,
    CognitiveGraphNode,
    CognitiveLoopTrace,
    CognitiveMove,
    LoopIteration,
    MoveApplication,
    MoveCandidate,
    RejectedMove,
    WorkingSemanticState,
    action_plan_from_cognitive_loop,
    render_plan_addendum,
    run_cognitive_graph_loop,
)
from worldpgt.cognition.deliberation_engine import deliberate
from worldpgt.cognition.decision_surface import (
    clarification_sentence,
    decision_intro_sentence,
    decision_next_question_sentence,
)
from worldpgt.cognition.reasoning_engine import reason_over_plan
from worldpgt.cognition.self_correction import self_correct_answer
from worldpgt.cognition.session_engine import CognitiveSession
from worldpgt.cognition.semantic_thought_graph import (
    SemanticThoughtEdge,
    SemanticThoughtGraphTrace,
    SemanticThoughtMove,
    SemanticThoughtNode,
    run_semantic_thought_graph,
)
from worldpgt.cognition.surface_selection import select_answer_variant
from worldpgt.cognition.task_planner import build_task_plan
from worldpgt.cognition.working_memory import TaskMemory
from worldpgt.cognition.thought_loop import run_thought_loop

if TYPE_CHECKING:
    from worldpgt.cognition.answer_session import CognitiveAnswerResult, CognitiveAnswerSession


def __getattr__(name: str):
    if name in {"CognitiveAnswerResult", "CognitiveAnswerSession"}:
        from worldpgt.cognition.answer_session import (
            CognitiveAnswerResult,
            CognitiveAnswerSession,
        )

        return {
            "CognitiveAnswerResult": CognitiveAnswerResult,
            "CognitiveAnswerSession": CognitiveAnswerSession,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "ActionPlan",
    "ActionCandidate",
    "AnswerDraft",
    "AnswerPlan",
    "CognitiveGraphEdge",
    "CognitiveGraphNode",
    "CognitiveLoopTrace",
    "CognitiveMove",
    "LoopIteration",
    "MoveApplication",
    "MoveCandidate",
    "RejectedMove",
    "WorkingSemanticState",
    "AnswerSelectionTrace",
    "AnswerVariant",
    "AnswerVariantEvaluation",
    "AllowedConclusion",
    "CandidateEvaluation",
    "ConclusionSupportCheck",
    "CognitiveAnswerResult",
    "CognitiveAnswerSession",
    "CognitiveSession",
    "CognitiveTaskPlan",
    "CritiqueFinding",
    "DeliberationTrace",
    "EvidenceItem",
    "EvidenceWorkspace",
    "MemoryEvent",
    "MissingEvidence",
    "MiniThought",
    "ReasoningProgram",
    "ReasoningTrace",
    "RepairAction",
    "SessionTraceRecord",
    "SessionTurnPlan",
    "SelfCorrectionTrace",
    "SemanticThoughtEdge",
    "SemanticThoughtGraphTrace",
    "SemanticThoughtMove",
    "SemanticThoughtNode",
    "TaskFrame",
    "TaskMemory",
    "TaskMemorySnapshot",
    "TaskPlanStep",
    "TaskSubgoal",
    "ThoughtHypothesis",
    "ThoughtCandidate",
    "ThoughtLoopDecision",
    "ThoughtLoopStep",
    "ThoughtLoopTrace",
    "ThoughtStep",
    "WorkingTask",
    "WorkingMemory",
    "build_task_plan",
    "clarification_sentence",
    "decision_intro_sentence",
    "decision_next_question_sentence",
    "deliberate",
    "action_plan_from_cognitive_loop",
    "reason_over_plan",
    "render_plan_addendum",
    "run_cognitive_graph_loop",
    "run_thought_loop",
    "run_semantic_thought_graph",
    "self_correct_answer",
    "select_answer_variant",
]
