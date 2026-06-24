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
from worldpgt.cognition.deliberation_engine import deliberate
from worldpgt.cognition.decision_surface import (
    clarification_sentence,
    decision_intro_sentence,
    decision_next_question_sentence,
)
from worldpgt.cognition.reasoning_engine import reason_over_plan
from worldpgt.cognition.self_correction import self_correct_answer
from worldpgt.cognition.session_engine import CognitiveSession
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
    "reason_over_plan",
    "run_thought_loop",
    "self_correct_answer",
    "select_answer_variant",
]
