"""Mini-thought adapter over the explicit reasoning engine."""

from __future__ import annotations

from worldpgt.cognition.reasoning_engine import reason_over_plan
from worldpgt.cognition.types import MiniThought
from worldpgt.entity_qa.semantic_speech_planner import SpeechPlan


def build_mini_thought(plan: SpeechPlan) -> MiniThought:
    """Project a full reasoning trace into the compact speech-facing shape."""

    trace = reason_over_plan(plan)
    conclusion = trace.primary_conclusion
    interpretation = conclusion.text if conclusion and conclusion.text else None
    return MiniThought(
        task=trace.task,
        working_memory=trace.working_memory,
        steps=trace.steps,
        action=trace.action,
        interpretation=interpretation,
        confidence=trace.confidence,
        trace=trace,
    )
