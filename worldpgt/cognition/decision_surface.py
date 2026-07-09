"""Human-facing phrasing for explicit thought-loop decisions.

The reasoning layer can keep internal guardrail language such as evidence,
support, and missing roles. This module turns those decisions into ordinary
speech without adding facts.
"""

from __future__ import annotations

from worldpgt.cognition.types import ActionPlan, AllowedConclusion, Confidence
from worldpgt.entity_qa.semantic_speech_planner import SpeechPlan


def decision_intro_sentence(
    plan: SpeechPlan,
    action: ActionPlan,
    conclusion: AllowedConclusion | None,
    confidence: Confidence,
) -> str:
    """Return a human-facing sentence for a selected reasoning decision."""

    candidates = decision_intro_candidates(plan, action, conclusion, confidence)
    return candidates[0] if candidates else ""


def decision_intro_candidates(
    plan: SpeechPlan,
    action: ActionPlan,
    conclusion: AllowedConclusion | None,
    confidence: Confidence,
) -> tuple[str, ...]:
    """Return safe candidate phrasings for a selected reasoning decision."""

    if action.next_action == "ask_clarification":
        return clarification_candidates(plan, action)
    if conclusion is None:
        return ()
    if conclusion.kind == "mechanism_gap":
        return _mechanism_gap_candidates(plan)
    if conclusion.kind == "thin_profile" or confidence == "thin":
        return _thin_profile_candidates(plan)
    if conclusion.kind == "profile_summary":
        return _profile_summary_candidates(plan)
    return (conclusion.text,)


def decision_next_question_sentence(action: ActionPlan) -> str:
    """Phrase the selected follow-up question without debug-style wording."""

    candidates = decision_next_question_candidates(action)
    return candidates[0] if candidates else ""


def decision_next_question_candidates(action: ActionPlan) -> tuple[str, ...]:
    """Return safe candidate phrasings for the selected follow-up question."""

    if not action.next_questions:
        return ()
    question = action.next_questions[0]
    return (
        f"The missing piece is: {question}",
        f"To explain it properly, I would need one more supported fact: {question}",
        f"What would help next is this: {question}",
    )


def clarification_sentence(plan: SpeechPlan, action: ActionPlan) -> str:
    """Phrase a clarification request selected by the thought loop."""

    candidates = clarification_candidates(plan, action)
    return candidates[0] if candidates else ""


def clarification_candidates(plan: SpeechPlan, action: ActionPlan) -> tuple[str, ...]:
    """Return safe clarification phrasings selected by the thought loop."""

    if action.next_questions:
        question = action.next_questions[0]
        if plan.subject:
            return (
                f"To answer that properly, I need one more piece: {question}",
                f"I need one more piece before answering that: {question}",
                question,
            )
        return (question,)
    if plan.subject:
        return (
            f"What should I focus on about {plan.subject}?",
            f"Which part of {plan.subject} should I answer about?",
        )
    return ("What should I focus on?",)


def _mechanism_gap_candidates(plan: SpeechPlan) -> tuple[str, ...]:
    subject = plan.subject or "it"
    if plan.purpose:
        return (
            f"Here is the honest version: I know what {subject} is and what "
            "service it provides, but I do not yet have the mechanism: the "
            "parts and steps that make it work.",
            (
                f"I can identify {subject} and describe the service it provides, "
                "but I do not yet have the mechanism: the parts and steps that "
                "make it work."
            ),
            (
                f"I know the basic shape of {subject}, but not the working "
                "details yet."
            ),
        )
    return (
        f"Here is the honest version: I can identify {subject}, but I do not "
        "yet have the mechanism: the parts and steps that make it work.",
        f"I know what {subject} refers to, but not the working mechanism yet.",
    )


def _thin_profile_candidates(plan: SpeechPlan) -> tuple[str, ...]:
    subject = plan.subject or "it"
    if plan.intro:
        return (
            f"That is the reliable part I have for {subject} right now.",
            f"Right now I can identify {subject}, but I do not have much more yet.",
            f"I can identify {subject}, but I do not have much more yet.",
        )
    return (
        f"I only have a very small amount to go on for {subject} right now.",
        f"I do not have much to say about {subject} yet.",
    )


def _profile_summary_candidates(plan: SpeechPlan) -> tuple[str, ...]:
    subject = plan.subject or "it"
    anchors: list[str] = []
    if plan.activity:
        anchors.append(plan.activity[0])
    if plan.origin:
        founders = [item for item in plan.origin if not item.startswith("founded ")]
        if founders:
            anchors.append(f"its founding by {', '.join(founders[:2])}")
    if not anchors:
        return ()
    return (
        f"Here are the supported pieces I can put together for {subject}.",
        f"For {subject}, I have a usable basic profile, not a complete story yet.",
    )
