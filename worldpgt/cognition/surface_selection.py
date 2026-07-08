"""Selection among corrected answer surface variants."""

from __future__ import annotations

from worldpgt.cognition.self_correction import self_correct_answer
from worldpgt.cognition.types import (
    AnswerSelectionTrace,
    AnswerVariant,
    AnswerVariantEvaluation,
)
from worldpgt.entity_qa.semantic_speech_planner import SpeechPlan


def select_answer_variant(
    variants: tuple[AnswerVariant, ...],
    plan: SpeechPlan,
) -> AnswerSelectionTrace:
    """Self-correct each variant and select the lowest-risk final answer."""

    evaluations = tuple(_evaluate_variant(variant, plan) for variant in variants)
    selected = _select_evaluation(evaluations)
    return AnswerSelectionTrace(
        variants=variants,
        evaluations=evaluations,
        selected_name=selected.variant_name if selected is not None else None,
        final_text=selected.correction.final_text if selected is not None else "",
    )


def _evaluate_variant(
    variant: AnswerVariant,
    plan: SpeechPlan,
) -> AnswerVariantEvaluation:
    correction = self_correct_answer(variant.text, plan)
    score, reason = _score_variant(variant, correction)
    return AnswerVariantEvaluation(
        variant_name=variant.name,
        correction=correction,
        score=score,
        reason=reason,
    )


def _score_variant(variant: AnswerVariant, correction) -> tuple[int, str]:
    score = 0
    reasons: list[str] = []
    for finding in correction.findings:
        if finding.severity == "error":
            score += 100
        elif finding.severity == "warning":
            score += 25
        else:
            score += 5
        reasons.append(finding.code)
    if correction.repairs:
        score += 10 * len(correction.repairs)
        reasons.append("repair_applied")
    if not correction.final_text:
        score += 1000
        reasons.append("empty_final_text")
    style_score, style_reasons = _style_score(correction.final_text or variant.text)
    score += style_score
    reasons.extend(style_reasons)
    return score, ", ".join(reasons) if reasons else "clean"


def _style_score(text: str) -> tuple[int, list[str]]:
    normalized = " ".join(text.lower().split())
    score = 0
    reasons: list[str] = []
    penalties = {
        "current evidence": 30,
        "unsupported memory": 30,
        "i should not": 30,
        "the facts i have": 20,
        "a useful next question": 15,
        "mechanism evidence": 10,
        "i can say": 3,
        "the main things i can say": 3,
        "that is the basic identification": 3,
        "right now i only know": 8,
        "operating mechanism is still missing": 25,
        "still missing here": 20,
    }
    for phrase, penalty in penalties.items():
        if phrase in normalized:
            score += penalty
            reasons.append(f"style:{phrase.replace(' ', '_')}")
    if len(text) > 420:
        score += 5
        reasons.append("style:long")
    return score, reasons


def _select_evaluation(
    evaluations: tuple[AnswerVariantEvaluation, ...],
) -> AnswerVariantEvaluation | None:
    if not evaluations:
        return None
    best = sorted(evaluations, key=lambda evaluation: evaluation.score)[0]
    baseline = next(
        (evaluation for evaluation in evaluations if evaluation.variant_name == "baseline"),
        None,
    )
    if baseline is None or best.variant_name == "baseline":
        return best
    if baseline.score < 100 and (best.score + 2) >= baseline.score:
        return baseline
    return best
