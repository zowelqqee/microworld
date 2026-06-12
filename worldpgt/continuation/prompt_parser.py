"""Deterministic prompt parsing: detect the strongest known ambiguous term in a prompt."""

from __future__ import annotations

from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.continuation.types import ContinuationPrompt


def parse_continuation_prompt(prompt: str, memory: ExplicitSenseMemory) -> ContinuationPrompt:
    """Pick the known term with the highest max sense score; ties go to first appearance."""
    terms = memory.find_ambiguous_terms(prompt)
    if not terms:
        return ContinuationPrompt(prompt=prompt, ambiguous_term=None, candidate_senses=[])

    best_term = None
    best_score = -1.0
    for term in terms:  # appearance order, so strict > keeps the first term on ties
        scores = memory.score_senses(prompt, term)
        top = max(scores.values()) if scores else 0.0
        if top > best_score:
            best_term = term
            best_score = top

    candidate_senses = [entry.sense_id for entry in memory.get_senses(best_term)]
    return ContinuationPrompt(
        prompt=prompt,
        ambiguous_term=best_term,
        candidate_senses=candidate_senses,
    )
