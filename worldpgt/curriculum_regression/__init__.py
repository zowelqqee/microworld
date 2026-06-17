"""Curriculum Regression Gate v1.

Turns the *proposal-only* curriculum and adversarial cases produced by the
Self-Learning Loop v1 into a deterministic safety regression benchmark, run
against the existing entity-QA and cross-page-QA pipelines over the accepted
wiki overlay.

This is a gate, not a mutation layer. It:

- never activates proposed extraction patterns or analyzer rules,
- never writes accepted memory, the accepted overlay, or the promoted overlay,
- never changes runtime behavior, planner thresholds, or validators,
- never adds neural / GPT / embedding / training / network code.

Its sole job is to confirm that dangerous proposed cases still audit/reject and
that missing-knowledge rows audit safely (which is correct current behavior).
"""

from __future__ import annotations

from .types import (
    DANGEROUS_CATEGORIES,
    AdversarialCase,
    AdversarialCaseResult,
    CurriculumCase,
    CurriculumCaseResult,
    ObservedResult,
)

__all__ = [
    "DANGEROUS_CATEGORIES",
    "AdversarialCase",
    "AdversarialCaseResult",
    "CurriculumCase",
    "CurriculumCaseResult",
    "ObservedResult",
]
