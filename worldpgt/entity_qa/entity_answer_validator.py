"""Answer validator for Entity QA v1.

Checks rendered answers for correctness against expected fields in the CSV row.
Deterministic rule-based. No ML. No network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_MAX_ANSWER_CHARS = 600

_BANNED_PHRASES = (
    "it depends on many things",
    "there are several possibilities",
    "in general",
    "various contexts",
    "something related",
    "many different",
    "contextually speaking",
)

_AUDIT_PREFIX = "i cannot answer from the wiki overlay because"


@dataclass
class ValidationResult:
    is_correct: bool
    quality_flagged: bool
    quality_reason: str


def validate(
    row: dict,
    decision: str,
    answer: str,
    intent: str,
) -> ValidationResult:
    """Check one entity QA output against its expected fields.

    Expected CSV columns:
    - expected_decision: "answer", "audit", or "no"
    - expected_intent: e.g. "define_entity"
    - expected_answer_contains: semicolon-separated keywords (for answer rows)
    """
    quality_flagged = False
    quality_reason = ""

    expected_decision = row.get("expected_decision", "")
    expected_intent = row.get("expected_intent", "")
    expected_contains_raw = row.get("expected_answer_contains", "")

    # Decision must match.
    if decision != expected_decision:
        return ValidationResult(is_correct=False, quality_flagged=False, quality_reason="")

    # For audit rows, intent and decision match is sufficient.
    if decision == "audit":
        if expected_decision == "audit":
            return ValidationResult(is_correct=True, quality_flagged=False, quality_reason="")
        return ValidationResult(is_correct=False, quality_flagged=False, quality_reason="")

    # Answer/no rows — check keyword containment.
    if expected_contains_raw:
        keywords = [k.strip().lower() for k in expected_contains_raw.split(";") if k.strip()]
        answer_lower = answer.lower()
        if not all(k in answer_lower for k in keywords):
            return ValidationResult(is_correct=False, quality_flagged=False, quality_reason="")

    # Quality checks on answer text.
    answer_lower = answer.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in answer_lower:
            quality_flagged = True
            quality_reason = f"banned_phrase:{phrase}"
            break

    if len(answer) > _MAX_ANSWER_CHARS:
        quality_flagged = True
        quality_reason = "answer_too_long"

    if not answer.strip():
        quality_flagged = True
        quality_reason = "empty_answer"

    return ValidationResult(
        is_correct=True,
        quality_flagged=quality_flagged,
        quality_reason=quality_reason,
    )
