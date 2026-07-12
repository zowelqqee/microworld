"""Surface composition from cognitive patterns.

The cognitive layer can shape presentation, but it cannot add factual support.
This module only ever appends one natural, actionable line to an
already-supported answer; it never changes the factual claim.
"""

from __future__ import annotations

from worldpgt.assistant_surface.types import FACTUAL_SUPPORT_KINDS


def apply_cognitive_surface(
    question: str,
    answer_text: str,
    *,
    support_kind: str,
    source_system: str,
    cognitive_plan: dict | None,
) -> str:
    """Add at most one natural, useful line from a matched cognitive pattern.

    Never changes the factual claim. Returns the answer unchanged unless a
    pattern gives the reader something concretely actionable to do next
    (e.g. a debugging move or a real follow-up question) — a plain
    informational answer is left exactly as it already reads.
    """

    del question
    text = (answer_text or "").strip()
    if not text or not cognitive_plan:
        return answer_text
    if source_system == "community_context":
        return answer_text
    if support_kind not in FACTUAL_SUPPORT_KINDS:
        return answer_text
    if _already_pattern_shaped(text):
        return answer_text

    patterns = cognitive_plan.get("cognitive_patterns")
    if not isinstance(patterns, list) or not patterns:
        return answer_text
    if cognitive_plan.get("factual_support_allowed_from_patterns") is not False:
        return answer_text
    if any(_pattern_allows_facts(pattern) for pattern in patterns if isinstance(pattern, dict)):
        return answer_text

    kinds = {
        str(pattern.get("kind"))
        for pattern in patterns
        if isinstance(pattern, dict) and pattern.get("kind")
    }
    if not kinds:
        return answer_text

    note = _natural_note(kinds, cognitive_plan)
    if not note:
        return answer_text
    return f"{text}\n\n{note}"


def _already_pattern_shaped(text: str) -> bool:
    lowered = text.lower()
    return (
        lowered.startswith("short answer:")
        or lowered.startswith("short version:")
        or lowered.startswith("short version (live web):")
        or "the cognitive pattern only shapes the explanation" in lowered
    )


def _pattern_allows_facts(pattern: dict) -> bool:
    return pattern.get("factual_support_allowed") is not False


def _natural_note(kinds: set[str], plan: dict) -> str:
    """One plain-language, actionable line — or "" when there is nothing to add."""

    if "debugging_pattern" in kinds:
        return "If this doesn't resolve it, try reducing the problem to the smallest example that still shows it."
    if "uncertainty_pattern" in kinds:
        return "Worth treating as a starting point — this can vary, so it's worth double-checking for your case."
    if "question_pattern" in kinds:
        value = str(plan.get("helpful_next_move") or "").strip()
        if value:
            return value.rstrip(".") + "."
    return ""
