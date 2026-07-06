"""Surface composition from cognitive patterns.

The cognitive layer can shape presentation, but it cannot add factual support.
This module only rewrites already-supported answers into a clearer answer
shape and keeps the factual source label explicit.
"""

from __future__ import annotations

from worldpgt.assistant_surface.types import FACTUAL_SUPPORT_KINDS


_SOURCE_LABELS = {
    "entity_qa": "Microworld memory",
    "cross_page_qa": "Microworld relation memory",
    "query_engine": "Microworld query memory",
    "web_search": "live web search",
    "context_pack": "Microworld policy context",
}


def apply_cognitive_surface(
    question: str,
    answer_text: str,
    *,
    support_kind: str,
    source_system: str,
    cognitive_plan: dict | None,
) -> str:
    """Apply a pattern-guided surface shape without changing factual support."""

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

    source_label = _SOURCE_LABELS.get(source_system or "", source_system or "the supported source")
    framing = _framing_line(kinds)
    next_check = _next_check(cognitive_plan)
    return "\n".join(
        [
            "Short answer:",
            text,
            "",
            "How I am framing it:",
            f"- {framing}",
            f"- The factual claim above still comes from {source_label}.",
            "- The cognitive pattern only shapes the explanation; it is not extra evidence.",
            f"- {next_check}",
        ]
    )


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


def _framing_line(kinds: set[str]) -> str:
    if "debugging_pattern" in kinds:
        return "Use the answer as the known part, then reduce the remaining problem to a small reproducible check."
    if "explanation_pattern" in kinds:
        return "Start with the supported claim, then keep the explanation plain and concrete."
    if "uncertainty_pattern" in kinds:
        return "Keep the caveat visible before turning the answer into a broader conclusion."
    if "question_pattern" in kinds:
        return "If this does not resolve the question, the next useful move is a narrower follow-up."
    if "style_tone_pattern" in kinds:
        return "Prefer clear phrasing over clever phrasing."
    return "Use the selected pattern as answer shape, not as factual evidence."


def _next_check(plan: dict) -> str:
    value = str(plan.get("helpful_next_move") or "").strip()
    if value:
        return value.rstrip(".") + "."
    return "Before extending this answer, check the missing or current facts in a source-backed layer."
