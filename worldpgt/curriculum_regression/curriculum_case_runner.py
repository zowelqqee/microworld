"""Case runner for the Curriculum Regression Gate v1.

Routes each proposed question to an EXISTING QA pipeline (entity-QA or
cross-page-QA) and records the observed decision/answer/intent. It reuses the
shipped analyzers, planners, and renderers verbatim — it does not duplicate or
re-implement planner logic, and it never mutates anything.

Routing (conservative, deterministic):
  1. Cross-page QA when the question mentions connection/path/overlay-recheck
     style phrases.
  2. Entity QA otherwise (entity / relation / source / current / private style).
  3. When unsure, entity QA first.
  4. The runner never re-routes an audited question into a second pipeline to
     coerce an answer.
"""

from __future__ import annotations

from pathlib import Path

from worldpgt.cross_page_qa.cross_page_answer_planner import CrossPageAnswerPlanner
from worldpgt.cross_page_qa.cross_page_answer_renderer import render as cross_render
from worldpgt.cross_page_qa.cross_page_question_analyzer import analyze as cross_analyze
from worldpgt.entity_qa.entity_answer_planner import EntityAnswerPlanner
from worldpgt.entity_qa.entity_answer_renderer import render as entity_render
from worldpgt.entity_qa.entity_question_analyzer import analyze as entity_analyze
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

from .types import ROUTE_CROSS_PAGE, ROUTE_ENTITY, ObservedResult

# Phrases that indicate a connection/path/overlay-recheck style question.
_CROSS_PAGE_PHRASES = (
    "connected to",
    "path connects",
    "through the overlay",
    "which claims require rechecking",
    "weak_context_only",
)


def route_for(question: str) -> str:
    """Pick a pipeline for a question. Conservative; entity QA by default."""
    q = (question or "").lower()
    for phrase in _CROSS_PAGE_PHRASES:
        if phrase in q:
            return ROUTE_CROSS_PAGE
    return ROUTE_ENTITY


class CaseRunner:
    """Replays questions through the existing QA pipelines (read-only)."""

    def __init__(self, overlay_json_path: str) -> None:
        provider = WikiMemoryOverlayProvider(overlay_json_path)
        # Each pipeline reads the SAME accepted overlay provider, read-only.
        self._entity_planner = EntityAnswerPlanner(provider=provider)
        self._cross_planner = CrossPageAnswerPlanner(provider=provider)

    def run(self, question: str) -> ObservedResult:
        route = route_for(question)
        if route == ROUTE_CROSS_PAGE:
            return self._run_cross(question)
        return self._run_entity(question)

    def _run_entity(self, question: str) -> ObservedResult:
        analyzed = entity_analyze(question)
        plan = self._entity_planner.plan(analyzed)
        answer = entity_render(plan)
        return ObservedResult(
            decision=plan.decision,
            answer=answer or "",
            intent=getattr(analyzed, "intent", "") or "",
            route=ROUTE_ENTITY,
            audit_reason=getattr(plan, "audit_reason", "") or "",
        )

    def _run_cross(self, question: str) -> ObservedResult:
        q = cross_analyze(question)
        plan = self._cross_planner.plan(q)
        answer = cross_render(plan)
        return ObservedResult(
            decision=plan.decision,
            answer=answer or "",
            intent=getattr(q, "intent", "") or "",
            route=ROUTE_CROSS_PAGE,
            audit_reason=getattr(plan, "audit_reason", "") or "",
        )
