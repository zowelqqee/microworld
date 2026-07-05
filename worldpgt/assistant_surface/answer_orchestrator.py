"""Answer orchestrator for the Assistant Surface v1.

Ties together the deterministic router, the context selector, and the existing
QA planners (Entity QA, Cross-page QA) to produce an :class:`AssistantAnswer`.

Priority order (safest first):

1. Hard safety route — current/live, private/sensitive, relation inversion,
   unsupported universal -> always audit.
2. Cross-page QA — connection / path questions.
3. Entity QA — definitions, relation lookup, source-qualified fact lookup,
   weak-link explanations.
4. Context-pack-backed safe policy answers (never new factual claims).
5. Unknown -> audit with a missing-knowledge explanation.

The orchestrator NEVER weakens safety to answer more often, NEVER produces an
unsupported factual answer, and NEVER lets weak-only or volatile context support
a stable claim. It does not modify any planner, validator, threshold, overlay,
or trusted memory. No network, no ML.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from worldpgt.assistant_surface.context_selector import ContextSelector, resolve_overlay
from worldpgt.assistant_surface.answer_style import resolve_answer_style
from worldpgt.assistant_surface.perf_timing import step as _timed_step
from worldpgt.assistant_surface.question_router import route as route_question
from worldpgt.assistant_surface.assistant_trace import (
    attach_context,
    finalize,
    new_trace,
)
from worldpgt.assistant_surface.types import (
    AssistantAnswer,
    AssistantContextSummary,
    AssistantRoute,
    OVERLAY_MODE_CUSTOM_PATH,
)
from worldpgt.cross_page_qa.cross_page_answer_planner import CrossPageAnswerPlanner
from worldpgt.cross_page_qa.cross_page_answer_renderer import render as render_cross_page
from worldpgt.cross_page_qa.cross_page_question_analyzer import analyze as analyze_cross_page
from worldpgt.entity_qa.entity_answer_planner import EntityAnswerPlanner
from worldpgt.entity_qa.entity_answer_renderer import render as render_entity
from worldpgt.entity_qa.entity_question_analyzer import analyze as analyze_entity
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.query_engine import executor as qe_executor
from worldpgt.query_engine import plan_builder as qe_plan_builder
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.assistant_surface.web_search import WebSearchProvider, render_web_answer

# Static, deterministic policy explanations (no factual claims, no overlay data).
_WEAK_LINK_POLICY_TEXT = (
    "A weak context link means two pages mention each other in Microworld's "
    "memory, but the overlay does NOT treat that as a verified fact. A weak link "
    "is contextual association only — it never proves ownership, founding, or any "
    "directional relation."
)
_SOURCE_QUALIFIED_POLICY_TEXT = (
    "A source-qualified fact is a claim the overlay only holds with an explicit "
    "source and as-of date (for example, a Forbes estimate). It is treated as "
    "volatile and may require rechecking — it is never a permanent or current "
    "value."
)

_EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
DEFAULT_WIKIDATA_P279_ONTOLOGY_LAYER_PATH = (
    _EXPERIMENTS
    / "knowledge_pump_v1"
    / "wikidata_p279_ontology_v1"
    / "wikidata_p279_ontology_layer.json"
)


def _load_ontology_layer(path: str | Path | None) -> list[dict]:
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


@lru_cache(maxsize=16)
def _cached_ontology_layer(path_str: str, mtime_ns: int, size: int) -> tuple[dict, ...]:
    del mtime_ns, size
    return tuple(_load_ontology_layer(path_str))


def _load_ontology_layer_cached(path: str | Path | None) -> list[dict]:
    if path is None:
        return []
    p = Path(path)
    if not p.is_file():
        return []
    stat = p.stat()
    return list(_cached_ontology_layer(str(p), stat.st_mtime_ns, stat.st_size))


@lru_cache(maxsize=16)
def _cached_surface_index(
    promoted_overlay_path: str,
    promoted_mtime_ns: int,
    promoted_size: int,
    accepted_mtime_ns: int,
    accepted_size: int,
    snapshot_mtime_ns: int,
    snapshot_size: int,
) -> EntitySurfaceIndex:
    del (
        promoted_mtime_ns,
        promoted_size,
        accepted_mtime_ns,
        accepted_size,
        snapshot_mtime_ns,
        snapshot_size,
    )
    return EntitySurfaceIndex(
        accepted_overlay_path=_EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json",
        promoted_overlay_path=Path(promoted_overlay_path),
        snapshot_overlay_path=_EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
    )


def _surface_index_for_overlay(promoted_overlay_path: str | Path) -> EntitySurfaceIndex:
    promoted = Path(promoted_overlay_path)
    accepted = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
    snapshot = _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
    promoted_stat = promoted.stat()
    accepted_stat = accepted.stat()
    snapshot_stat = snapshot.stat()
    return _cached_surface_index(
        str(promoted),
        promoted_stat.st_mtime_ns,
        promoted_stat.st_size,
        accepted_stat.st_mtime_ns,
        accepted_stat.st_size,
        snapshot_stat.st_mtime_ns,
        snapshot_stat.st_size,
    )


class AnswerOrchestrator:
    def __init__(
        self,
        overlay_mode: str = "promoted",
        overlay_path: str | None = None,
        ontology_layer_path: str | None = None,
        inference_workspace=None,
        web_search_provider: WebSearchProvider | None = None,
        web_search_enabled: bool | None = None,
    ) -> None:
        self.overlay_mode = OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode
        overlay_path_resolved = overlay_path
        if overlay_path_resolved is None:
            overlay_path_resolved, _ = resolve_overlay(overlay_mode)
        self._provider = WikiMemoryOverlayProvider(overlay_path_resolved)
        self._surface_index = _surface_index_for_overlay(overlay_path_resolved)
        self.ontology_layer_path = ontology_layer_path
        if self.ontology_layer_path is None and DEFAULT_WIKIDATA_P279_ONTOLOGY_LAYER_PATH.is_file():
            self.ontology_layer_path = str(DEFAULT_WIKIDATA_P279_ONTOLOGY_LAYER_PATH)
        ontology_layer_items = _load_ontology_layer_cached(self.ontology_layer_path)
        self._ontology_layer_items = ontology_layer_items
        self._inference_workspace = inference_workspace
        if web_search_enabled is None:
            web_search_enabled = os.environ.get("MICROWORLD_WEB_SEARCH_ENABLED", "").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        self._web_search_enabled_default = web_search_enabled
        self._web_search_provider = web_search_provider
        self._entity_planner = EntityAnswerPlanner(
            provider=self._provider,
            ontology_layer_items=ontology_layer_items,
            inference_workspace=inference_workspace,
        )
        self._cross_page_planner = CrossPageAnswerPlanner(provider=self._provider)
        self._selector = ContextSelector(overlay_mode, overlay_path=overlay_path)

    _QE_ONLY_HINTS: frozenset[str] = frozenset({
        "count",
        "size_compare",
        "filtered_lookup",
    })

    # ------------------------------------------------------------------ #
    def answer(
        self,
        question: str,
        *,
        answer_style: str = "normal",
        web_search_enabled: bool | None = None,
    ) -> AssistantAnswer:
        with _timed_step("orchestrator.answer_style"):
            style_resolution = resolve_answer_style(question)
            if style_resolution.answer_style != "normal":
                answer_style = style_resolution.answer_style
                question = style_resolution.question

        with _timed_step("orchestrator.route_question"):
            route = route_question(question, self._surface_index)
        trace = new_trace(route)
        with _timed_step("orchestrator.context_select"):
            _pack, ctx = self._selector.select(question)
        attach_context(trace, ctx)

        # 1. Hard safety route — always audit.
        if route.is_hard_safety:
            with _timed_step("orchestrator.branch:hard_safety_audit"):
                if route.intent == "current_live_request":
                    answer = self._web_search_current_answer(
                        route,
                        ctx,
                        trace,
                        web_search_enabled=web_search_enabled,
                    )
                    if answer is not None:
                        return answer
                return self._hard_safety_audit(route, ctx, trace)

        # 2. Connection / path questions -> cross-page QA.
        if route.intent == "connection_path":
            with _timed_step("orchestrator.branch:connection_answer"):
                return self._connection_answer(route, ctx, trace)

        # 3a. Source-qualified facts -> entity QA.
        if route.intent == "source_qualified_fact":
            with _timed_step("orchestrator.branch:source_fact_answer"):
                return self._source_fact_answer(route, ctx, trace)

        # 3b. Weak-link / policy explanations -> safe policy answer.
        if route.intent == "weak_link_policy":
            with _timed_step("orchestrator.branch:policy_answer"):
                return self._policy_answer(route, ctx, trace)

        # 3c. Entity relation lookup.
        if route.intent == "entity_relation":
            with _timed_step("orchestrator.branch:entity_relation_answer"):
                return self._entity_relation_answer(route, ctx, trace)

        # 3d. Entity definition.
        if route.intent == "entity_definition":
            with _timed_step("orchestrator.branch:entity_definition_answer"):
                return self._entity_definition_answer(
                    route,
                    ctx,
                    trace,
                    answer_style=answer_style,
                )

        # 5. Unknown -> audit, missing knowledge.
        with _timed_step("orchestrator.branch:unknown_audit"):
            return self._unknown_audit(route, ctx, trace)

    # ------------------------------------------------------------------ #
    # Audit builders.
    # ------------------------------------------------------------------ #
    def _hard_safety_audit(self, route, ctx, trace) -> AssistantAnswer:
        reasons = {
            "current_live_request": (
                "this asks for current/live data, and no current source-qualified "
                "snapshot is available in Microworld's memory",
                "missing_knowledge",
            ),
            "private_sensitive_request": (
                "this asks for private/sensitive personal information, which "
                "Microworld's memory does not hold and must not fabricate",
                "audit_blocked_context",
            ),
            "relation_inversion": (
                "this asserts a reversed or unsupported directional relation, which "
                "the overlay does not support",
                "audit_blocked_context",
            ),
            "unsupported_universal": (
                "this asks for an unsupported universal generalization, which the "
                "overlay never licenses from individual relations",
                "audit_blocked_context",
            ),
        }
        reason, support_kind = reasons[route.intent]
        finalize(trace, "safety_gate", support_kind)
        return self._make(
            route, ctx, trace,
            decision="audit",
            answer_text="",
            supported=False,
            support_kind=support_kind,
            source_system="safety_gate",
            audit_reason=reason,
        )

    def _web_search_current_answer(
        self,
        route,
        ctx,
        trace,
        *,
        web_search_enabled: bool | None,
    ) -> AssistantAnswer | None:
        enabled = self._web_search_enabled_default if web_search_enabled is None else web_search_enabled
        if self._web_search_provider is None and enabled:
            from worldpgt.web_search.duckduckgo import DuckDuckGoInstantAnswerProvider

            self._web_search_provider = DuckDuckGoInstantAnswerProvider()
        if self._web_search_provider is None:
            return None
        try:
            results = self._web_search_provider.search(route.question, max_results=3)
        except Exception as exc:
            trace.add(f"web_search: error={exc.__class__.__name__}")
            return None
        if not results:
            trace.add("web_search: no_source_results")
            return None
        text = render_web_answer(route.question, results)
        if not text:
            trace.add("web_search: render_empty")
            return None
        trace.add(f"web_search: result_count={len(results)}")
        finalize(trace, "web_search", "web_search_result")
        return self._make(
            route,
            ctx,
            trace,
            decision="answer",
            answer_text=text,
            supported=True,
            support_kind="web_search_result",
            source_system="web_search",
            extra_risk=["current_live", "web_search_live", "source_qualified_volatile"],
        )

    def _unknown_audit(self, route, ctx, trace) -> AssistantAnswer:
        finalize(trace, "context_pack", "missing_knowledge")
        if route.notes == "semantic intersection query not supported":
            reason = "structured intersection questions are not supported by the current QA planner"
        elif route.notes == "question not understood":
            reason = "question not understood"
        else:
            reason = (
                "no explicitly supported answer exists in Microworld's current "
                "memory for this question"
            )
        return self._make(
            route, ctx, trace,
            decision="audit",
            answer_text="",
            supported=False,
            support_kind="missing_knowledge",
            source_system="context_pack",
            audit_reason=reason,
        )

    # ------------------------------------------------------------------ #
    # Answer builders.
    # ------------------------------------------------------------------ #
    def _connection_answer(self, route, ctx, trace) -> AssistantAnswer:
        analyzed = analyze_cross_page(route.question)
        plan = self._cross_page_planner.plan(analyzed)
        trace.add(f"cross_page_qa: intent={analyzed.intent}, decision={plan.decision}")

        # Require an explicit (non-weak) candidate path AND a planner answer.
        if plan.decision == "answer" and ctx.has_explicit_path:
            finalize(trace, "cross_page_qa", "explicit_connection_path")
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_cross_page(plan),
                supported=True,
                support_kind="explicit_connection_path",
                source_system="cross_page_qa",
            )

        finalize(trace, "cross_page_qa", "missing_knowledge")
        return self._make(
            route, ctx, trace,
            decision="audit",
            answer_text="",
            supported=False,
            support_kind="missing_knowledge",
            source_system="cross_page_qa",
            audit_reason=(
                "no explicit stable path connects these entities "
                "(only weak or no links)"
            ),
        )

    def _source_fact_answer(self, route, ctx, trace) -> AssistantAnswer:
        analyzed = analyze_entity(route.question, index=self._surface_index)
        plan = self._entity_planner.plan(analyzed)
        trace.add(f"entity_qa: intent={analyzed.intent}, decision={plan.decision}")

        if plan.decision == "answer" and ctx.has_source_fact:
            finalize(trace, "entity_qa", "source_qualified_fact")
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_entity(plan),
                supported=True,
                support_kind="source_qualified_fact",
                source_system="entity_qa",
                extra_risk=["source_qualified_volatile"],
            )

        finalize(trace, "entity_qa", "missing_knowledge")
        return self._make(
            route, ctx, trace,
            decision="audit",
            answer_text="",
            supported=False,
            support_kind="missing_knowledge",
            source_system="entity_qa",
            audit_reason=(
                "no source-qualified fact was found for this subject"
            ),
        )

    def _policy_answer(self, route, ctx, trace) -> AssistantAnswer:
        # Prefer the existing entity link-explanation renderer for "why is X
        # linked to Y"; otherwise emit a static policy explanation. Either way
        # this is a safe policy answer, never a new factual claim.
        analyzed = analyze_entity(route.question, index=self._surface_index)
        plan = self._entity_planner.plan(analyzed)
        trace.add(f"entity_qa(policy): intent={analyzed.intent}, decision={plan.decision}")

        if plan.decision == "answer" and analyzed.intent == "link_explanation":
            text = render_entity(plan)
        elif "source" in route.question.lower():
            text = _SOURCE_QUALIFIED_POLICY_TEXT
        else:
            text = _WEAK_LINK_POLICY_TEXT

        finalize(trace, "context_pack", "safe_policy_answer")
        return self._make(
            route, ctx, trace,
            decision="answer",
            answer_text=text,
            supported=True,
            support_kind="safe_policy_answer",
            source_system="context_pack",
        )

    def _entity_relation_answer(self, route, ctx, trace) -> AssistantAnswer:
        semantic = parse_semantic_query(route.question, self._surface_index)
        qe_plan = qe_plan_builder.build(route.question, semantic)
        if qe_plan.question_type_hint in self._QE_ONLY_HINTS and qe_plan.confidence > 0.7:
            qe_result = qe_executor.execute(qe_plan, self._provider, self._ontology_layer_items)
            if qe_result.success:
                finalize(trace, "query_engine", qe_result.support_kind or "stable_relation")
                return self._make(
                    route, ctx, trace,
                    decision=qe_result.decision,
                    answer_text=qe_result.answer_text or "",
                    supported=True,
                    support_kind=qe_result.support_kind or "stable_relation",
                    source_system="query_engine",
                )

        analyzed = analyze_entity(route.question, index=self._surface_index)
        plan = self._entity_planner.plan(analyzed)
        trace.add(f"entity_qa: intent={analyzed.intent}, decision={plan.decision}")

        if plan.decision == "no":
            support_kind = str(plan.render_args.get("support_kind", "explicit_type_contradiction"))
            finalize(trace, "entity_qa", support_kind)
            return self._make(
                route, ctx, trace,
                decision="no",
                answer_text=render_entity(plan),
                supported=True,
                support_kind=support_kind,
                source_system="entity_qa",
            )

        if plan.decision == "answer" and plan.render_template == "ontology_is_a":
            finalize(trace, "entity_qa", "explicit_is_a_chain")
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_entity(plan),
                supported=True,
                support_kind="explicit_is_a_chain",
                source_system="entity_qa",
            )

        if plan.decision == "answer" and plan.render_template in {
            "inverse_relation_lookup",
            "comparative_intersection",
        }:
            support_kind = self._planner_relation_support_kind(plan.render_args)
            finalize(trace, "entity_qa", support_kind)
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_entity(plan),
                supported=True,
                support_kind=support_kind,
                source_system="entity_qa",
            )

        if plan.decision == "answer" and plan.render_template == "definition_relation_lookup":
            finalize(trace, "entity_qa", "stable_definition")
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_entity(plan),
                supported=True,
                support_kind="stable_definition",
                source_system="entity_qa",
            )

        if plan.decision == "answer" and (
            ctx.has_stable_relation or ctx.has_semi_stable_relation
        ):
            support_kind = (
                "stable_relation" if ctx.has_stable_relation else "semi_stable_relation"
            )
            finalize(trace, "entity_qa", support_kind)
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_entity(plan),
                supported=True,
                support_kind=support_kind,
                source_system="entity_qa",
            )

        finalize(trace, "entity_qa", "missing_knowledge")
        return self._make(
            route, ctx, trace,
            decision="audit",
            answer_text="",
            supported=False,
            support_kind="missing_knowledge",
            source_system="entity_qa",
            audit_reason=plan.audit_reason or "No relevant information found for this topic.",
        )

    @staticmethod
    def _planner_relation_support_kind(render_args: dict) -> str:
        relations = list(render_args.get("relations", []))
        relations.extend(render_args.get("common_pairs", []))
        if any(r.get("stability") == "semi_stable" for r in relations):
            return "semi_stable_relation"
        return "stable_relation"

    def _entity_definition_answer(
        self,
        route,
        ctx,
        trace,
        *,
        answer_style: str = "normal",
    ) -> AssistantAnswer:
        analyzed = analyze_entity(route.question, index=self._surface_index)
        plan = self._entity_planner.plan(analyzed)
        trace.add(f"entity_qa: intent={analyzed.intent}, decision={plan.decision}")
        if plan.render_template == "open_synthesis":
            plan.render_args["answer_style"] = answer_style

        plan_relations = plan.render_args.get("relations", []) if plan.render_args else []
        has_is_a_relation = any(r.get("predicate") == "is_a" for r in plan_relations)

        if plan.decision == "answer" and ctx.has_stable_definition:
            finalize(trace, "entity_qa", "stable_definition")
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_entity(plan),
                supported=True,
                support_kind="stable_definition",
                source_system="entity_qa",
            )

        if plan.decision == "answer" and has_is_a_relation and (
            ctx.has_stable_relation or ctx.has_semi_stable_relation
        ):
            support_kind = (
                "stable_relation" if ctx.has_stable_relation else "semi_stable_relation"
            )
            finalize(trace, "entity_qa", support_kind)
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=render_entity(plan),
                supported=True,
                support_kind=support_kind,
                source_system="entity_qa",
            )

        finalize(trace, "entity_qa", "missing_knowledge")
        return self._make(
            route, ctx, trace,
            decision="audit",
            answer_text="",
            supported=False,
            support_kind="missing_knowledge",
            source_system="entity_qa",
            audit_reason="I don't have a definition for this in my knowledge base.",
        )

    # ------------------------------------------------------------------ #
    def _make(
        self,
        route: AssistantRoute,
        ctx: AssistantContextSummary,
        trace,
        *,
        decision: str,
        answer_text: str,
        supported: bool,
        support_kind: str,
        source_system: str,
        audit_reason: str | None = None,
        extra_risk: list[str] | None = None,
    ) -> AssistantAnswer:
        risk = list(route.risk_flags)
        if extra_risk:
            risk.extend(extra_risk)
        risk = list(dict.fromkeys(risk))
        # The audit reason is carried as the answer body for audits so the
        # renderer can present it.
        body = answer_text
        if decision == "audit" and audit_reason:
            body = audit_reason
        return AssistantAnswer(
            question=route.question,
            decision=decision,
            route=route.intent,
            answer_text=body,
            overlay_mode=self.overlay_mode,
            supported_by_context=supported,
            support_kind=support_kind,
            risk_flags=risk,
            source_system=source_system,
            trace=trace,
            safe_for_general_runtime=False,
        )
