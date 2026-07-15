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
import secrets
import re
from dataclasses import replace
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from worldpgt.assistant_surface.context_selector import ContextSelector, resolve_overlay
from worldpgt.assistant_surface.answer_style import resolve_answer_style
from worldpgt.assistant_surface.perf_timing import step as _timed_step
from worldpgt.assistant_surface.question_router import route as route_question
from worldpgt.assistant_surface.assistant_trace import (
    attach_cognitive_plan,
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
from worldpgt.reasoning.graph_input import GraphInputLayer
from worldpgt.assistant_surface.community_context import (
    CognitivePatternProvider,
    CommunityContextProvider,
    FileCognitivePatternProvider,
    FileCommunityContextProvider,
    apply_community_tone,
    render_context_answer,
)
from worldpgt.assistant_surface.cognitive_surface import apply_cognitive_surface
from worldpgt.assistant_surface.web_search import WebSearchProvider, render_web_answer
from worldpgt.cognition.phrase_graph import generate as generate_phrase_graph
from worldpgt.cognition.creative_generator import generate_creative
from worldpgt.cognition.reasoning_engine import reason_over_plan
from worldpgt.entity_qa.symbolic_text_generator import generate_text
from worldpgt.entity_qa.semantic_speech_planner import build_speech_plan
from worldpgt.web_search.live_cache import LiveSearchCache, entity_key_from_title
from worldpgt.web_search.answer_extraction import extract_answer
from worldpgt.web_search.query_intent import filter_and_rank_results


_FACTUAL_LOOKUP_SHAPE_RE = re.compile(
    r"^\s*(?:what\s+(?:does|did)|who\s+(?:is|was|founded)|where\s+(?:is|are|was|were))\b",
    re.IGNORECASE,
)
_COMMUNITY_ADVICE_SHAPE_RE = re.compile(
    r"\b(?:common\s+concerns?|advice|tips|people(?:'s)?\s+experiences?)\b",
    re.IGNORECASE,
)

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
_CURRENT_OFFICEHOLDER_QUERY_RE = re.compile(
    r"\b(?:current\s+)?(?:president|prime minister|mayor|governor|officeholder|incumbent)\b",
    re.IGNORECASE,
)

_HOW_WORKS_RE = re.compile(
    r"^\s*(?:how\s+(?:do|does)\s+.+?\bwork\b|"
    r"explain\s+how\s+.+?\bworks?\b|"
    r"what\s+is\s+the\s+operating\s+mechanism\s+of\s+.+?)",
    re.IGNORECASE,
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
    promoted_path = Path(promoted_overlay_path)
    graph_items = json.loads(promoted_path.read_text(encoding="utf-8"))
    return EntitySurfaceIndex(
        accepted_overlay_path=_EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json",
        promoted_overlay_path=promoted_path,
        snapshot_overlay_path=_EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
        graph_input=GraphInputLayer.from_overlay_items(
            graph_items if isinstance(graph_items, list) else ()
        ),
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
        web_search_deadline_sec: float | None = None,
        live_cache: LiveSearchCache | None = None,
        community_context_provider: CommunityContextProvider | None = None,
        community_context_path: str | None = None,
        community_context_enabled: bool | None = None,
        cognitive_pattern_provider: CognitivePatternProvider | None = None,
        cognitive_patterns_path: str | None = None,
        cognitive_patterns_enabled: bool | None = None,
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
        self._web_search_deadline_sec = web_search_deadline_sec
        self._live_cache = live_cache
        if community_context_enabled is None:
            community_env_enabled = os.environ.get(
                "MICROWORLD_COMMUNITY_CONTEXT_ENABLED", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            community_context_enabled = (
                community_context_provider is not None
                or bool(community_context_path)
                or community_env_enabled
            )
        self._community_context_enabled_default = community_context_enabled
        if community_context_provider is None and community_context_path:
            community_context_provider = FileCommunityContextProvider(community_context_path)
        self._community_context_provider = community_context_provider
        if cognitive_patterns_path is None and community_context_path:
            sibling = Path(community_context_path).parent / "cognitive_pattern_events.json"
            if sibling.is_file():
                cognitive_patterns_path = str(sibling)
        if cognitive_patterns_enabled is None:
            cognitive_env_enabled = os.environ.get(
                "MICROWORLD_COGNITIVE_PATTERNS_ENABLED", ""
            ).strip().lower() in {"1", "true", "yes", "on"}
            cognitive_patterns_enabled = (
                cognitive_pattern_provider is not None
                or bool(cognitive_patterns_path)
                or cognitive_env_enabled
            )
        if cognitive_pattern_provider is None and cognitive_patterns_path:
            cognitive_pattern_provider = FileCognitivePatternProvider(cognitive_patterns_path)
        self._cognitive_patterns_enabled_default = cognitive_patterns_enabled
        self._cognitive_pattern_provider = cognitive_pattern_provider
        self._entity_planner = EntityAnswerPlanner(
            provider=self._provider,
            ontology_layer_items=ontology_layer_items,
            inference_workspace=inference_workspace,
        )
        self._cross_page_planner = CrossPageAnswerPlanner(provider=self._provider)
        self._selector = ContextSelector(overlay_mode, overlay_path=overlay_path)

    def set_community_context_provider(
        self,
        provider: CommunityContextProvider | None,
        *,
        enabled: bool = True,
    ) -> None:
        self._community_context_provider = provider
        self._community_context_enabled_default = bool(enabled and provider is not None)

    def set_cognitive_pattern_provider(
        self,
        provider: CognitivePatternProvider | None,
        *,
        enabled: bool = True,
    ) -> None:
        self._cognitive_pattern_provider = provider
        self._cognitive_patterns_enabled_default = bool(enabled and provider is not None)

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
        community_context_enabled: bool | None = None,
        cognitive_patterns_enabled: bool | None = None,
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
        if not route.is_hard_safety:
            self._attach_cognitive_pattern_plan(
                route.question,
                trace,
                cognitive_patterns_enabled=cognitive_patterns_enabled,
            )

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
                        return self._apply_community_tone_if_enabled(
                            answer,
                            community_context_enabled=community_context_enabled,
                        )
                return self._hard_safety_audit(route, ctx, trace)

        # 2. Creative free-generation — a separate layer from factual QA.
        # It is reached only after every hard-safety screen in the router has
        # passed, so it can never fabricate a private/current-sensitive claim.
        # Output is recombined from learned prose (never a fact lookup) and is
        # explicitly labelled as generated, not verified.
        if route.intent == "creative_request":
            with _timed_step("orchestrator.branch:creative"):
                return self._apply_community_tone_if_enabled(
                    self._creative_answer(route, ctx, trace),
                    community_context_enabled=community_context_enabled,
                )

        # 2. Connection / path questions -> cross-page QA.
        if route.intent == "connection_path":
            with _timed_step("orchestrator.branch:connection_answer"):
                result = self._connection_answer(route, ctx, trace)

        # 3a. Source-qualified facts -> entity QA.
        elif route.intent == "source_qualified_fact":
            with _timed_step("orchestrator.branch:source_fact_answer"):
                result = self._source_fact_answer(route, ctx, trace)

        # 3b. Weak-link / policy explanations -> safe policy answer.
        elif route.intent == "weak_link_policy":
            with _timed_step("orchestrator.branch:policy_answer"):
                result = self._policy_answer(route, ctx, trace)

        # 3c. Entity relation lookup.
        elif route.intent == "entity_relation":
            with _timed_step("orchestrator.branch:entity_relation_answer"):
                result = self._entity_relation_answer(route, ctx, trace)

        # 3d. Entity definition.
        elif route.intent == "entity_definition":
            with _timed_step("orchestrator.branch:entity_definition_answer"):
                result = self._entity_definition_answer(
                    route,
                    ctx,
                    trace,
                    answer_style=answer_style,
                    web_search_enabled=web_search_enabled,
                )

        # 5. Unknown -> audit, missing knowledge.
        else:
            with _timed_step("orchestrator.branch:unknown_audit"):
                result = self._unknown_audit(route, ctx, trace)

        # 6. A genuine "we don't have this" audit (never a safety-blocked one —
        # those return above, in step 1, untouched) gets one more chance via
        # live web search before it is returned as final.
        if result.decision == "audit" and result.support_kind == "missing_knowledge":
            graph_answer = self._outgoing_graph_neighborhood_answer(route, ctx, trace)
            if graph_answer is not None:
                return graph_answer
            with _timed_step("orchestrator.branch:web_search_fallback"):
                web_answer = self._web_search_current_answer(
                    route,
                    ctx,
                    trace,
                    web_search_enabled=web_search_enabled,
                )
                if web_answer is not None:
                    web_answer = self._apply_cognitive_surface_if_enabled(
                        web_answer,
                        cognitive_patterns_enabled=cognitive_patterns_enabled,
                    )
                    return self._apply_community_tone_if_enabled(
                        web_answer,
                        community_context_enabled=community_context_enabled,
                    )
                # Community material can help with advice and phrasing, but it
                # must never replace a missing factual lookup with generic
                # prose.  This remains true even when the entity itself failed
                # to resolve (for example, a shortened paper title).
                if not self._is_factual_lookup(route):
                    community_answer = self._community_context_answer(
                        route,
                        ctx,
                        trace,
                        community_context_enabled=community_context_enabled,
                    )
                    if community_answer is not None:
                        return community_answer
                else:
                    trace.add("community_context: skipped_for_factual_lookup")

        result = self._apply_cognitive_surface_if_enabled(
            result,
            cognitive_patterns_enabled=cognitive_patterns_enabled,
        )
        return self._apply_community_tone_if_enabled(
            result,
            community_context_enabled=community_context_enabled,
        )

    def _community_tone_enabled(self, community_context_enabled: bool | None) -> bool:
        enabled = (
            self._community_context_enabled_default
            if community_context_enabled is None
            else community_context_enabled
        )
        return bool(enabled and self._community_context_provider is not None)

    def _outgoing_graph_neighborhood_answer(self, route, ctx, trace) -> AssistantAnswer | None:
        """Use one resolved entity's direct graph neighborhood as a last factual path.

        This is intentionally limited to questions whose relation was not
        understood at all.  A failed exact relation lookup must not be replaced
        with unrelated facts from the same vertex.
        """
        if route.intent != "unknown_or_unsupported":
            return None
        matches = self._surface_index.find_in_text(route.question)
        subjects = list(dict.fromkeys(canonical for _surface, canonical, _start, _end in matches))
        if len(subjects) != 1:
            trace.add(f"graph_neighborhood: resolved_subjects={len(subjects)}")
            return None
        subject = subjects[0]
        plan = self._entity_planner.plan_outgoing_neighborhood(route.question, subject)
        trace.add(
            f"graph_neighborhood: subject={subject}, decision={plan.decision}, "
            f"edge_count={len(plan.render_args.get('relations', []))}"
        )
        if plan.decision != "answer":
            return None
        support_kind = self._planner_relation_support_kind(plan.render_args)
        finalize(trace, "entity_qa", support_kind)
        return self._make(
            route,
            ctx,
            trace,
            decision="answer",
            answer_text=render_entity(plan),
            supported=True,
            support_kind=support_kind,
            source_system="entity_qa",
        )

    @staticmethod
    def _is_factual_lookup(route: AssistantRoute) -> bool:
        """Whether an unsupported question must fail closed instead of using community prose."""
        if _COMMUNITY_ADVICE_SHAPE_RE.search(route.question):
            return False
        return route.intent in {
            "connection_path",
            "entity_definition",
            "entity_relation",
            "source_qualified_fact",
        } or bool(_FACTUAL_LOOKUP_SHAPE_RE.match(route.question))

    def _cognitive_patterns_enabled(self, cognitive_patterns_enabled: bool | None) -> bool:
        enabled = (
            self._cognitive_patterns_enabled_default
            if cognitive_patterns_enabled is None
            else cognitive_patterns_enabled
        )
        return bool(enabled and self._cognitive_pattern_provider is not None)

    def _attach_cognitive_pattern_plan(
        self,
        question: str,
        trace,
        *,
        cognitive_patterns_enabled: bool | None,
    ) -> None:
        if not self._cognitive_patterns_enabled(cognitive_patterns_enabled):
            return
        try:
            plan = self._cognitive_pattern_provider.plan(question, max_patterns=5)
        except Exception as exc:
            trace.add(f"cognitive_patterns: error={exc.__class__.__name__}")
            return
        if not isinstance(plan, dict):
            trace.add("cognitive_patterns: invalid_plan")
            return
        plan["factual_support_allowed_from_patterns"] = False
        attach_cognitive_plan(trace, plan)

    def _apply_community_tone_if_enabled(
        self,
        answer: AssistantAnswer,
        *,
        community_context_enabled: bool | None,
    ) -> AssistantAnswer:
        if not self._community_tone_enabled(community_context_enabled):
            return answer
        if answer.decision != "answer":
            return answer
        if answer.source_system == "community_context":
            return answer
        styled = apply_community_tone(
            answer.question,
            answer.answer_text,
            support_kind=answer.support_kind,
            source_system=answer.source_system,
        )
        if styled == answer.answer_text:
            return answer
        risk = list(dict.fromkeys([*answer.risk_flags, "community_style_tone"]))
        return replace(answer, answer_text=styled, risk_flags=risk)

    def _apply_cognitive_surface_if_enabled(
        self,
        answer: AssistantAnswer,
        *,
        cognitive_patterns_enabled: bool | None,
    ) -> AssistantAnswer:
        if not self._cognitive_patterns_enabled(cognitive_patterns_enabled):
            return answer
        if answer.decision != "answer":
            return answer
        plan = answer.trace.cognitive_plan if answer.trace else None
        if not plan:
            return answer
        shaped = apply_cognitive_surface(
            answer.question,
            answer.answer_text,
            support_kind=answer.support_kind,
            source_system=answer.source_system,
            cognitive_plan=plan,
        )
        if shaped == answer.answer_text:
            return answer
        risk = list(dict.fromkeys([*answer.risk_flags, "cognitive_pattern_surface"]))
        return replace(answer, answer_text=shaped, risk_flags=risk)

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

    @staticmethod
    def _web_answer_lead(question: str, results) -> str | None:
        """Extract the single sentence answering ``question`` from the top
        result's text, so the rendered answer leads with the specific fact
        instead of a generic abstract. Returns None when nothing qualifies
        (render then falls back to the truncated snippet)."""
        if not results:
            return None
        primary = results[0]
        subject = primary.title
        for suffix in (" - Wikipedia", " — Wikipedia"):
            if subject.endswith(suffix):
                subject = subject[: -len(suffix)]
                break
        return extract_answer(question, primary.snippet, subject=subject)

    @staticmethod
    def _requires_extracted_web_lead(question: str) -> bool:
        return bool(_CURRENT_OFFICEHOLDER_QUERY_RE.search(question or ""))

    def _augment_mechanism_gap(
        self,
        route,
        ctx,
        trace,
        plan,
        *,
        answer_text: str,
        web_search_enabled: bool | None,
    ) -> str:
        """For "how does X work" questions, don't silently narrate around a
        missing mechanism.

        Reuses the same reasoning trace (``reason_over_plan``) that already
        detects missing evidence roles elsewhere, instead of guessing from
        answer text — then gives live web search one targeted try at filling
        the mechanism specifically before admitting the gap honestly. Never
        replaces an already-supported answer; only appends.
        """

        if not _HOW_WORKS_RE.search(route.question or ""):
            return answer_text
        speech_first_gap = any(
            step.startswith("speech_first: task=mechanism_explanation; action=answer_with_gap")
            for step in trace.steps
        )
        if speech_first_gap and not web_search_enabled:
            return answer_text
        if plan.render_template != "open_synthesis" or not plan.render_args:
            return answer_text
        synthesis = plan.render_args.get("synthesis")
        if synthesis is None or not getattr(synthesis, "matched", False):
            return answer_text

        speech_plan = build_speech_plan(synthesis, route.question)
        reasoning_trace = reason_over_plan(speech_plan)
        if not any(m.role == "mechanism" for m in reasoning_trace.missing_evidence):
            return answer_text

        subject = synthesis.subject or route.subject or "it"
        trace.add("mechanism_gap: detected_missing_mechanism_evidence")
        web_answer = self._web_search_current_answer(
            route, ctx, trace, web_search_enabled=web_search_enabled
        )
        if web_answer is not None and web_answer.answer_text:
            trace.add("mechanism_gap: filled_by_web_search")
            return (
                f"{answer_text.rstrip()}\n\n"
                f"I don't have verified mechanism details for {subject} "
                f"in memory, so here is what a live web search found:\n\n"
                f"{web_answer.answer_text}"
            )
        if speech_first_gap:
            return answer_text
        trace.add("mechanism_gap: admitted_no_web_result")
        return (
            f"{answer_text.rstrip()}\n\n"
            f"Here is the honest version: I can identify {subject} and describe "
            "what is already supported, but I do not yet have the mechanism: "
            "the parts and steps that make it work."
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
        if not enabled:
            return None
        if self._web_search_provider is None:
            from worldpgt.web_search.composite import CompositeSearchProvider, DEFAULT_DEADLINE_SEC

            deadline = (
                DEFAULT_DEADLINE_SEC
                if self._web_search_deadline_sec is None
                else self._web_search_deadline_sec
            )
            self._web_search_provider = CompositeSearchProvider(deadline_sec=deadline)
        if self._web_search_provider is None:
            return None

        if self._live_cache is not None:
            # 1. Exact repeat of this question — fastest path.
            cached = self._live_cache.get(route.question)
            cache_note = "cache_hit"
            # 2. A DIFFERENT question naming an entity already learned from a
            # previous live search — no network, local extraction only. This
            # is what makes the system get faster with use: the first
            # question about an entity pays the network cost once, every
            # later question about the same entity (any wording) doesn't.
            if cached is None:
                cached = self._live_cache.find_entity_in_question(route.question)
                cache_note = "entity_cache_hit"
            if cached is not None:
                _intent, ranked_results = filter_and_rank_results(route.question, cached.results)
                if not ranked_results:
                    trace.add(f"web_search: {cache_note} irrelevant_cached_results")
                    return None
                lead = self._web_answer_lead(route.question, ranked_results)
                if self._requires_extracted_web_lead(route.question) and not lead:
                    trace.add(f"web_search: {cache_note} lacked_specific_current_officeholder_answer")
                    return None
                cached_text = render_web_answer(
                    route.question,
                    ranked_results,
                    fetched_at=cached.fetched_at,
                    from_cache=True,
                    lead=lead,
                )
                if cached_text:
                    trace.add(f"web_search: {cache_note} fetched_at={cached.fetched_at}")
                    finalize(trace, "web_search", "web_search_result")
                    return self._make(
                        route,
                        ctx,
                        trace,
                        decision="answer",
                        answer_text=cached_text,
                        supported=True,
                        support_kind="web_search_result",
                        source_system="web_search",
                        extra_risk=[
                            "current_live", "web_search_live",
                            "source_qualified_volatile", "web_search_cached",
                        ],
                    )

        try:
            results = self._web_search_provider.search(route.question, max_results=3)
        except Exception as exc:
            trace.add(f"web_search: error={exc.__class__.__name__}")
            return None
        if not results:
            trace.add("web_search: no_source_results")
            return None
        _intent, results = filter_and_rank_results(route.question, results)
        if not results:
            trace.add("web_search: no_relevant_source_results")
            return None
        fetched_at = datetime.now(timezone.utc).isoformat()
        lead = self._web_answer_lead(route.question, results)
        if self._requires_extracted_web_lead(route.question) and not lead:
            trace.add("web_search: lacked_specific_current_officeholder_answer")
            return None
        text = render_web_answer(
            route.question,
            results,
            fetched_at=fetched_at,
            from_cache=False,
            lead=lead,
        )
        if not text:
            trace.add("web_search: render_empty")
            return None
        if self._live_cache is not None:
            self._live_cache.put(route.question, results)
            entity_key = entity_key_from_title(results[0].title)
            if entity_key:
                self._live_cache.put_entity(entity_key, results)
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

    def _community_context_answer(
        self,
        route,
        ctx,
        trace,
        *,
        community_context_enabled: bool | None,
    ) -> AssistantAnswer | None:
        enabled = (
            self._community_context_enabled_default
            if community_context_enabled is None
            else community_context_enabled
        )
        if not enabled or self._community_context_provider is None:
            return None
        try:
            results = self._community_context_provider.search(route.question, max_results=4)
        except Exception as exc:
            trace.add(f"community_context: error={exc.__class__.__name__}")
            return None
        if not results:
            trace.add("community_context: no_matches")
            return None
        text = render_context_answer(route.question, results)
        if not text:
            trace.add("community_context: render_empty")
            return None
        trace.add(f"community_context: result_count={len(results)}")
        finalize(trace, "community_context", "safe_policy_answer")
        return self._make(
            route,
            ctx,
            trace,
            decision="answer",
            answer_text=text,
            supported=True,
            support_kind="safe_policy_answer",
            source_system="community_context",
            extra_risk=["community_context_only", "low_trust_source"],
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
        web_search_enabled: bool | None = None,
    ) -> AssistantAnswer:
        analysis_question = route.question
        if route.notes == "mechanism open synthesis query" and route.subject:
            analysis_question = f"How does {route.subject} work?"
        analyzed = analyze_entity(analysis_question, index=self._surface_index)
        plan = self._entity_planner.plan(analyzed)
        trace.add(f"entity_qa: intent={analyzed.intent}, decision={plan.decision}")
        if plan.render_template == "open_synthesis":
            plan.render_args["answer_style"] = answer_style

        plan_relations = plan.render_args.get("relations", []) if plan.render_args else []
        has_is_a_relation = any(r.get("predicate") == "is_a" for r in plan_relations)

        if plan.decision == "answer" and ctx.has_stable_definition:
            finalize(trace, "entity_qa", "stable_definition")
            speech_first = self._render_open_synthesis_speech_first(
                route, plan, trace, answer_style=answer_style
            )
            answer_text = self._augment_mechanism_gap(
                route, ctx, trace, plan,
                answer_text=speech_first or render_entity(plan),
                web_search_enabled=web_search_enabled,
            )
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=answer_text,
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
            speech_first = self._render_open_synthesis_speech_first(
                route, plan, trace, answer_style=answer_style
            )
            answer_text = self._augment_mechanism_gap(
                route, ctx, trace, plan,
                answer_text=speech_first or render_entity(plan),
                web_search_enabled=web_search_enabled,
            )
            return self._make(
                route, ctx, trace,
                decision="answer",
                answer_text=answer_text,
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

    def _render_open_synthesis_speech_first(
        self,
        route,
        plan,
        trace,
        *,
        answer_style: str,
    ) -> str | None:
        """Render open synthesis as reasoning -> speech over KB facts.

        The entity planner is treated as a knowledge-base lookup that returns a
        supported ``SynthesisAnswer``. From there, the answer surface is driven
        by a speech plan and reasoning action before any final wording is
        chosen. This keeps facts and speech separate: the speech layer can
        choose wording/order/gap handling, but it never queries memory or adds
        facts.
        """

        if plan.render_template != "open_synthesis" or not plan.render_args:
            return None
        synthesis = plan.render_args.get("synthesis")
        if synthesis is None or not getattr(synthesis, "matched", False):
            return None

        speech_plan = build_speech_plan(
            synthesis,
            route.question,
            answer_style=answer_style,
        )
        reasoning_trace = reason_over_plan(speech_plan)
        action = reasoning_trace.action
        if reasoning_trace.thought_loop and reasoning_trace.thought_loop.decision:
            action = reasoning_trace.thought_loop.decision.action

        trace.add(
            "speech_first: "
            f"task={reasoning_trace.task.intent}; "
            f"action={action.next_action}; "
            f"confidence={reasoning_trace.confidence}"
        )

        if action.next_action == "answer" and not reasoning_trace.missing_evidence:
            learned = generate_phrase_graph(synthesis, answer_style=answer_style)
            if learned:
                trace.add("speech_first: renderer=phrase_graph")
                return learned

        trace.add("speech_first: renderer=reasoned_symbolic")
        return generate_text(speech_plan, action=action)

    # ------------------------------------------------------------------ #
    _CREATIVE_LABEL = (
        "[Creative mode — generated, recombined from learned text, not verified fact.]"
    )

    def _creative_answer(
        self,
        route: AssistantRoute,
        ctx: AssistantContextSummary,
        trace,
    ) -> AssistantAnswer:
        """Free-generation path: recombine learned prose under the inverted gate.

        The opposite polarity to the factual layer. The factual path answers
        only from grounded overlay facts and audits on a gap; this path invents
        by recombining learned word transitions and allows output only when it
        does not recite a corpus 4-gram. It asserts no fact, so it is returned
        with a mandatory creative label, ``support_kind='creative_generated'``,
        and ``supported=False`` — never mistakable for a verified answer.
        """

        # Creative mode is the one surface where per-request determinism is a
        # UX bug, not a feature: re-asking should re-roll a fresh passage. So a
        # random seed varies the output each call, while the generator stays
        # deterministic *given* a seed (used by tests/replay). Same choice
        # poetry_lab's NarrativeEngine makes when no explicit seed is supplied.
        passage = generate_creative(
            route.question, seed=secrets.token_hex(8), num_sentences=3
        )
        if not passage:
            trace.add("creative: renderer=creative_generator; result=empty")
            return self._make(
                route,
                ctx,
                trace,
                decision="audit",
                answer_text="",
                supported=False,
                support_kind="missing_knowledge",
                source_system="creative_generator",
                audit_reason=(
                    "I don't have enough learned material to generate something here."
                ),
            )
        trace.add("creative: renderer=creative_generator; gate=novelty(4-gram)")
        return self._make(
            route,
            ctx,
            trace,
            decision="answer",
            answer_text=f"{self._CREATIVE_LABEL}\n\n{passage}",
            supported=False,
            support_kind="creative_generated",
            source_system="creative_generator",
            extra_risk=["creative_generated"],
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
