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

from worldpgt.assistant_surface.context_selector import ContextSelector, resolve_overlay
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
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

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


class AnswerOrchestrator:
    def __init__(self, overlay_mode: str = "promoted", overlay_path: str | None = None) -> None:
        self.overlay_mode = OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode
        overlay_path_resolved = overlay_path
        if overlay_path_resolved is None:
            overlay_path_resolved, _ = resolve_overlay(overlay_mode)
        self._provider = WikiMemoryOverlayProvider(overlay_path_resolved)
        self._entity_planner = EntityAnswerPlanner(provider=self._provider)
        self._cross_page_planner = CrossPageAnswerPlanner(provider=self._provider)
        self._selector = ContextSelector(overlay_mode, overlay_path=overlay_path)

    # ------------------------------------------------------------------ #
    def answer(self, question: str) -> AssistantAnswer:
        route = route_question(question)
        trace = new_trace(route)
        _pack, ctx = self._selector.select(question)
        attach_context(trace, ctx)

        # 1. Hard safety route — always audit.
        if route.is_hard_safety:
            return self._hard_safety_audit(route, ctx, trace)

        # 2. Connection / path questions -> cross-page QA.
        if route.intent == "connection_path":
            return self._connection_answer(route, ctx, trace)

        # 3a. Source-qualified facts -> entity QA.
        if route.intent == "source_qualified_fact":
            return self._source_fact_answer(route, ctx, trace)

        # 3b. Weak-link / policy explanations -> safe policy answer.
        if route.intent == "weak_link_policy":
            return self._policy_answer(route, ctx, trace)

        # 3c. Entity relation lookup.
        if route.intent == "entity_relation":
            return self._entity_relation_answer(route, ctx, trace)

        # 3d. Entity definition.
        if route.intent == "entity_definition":
            return self._entity_definition_answer(route, ctx, trace)

        # 5. Unknown -> audit, missing knowledge.
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

    def _unknown_audit(self, route, ctx, trace) -> AssistantAnswer:
        finalize(trace, "context_pack", "missing_knowledge")
        return self._make(
            route, ctx, trace,
            decision="audit",
            answer_text="",
            supported=False,
            support_kind="missing_knowledge",
            source_system="context_pack",
            audit_reason=(
                "no explicitly supported answer exists in Microworld's current "
                "memory for this question"
            ),
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
                "the overlay has no explicit stable path connecting these entities "
                "(only weak or no links)"
            ),
        )

    def _source_fact_answer(self, route, ctx, trace) -> AssistantAnswer:
        analyzed = analyze_entity(route.question)
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
                "no source-qualified fact for this subject is present in the overlay"
            ),
        )

    def _policy_answer(self, route, ctx, trace) -> AssistantAnswer:
        # Prefer the existing entity link-explanation renderer for "why is X
        # linked to Y"; otherwise emit a static policy explanation. Either way
        # this is a safe policy answer, never a new factual claim.
        analyzed = analyze_entity(route.question)
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
        analyzed = analyze_entity(route.question)
        plan = self._entity_planner.plan(analyzed)
        trace.add(f"entity_qa: intent={analyzed.intent}, decision={plan.decision}")

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
            audit_reason="no stable relation for this subject is present in the overlay",
        )

    def _entity_definition_answer(self, route, ctx, trace) -> AssistantAnswer:
        analyzed = analyze_entity(route.question)
        plan = self._entity_planner.plan(analyzed)
        trace.add(f"entity_qa: intent={analyzed.intent}, decision={plan.decision}")

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
            audit_reason="no stable definition for this entity is present in the overlay",
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
