"""Integrated answer router — unified dispatch across all answer branches.

A NEW dispatch layer that composes the existing branches WITHOUT modifying any of
them. Order (per the fixed architecture):

  1. Safety screen (UNCHANGED): ``route_question`` — if hard-safety
     (current/live, private/sensitive), delegate entirely to the QA orchestrator,
     which audits exactly as today.
  2. Branch routing (safe questions only): explicit fast-path markers, then the
     centroid ``BranchRouter``, then QA default.
  3. Dispatch to the unchanged branch module and wrap the output in a unified
     result whose ``confidence_level`` keeps every category architecturally
     DISTINCT (never collapsing speculative_extended into speculative_verified,
     nor constrained-generation into grounded QA).

Production branches are imported read-only and unchanged: QA (AnswerOrchestrator),
reflective_reasoning_v1 (verified speculation), reflective_reasoning_extended_v2
(extended speculation), constrained_creative_v1, and the Creative generator.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from worldpgt.assistant_surface.answer_orchestrator import (
    AnswerOrchestrator,
    _surface_index_for_overlay,
    route_question,
)
from worldpgt.assistant_surface.context_selector import resolve_overlay
from worldpgt.cognition.creative_generator import generate_creative
from worldpgt.reasoning.branch_router import BranchRouter
from worldpgt.reasoning import reflective_reasoning_v1 as rr1
from worldpgt.reasoning import reflective_reasoning_extended_v2 as rr2
from worldpgt.reasoning import constrained_creative_v1 as cc1

# Architecturally-distinct confidence levels. None may be merged in output.
CONFIDENCE_GROUNDED = "grounded"                     # QA fact answer
CONFIDENCE_GROUNDED_GENERATION = "grounded_generation"  # constrained-creative
CONFIDENCE_SPECULATIVE_VERIFIED = "speculative_verified"  # reflective v1 (proven)
CONFIDENCE_SPECULATIVE_EXTENDED = "speculative_extended"  # reflective v2 (weaker)
CONFIDENCE_CREATIVE = "creative_generated"           # pure creative
CONFIDENCE_AUDIT = "audit"


@dataclass
class IntegratedAnswer:
    question: str
    branch: str
    route_method: str
    confidence_level: str
    support_kind: str
    decision: str
    answer_text: str
    caution: str | None = None          # set only for speculative_extended
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "branch": self.branch,
            "route_method": self.route_method,
            "confidence_level": self.confidence_level,
            "support_kind": self.support_kind,
            "decision": self.decision,
            "answer_text": self.answer_text,
            "caution": self.caution,
            "detail": self.detail,
        }


class IntegratedAnswerRouter:
    def __init__(self, overlay_mode: str = "promoted", overlay_path: str | None = None):
        self.orchestrator = AnswerOrchestrator(overlay_mode, overlay_path=overlay_path)
        resolved = overlay_path or resolve_overlay(overlay_mode)[0]
        self._surface_index = _surface_index_for_overlay(resolved)
        self._overlay_items = json.loads(Path(resolved).read_text(encoding="utf-8"))
        self._edges = rr1.load_edges(self._overlay_items)
        self.router = BranchRouter().build()

    # ── public entry ───────────────────────────────────────────────────────── #

    def answer(self, question: str, *, force_branch: str | None = None) -> IntegratedAnswer:
        # 1. Safety screen — UNCHANGED. Hard-safety always goes through the QA
        #    orchestrator's audit path; the router never sees a blocked request.
        route = route_question(question, self._surface_index)
        if route.is_hard_safety:
            return self._qa(question, route_method="safety_screen", branch="qa_safety")

        # 2. Branch routing.
        if force_branch:
            branch, method = force_branch, "override"
        else:
            rr = self.router.route(question)
            branch, method = rr.branch, rr.method
            # Honor the proven creative intent classifier: if route_question tagged
            # this a creative_request and the router did not claim reflective/
            # constrained, treat it as pure creative.
            if route.intent == "creative_request" and branch in ("qa", "pure_creative"):
                branch, method = "pure_creative", "intent+router"

        # 3. Dispatch to the unchanged branch.
        if branch == "reflective":
            return self._reflective(question, method)
        if branch == "constrained_creative":
            return self._constrained(question, method)
        if branch == "pure_creative":
            return self._pure_creative(question, method)
        return self._qa(question, route_method=method, branch="qa")

    # ── branch dispatchers (each delegates to an unchanged module) ───────────── #

    def _qa(self, question: str, *, route_method: str, branch: str) -> IntegratedAnswer:
        ans = self.orchestrator.answer(question)
        level = CONFIDENCE_AUDIT if ans.decision == "audit" else CONFIDENCE_GROUNDED
        return IntegratedAnswer(
            question=question, branch=branch, route_method=route_method,
            confidence_level=level,
            # The unified public contract deliberately has a small, disjoint
            # vocabulary.  Preserve the QA branch's more precise evidence label
            # below so integration does not discard provenance.
            support_kind="audit" if ans.decision == "audit" else "grounded",
            decision=ans.decision, answer_text=ans.answer_text,
            detail={
                "source_system": ans.source_system,
                "branch_support_kind": ans.support_kind,
            },
        )

    def _reflective(self, question: str, method: str) -> IntegratedAnswer:
        # v1 (verified speculation) via the strict canonical regex first.
        plan = rr1.reflect(question, self._overlay_items)
        if plan is not None:
            if plan.decision == "speculative":
                return self._verified(question, method, plan)
            if plan.decision == "grounded_deferral":
                return self._qa(question, route_method=f"{method}->deferral", branch="qa")
            # plan.decision == "audit": v1 matched but declined; try fallbacks next.

        # Rule-level paraphrase fallback: call the PROVEN rules directly with the
        # extracted entities, so natural paraphrases beyond the strict regex still
        # reach the verified rules. The rules self-gate defensibility.
        ents = self._distinct_entities(question)
        low = question.lower()

        # (a) Counterfactual paraphrase — fire ONLY on a founding/existence cue
        #     (not production/development verbs), matching the proven rule's scope.
        existence_cue = any(k in low for k in (
            "without", "never been created", "never been founded", "never existed",
            "had never been", "hadn't founded", "had not founded", "never founded",
            "never created",
        ))
        production_cue = any(k in low for k in (
            "made", "make", "produce", "develop", "manufactur", "stopped",
        ))
        if existence_cue and not production_cue:
            cf = self._counterfactual_from_entities(ents)
            if cf is not None and cf.decision == "speculative":
                return self._verified(question, f"{method}->rule_fallback", cf)

        # Association paraphrase between two entities. ORDER MATTERS for keeping
        # the confidence levels distinct (see the OPEN QUESTION in
        # artifacts/integration_v1/final_report.md):
        #   (b) co-attribution (EXTENDED) is tried FIRST — two peer entities that
        #       share a kinship object are a similarity, not a derived bridge.
        #   (c) abduction (VERIFIED) only after, and only for a COHERENT directed
        #       bridge (the two premise edges must not share the same object —
        #       that degenerate case is v1 abduction's undirected path treating a
        #       shared object as a bridge, which both mislabels the level and
        #       renders as "X develops O, which develops O").
        association_cue = any(k in low for k in (
            "why might", "why is", "associated", "linked", "connected", "related", "tie",
        ))
        if association_cue and len(ents) >= 2:
            ext = rr2.co_attribution_for_pair(self._edges, ents[0], ents[1])
            if ext.decision == "speculative_extended":
                step = ext.steps[0]
                return IntegratedAnswer(
                    question=question, branch="reflective_extended", route_method=method,
                    confidence_level=CONFIDENCE_SPECULATIVE_EXTENDED,
                    support_kind=rr2.SUPPORT_KIND, decision="answer",
                    answer_text=rr2.render_extended(step),
                    caution="lower-confidence: shared-attribute association, not a verified inference",
                    detail={"x": step.x, "y": step.y, "shared_object": step.shared_object},
                )
            ab = rr1.abduction_explanation(self._edges, ents[0], ents[1])
            if ab.decision == "speculative" and not self._is_degenerate_bridge(ab):
                return self._verified(question, f"{method}->rule_fallback", ab)
            if ab.decision == "grounded_deferral":
                return self._qa(question, route_method=f"{method}->deferral", branch="qa")

        # Nothing defensible: honest reflective audit if v1 declined, else QA default.
        if plan is not None and plan.decision == "audit":
            return IntegratedAnswer(
                question=question, branch="reflective", route_method=method,
                confidence_level=CONFIDENCE_AUDIT, support_kind="audit",
                decision="audit", answer_text=rr1.render_reflective_plan(plan),
                detail={"audit_reason": plan.audit_reason, "branch_support_kind": "missing_knowledge"},
            )
        return self._qa(question, route_method=f"{method}->qa_fallback", branch="qa")

    def _verified(self, question, method, plan) -> IntegratedAnswer:
        return IntegratedAnswer(
            question=question, branch="reflective", route_method=method,
            confidence_level=CONFIDENCE_SPECULATIVE_VERIFIED,
            support_kind="speculative_inference", decision="answer",
            answer_text=self._render_verified_for_surface(plan),
            # The concise surface deliberately does not dump every graph edge;
            # the complete construction-time provenance remains inspectable.
            detail={"rule": plan.rule, "reasoning_trace": plan.to_dict()},
        )

    @staticmethod
    def _render_verified_for_surface(plan) -> str:
        """Render a human answer, not an unbounded graph dump.

        The rule and its full conclusion set are unchanged and are retained in
        ``detail.reasoning_trace``.  This presentation layer summarizes a
        counterfactual around the affected entity and names at most two of its
        directly recorded capabilities.
        """
        if plan.rule != "counterfactual_removal" or plan.step is None:
            return rr1.render_reflective_plan(plan)

        focal = plan.step.premises[0]
        affected = [edge for edge in plan.step.conclusion_facts if edge.s == focal.o]
        capability_order = {"develops": 0, "produces": 1, "provides": 2}
        affected.sort(key=lambda edge: (capability_order.get(edge.p, 9), edge.p, edge.o))
        examples = affected[:2]
        if examples:
            gerunds = {"develops": "developing", "produces": "producing", "provides": "providing"}
            rendered = " and ".join(
                f"{gerunds.get(edge.p, edge.predicate.replace('_', ' '))} {edge.object}"
                for edge in examples
            )
            consequence = (
                f"Its recorded activities — including {rendered} — would then also be in question."
            )
        else:
            consequence = "Its recorded facts would then be in question."
        return (
            f"If {focal.subject} had not {focal.predicate.replace('_', ' ')} {focal.object}, "
            f"the existence of {focal.object} in this evidence slice would be in question. "
            f"{consequence} This is a speculative inference, not a stored fact."
        )

    @staticmethod
    def _is_degenerate_bridge(plan) -> bool:
        """True when an abduction plan's two premises share the same object — the
        undirected-path artifact that conflates a shared-attribute sibling pair
        with a directed bridge (renders as 'X develops O, which develops O')."""
        step = plan.step
        if step is None or len(step.premises) < 2:
            return False
        return step.premises[0].o == step.premises[1].o

    def _counterfactual_from_entities(self, ents):
        """Find a founding/existence edge touching the named entities and build a
        counterfactual over it via the proven rule (which self-gates)."""
        norm = {rr1._norm(e) for e in ents}
        best = None
        for e in self._edges:
            if e.p not in rr1.EXISTENCE_CONFERRING:
                continue
            both = e.s in norm and e.o in norm
            one = e.s in norm or e.o in norm
            if both:
                return rr1.counterfactual_removal(self._edges, e.subject, e.predicate, e.object)
            if one and best is None:
                best = e
        if best is not None:
            return rr1.counterfactual_removal(self._edges, best.subject, best.predicate, best.object)
        return None

    def _constrained(self, question: str, method: str) -> IntegratedAnswer:
        subject = self._primary_entity(question)
        if subject is not None:
            spec = cc1.select_facts(self._overlay_items, subject, n=3)
            if spec.n >= 2:
                text = cc1.generate_constrained(spec)
                return IntegratedAnswer(
                    question=question, branch="constrained_creative", route_method=method,
                    confidence_level=CONFIDENCE_GROUNDED_GENERATION,
                    support_kind="grounded_generation", decision="answer",
                    answer_text=text,
                    detail={"subject": subject, "n_facts": spec.n},
                )
        # No usable subject/facts -> conservative QA default.
        return self._qa(question, route_method=f"{method}->qa_fallback", branch="qa")

    def _pure_creative(self, question: str, method: str) -> IntegratedAnswer:
        # The router has already established an explicit, safe pure-creative
        # intent.  Invoke the existing generator directly: routing it through
        # the older QA intent classifier a second time would wrongly reject
        # valid paraphrases such as "Tell a fictional story …".
        text = generate_creative(question, seed="integrated_router")
        if text:
            return IntegratedAnswer(
                question=question, branch="pure_creative", route_method=method,
                confidence_level=CONFIDENCE_CREATIVE, support_kind="creative_generated",
                decision="answer", answer_text=text,
                detail={"source_system": "creative_generator", "branch_support_kind": "creative_generated"},
            )
        return self._qa(question, route_method=f"{method}->qa_fallback", branch="qa")

    # ── helpers ──────────────────────────────────────────────────────────────── #

    def _distinct_entities(self, question: str) -> list[str]:
        seen, out = set(), []
        for _surface, canonical, _s, _e in self._surface_index.find_in_text(question):
            if canonical and canonical.lower() not in seen:
                seen.add(canonical.lower())
                out.append(canonical)
        return out

    def _primary_entity(self, question: str) -> str | None:
        ents = self._distinct_entities(question)
        return ents[0] if ents else None
