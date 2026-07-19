"""Isolated reflective-reasoning experiment — construction-time-labelled speculation.

This module adds a *third* class of answer plan alongside grounded answer and
audit: a **speculative inference** whose grounded/speculative label is attached
**at construction time** (the planner knows the origin of every step), never
recovered from finished text. It is the tractable counterpart to the closed
``informed_reflection`` branch, whose post-hoc text classification failed because
surface markers cannot recover scope.

Two explicit, inspectable inference rules — both cleared their gate pilots
(``artifacts/reflective_reasoning_core_v1/pilot_report.md`` and
``pilot_abduction_report.md``). Each ships with the *structural filter* the pilot
validated, and each prefers to **decline** (audit) rather than emit a
non-defensible conclusion:

* ``counterfactual_removal`` — "what if S had not P O": fires only when P is
  existence-conferring and O is itself a graph entity, then reports the facts
  about O that would be in question.
* ``abduction_path_explanation`` — "why might S be associated with O": fires only
  on a genuine 2-hop bridge S -> M -> O (defers to grounded QA on a direct edge;
  declines on spurious 3-hop-through-shared-entity paths).

Deliberately no API/server imports: callers pass the evidence slice, so this is
safe for A/B benchmarks and never alters the production grounded route. Grounded
accuracy is preserved by construction — the grounded planner is not touched.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

Rule = Literal["counterfactual_removal", "abduction_path_explanation"]
Decision = Literal["speculative", "grounded_deferral", "audit"]

# Existence-conferring predicates: removing such a relation plausibly threatens
# the existence of the object entity, so downstream facts about it are defensibly
# "in question". Refined by the counterfactual pilot (activity predicates such as
# ``develops``/``produces`` over generic products yielded no defensible set).
EXISTENCE_CONFERRING = frozenset({
    "founded", "founded_by", "created_by", "developed_by",
    "product_of", "construction_started",
})


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


# --------------------------------------------------------------------------- #
# Evidence
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Edge:
    subject: str
    predicate: str
    object: str
    evidence_id: str

    @property
    def s(self) -> str:
        return _norm(self.subject)

    @property
    def p(self) -> str:
        return _norm(self.predicate)

    @property
    def o(self) -> str:
        return _norm(self.object)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "evidence_id": self.evidence_id,
        }


def load_edges(overlay_items: Iterable[dict]) -> list[Edge]:
    """Extract overlay_relation edges from an overlay slice (same shape the
    grounded planner consumes). Non-relation items are ignored."""
    edges: list[Edge] = []
    for i, item in enumerate(overlay_items or ()):
        if not isinstance(item, dict) or item.get("overlay_type") != "overlay_relation":
            continue
        subject = str(item.get("subject") or "").strip()
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        if not (subject and predicate and obj):
            continue
        eid = str(item.get("evidence_id") or item.get("id") or f"edge:{i}")
        edges.append(Edge(subject, predicate, obj, eid))
    return edges


# --------------------------------------------------------------------------- #
# Plan structures
# --------------------------------------------------------------------------- #

@dataclass
class SpeculativeStep:
    """A single inference: grounded premises -> named rule -> speculative
    conclusion. Every element is inspectable (premises carry evidence ids)."""

    rule: Rule
    premises: list[Edge] = field(default_factory=list)
    conclusion_facts: list[Edge] = field(default_factory=list)
    bridge_node: str | None = None  # abduction: the intermediate M in S -> M -> O

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "premise_evidence_ids": [e.evidence_id for e in self.premises],
            "premises": [e.to_dict() for e in self.premises],
            "conclusion_evidence_ids": [e.evidence_id for e in self.conclusion_facts],
            "conclusion_facts": [e.to_dict() for e in self.conclusion_facts],
            "bridge_node": self.bridge_node,
        }


@dataclass
class ReflectivePlan:
    """Inspectable output. ``decision`` is known at construction time.

    * ``speculative``        — a defensible speculative_step was built.
    * ``grounded_deferral``  — a direct grounded fact answers this; the grounded
      planner should handle it (abduction must not speculate over a stored fact).
    * ``audit``              — no defensible construction; decline honestly.
    """

    question: str
    rule: Rule
    decision: Decision
    step: SpeculativeStep | None = None
    audit_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "rule": self.rule,
            "decision": self.decision,
            "support_kind": (
                "speculative_inference" if self.decision == "speculative"
                else "grounded" if self.decision == "grounded_deferral"
                else "missing_knowledge"
            ),
            "step": self.step.to_dict() if self.step else None,
            "audit_reason": self.audit_reason,
        }


# --------------------------------------------------------------------------- #
# Adjacency helpers
# --------------------------------------------------------------------------- #

def _subjects(edges: list[Edge]) -> set[str]:
    return {e.s for e in edges}


def _find_edge(edges: list[Edge], s: str, p: str, o: str) -> Edge | None:
    sn, pn, on = _norm(s), _norm(p), _norm(o)
    for e in edges:
        if e.s == sn and e.p == pn and e.o == on:
            return e
    return None


def _direct_edges(edges: list[Edge], s: str, o: str) -> list[Edge]:
    sn, on = _norm(s), _norm(o)
    return [e for e in edges if (e.s == sn and e.o == on) or (e.s == on and e.o == sn)]


# --------------------------------------------------------------------------- #
# Rule 1 — counterfactual removal
# --------------------------------------------------------------------------- #

def counterfactual_removal(
    edges: list[Edge], subject: str, predicate: str, obj: str
) -> ReflectivePlan:
    """"What if <subject> had not <predicate> <obj>?" (pilot-validated filter).

    Fires only when the focal predicate is existence-conferring and the object is
    itself a graph entity; then the conclusion is exactly the facts referencing
    that object node (they would be in question if the entity did not exist).
    Otherwise audits.
    """
    question = f"What if {subject} had not {predicate} {obj}?"
    focal = _find_edge(edges, subject, predicate, obj)
    if focal is None:
        return ReflectivePlan(question, "counterfactual_removal", "audit",
                              audit_reason="focal fact is not in the evidence slice")

    if focal.p not in EXISTENCE_CONFERRING:
        return ReflectivePlan(
            question, "counterfactual_removal", "audit",
            audit_reason=(
                f"predicate '{predicate}' is not existence-conferring; removing it "
                "does not defensibly threaten downstream facts"
            ),
        )

    obj_norm = focal.o
    if obj_norm not in _subjects(edges):
        return ReflectivePlan(
            question, "counterfactual_removal", "audit",
            audit_reason=(
                f"object '{obj}' is not itself a graph entity (has no downstream "
                "facts), so there is nothing defensible to put in question"
            ),
        )

    # Admit connected facts that reference the object node (excluding the focal).
    conclusion = [
        e for e in edges
        if e.evidence_id != focal.evidence_id and (e.s == obj_norm or e.o == obj_norm)
    ]
    if not conclusion:
        return ReflectivePlan(
            question, "counterfactual_removal", "audit",
            audit_reason="no other facts reference the object entity",
        )
    step = SpeculativeStep(
        rule="counterfactual_removal", premises=[focal], conclusion_facts=conclusion,
    )
    return ReflectivePlan(question, "counterfactual_removal", "speculative", step=step)


# --------------------------------------------------------------------------- #
# Rule 2 — why-might abduction (2-hop bridge)
# --------------------------------------------------------------------------- #

def abduction_explanation(edges: list[Edge], subject: str, obj: str) -> ReflectivePlan:
    """"Why might <subject> be associated with <obj>?" (pilot-validated filter).

    * A direct edge S-O -> grounded_deferral (not a speculation).
    * A 2-hop bridge S -> M -> O -> speculative explanation via M.
    * Otherwise (no bridge, or only spurious 3-hop-through-shared-entity) -> audit.
    """
    question = f"Why might {subject} be associated with {obj}?"
    sn, on = _norm(subject), _norm(obj)

    if _direct_edges(edges, subject, obj):
        return ReflectivePlan(
            question, "abduction_path_explanation", "grounded_deferral",
            audit_reason="a direct grounded fact links these; not a speculation",
        )

    # 2-hop bridges: S -?- M -?- O (undirected traversal along relations).
    from_s: dict[str, list[Edge]] = {}
    for e in edges:
        if e.s == sn:
            from_s.setdefault(e.o, []).append(e)
        elif e.o == sn:
            from_s.setdefault(e.s, []).append(e)

    best: SpeculativeStep | None = None
    for mid, first_edges in from_s.items():
        if mid == sn or mid == on:
            continue
        second_edges = [
            e for e in edges
            if (e.s == mid and e.o == on) or (e.o == mid and e.s == on)
        ]
        if not second_edges:
            continue
        first = _prefer(first_edges)
        second = _prefer(second_edges)
        # Bridge node label as stored (from whichever endpoint is the mid node).
        bridge_label = first.object if first.s == sn else first.subject
        candidate = SpeculativeStep(
            rule="abduction_path_explanation",
            premises=[first, second],
            conclusion_facts=[],
            bridge_node=bridge_label,
        )
        # Prefer an existence/leadership first hop for a stronger explanation.
        if best is None or _hop_rank(first) < _hop_rank(best.premises[0]):
            best = candidate

    if best is None:
        return ReflectivePlan(
            question, "abduction_path_explanation", "audit",
            audit_reason="no 2-hop explanatory bridge; only spurious or no path",
        )
    return ReflectivePlan(question, "abduction_path_explanation", "speculative", step=best)


_HOP_PRIORITY = {"founded": 0, "founded_by": 0, "created_by": 0, "leader_of": 1,
                 "developed_by": 1, "known_for": 2}


def _hop_rank(edge: Edge) -> int:
    return _HOP_PRIORITY.get(edge.p, 3)


def _prefer(edges: list[Edge]) -> Edge:
    return sorted(edges, key=lambda e: (_hop_rank(e), e.evidence_id))[0]


# --------------------------------------------------------------------------- #
# Conservative question admission (callers may also build queries directly)
# --------------------------------------------------------------------------- #

_WHATIF_RE = re.compile(
    r"^\s*what if\s+(?P<s>.+?)\s+had not\s+(?P<p>\S+)\s+(?P<o>.+?)\s*\??\s*$",
    re.IGNORECASE,
)
_WHYMIGHT_RE = re.compile(
    r"^\s*why might\s+(?P<s>.+?)\s+be associated with\s+(?P<o>.+?)\s*\??\s*$",
    re.IGNORECASE,
)


def reflect(question: str, overlay_items: Iterable[dict]) -> ReflectivePlan | None:
    """Route a natural-language reflective question to a rule. Returns None if the
    question matches no supported reflective pattern (an admission gate, not NLU)."""
    edges = load_edges(overlay_items)
    m = _WHATIF_RE.match(question or "")
    if m:
        return counterfactual_removal(edges, m["s"], m["p"], m["o"])
    m = _WHYMIGHT_RE.match(question or "")
    if m:
        return abduction_explanation(edges, m["s"], m["o"])
    return None


# --------------------------------------------------------------------------- #
# Renderer — framing built from plan structure, never guessed from text
# --------------------------------------------------------------------------- #

def render_reflective_plan(plan: ReflectivePlan) -> str:
    if plan.decision == "audit":
        return (
            f"I can't responsibly speculate on that: {plan.audit_reason}."
        )
    if plan.decision == "grounded_deferral":
        return (
            "That is answerable from stored facts, not speculation "
            f"({plan.audit_reason})."
        )
    step = plan.step
    assert step is not None
    if step.rule == "counterfactual_removal":
        focal = step.premises[0]
        affected = "; ".join(
            f"{e.subject} {e.predicate} {e.object}" for e in step.conclusion_facts
        )
        return (
            f"Based on what is known — that {focal.subject} {focal.predicate} "
            f"{focal.object} — one might reason that, had that not been so, the "
            f"following would be in question: {affected}. "
            "(This is a speculative inference, not a stored fact.)"
        )
    # abduction
    first, second = step.premises[0], step.premises[1]
    return (
        f"Based on what is known, one might reason that {first.subject} "
        f"{_lexicalize(first.predicate)} {first.object}, which "
        f"{_lexicalize(second.predicate)} {second.object} — a plausible reason "
        "for the association. (This is a speculative inference, not a stored fact.)"
    )


# Light predicate lexicalization for the speculative renderer (readability only;
# the plan trace keeps the raw predicate). Unknown predicates fall back to the
# underscore-stripped form.
_PREDICATE_PHRASE = {
    "founded": "founded",
    "founded_by": "was founded by",
    "leader_of": "leads",
    "known_for": "is known for",
    "developed_by": "was developed by",
    "develops": "develops",
    "produces": "produces",
    "located_in": "is located in",
    "owned_by": "is owned by",
    "part_of": "is part of",
}


def _lexicalize(predicate: str) -> str:
    return _PREDICATE_PHRASE.get(_norm(predicate), (predicate or "").replace("_", " "))
