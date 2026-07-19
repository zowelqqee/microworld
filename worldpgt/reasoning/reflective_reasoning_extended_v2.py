"""Isolated reflective-reasoning EXTENSION (v2) — the lower-confidence class.

Adds ONE new composition pattern beyond the two proven rules, at an explicitly
LOWER confidence level: ``speculative_extended``. It is kept in a separate module
and a separate ``support_kind`` on purpose — it must never be merged with, or
dilute, the proven ``speculative_inference`` rules in
``reflective_reasoning_v1`` (which this module imports read-only and does not
modify). See ``artifacts/reflective_reasoning_core_v2/design_and_pilot_report.md``.

Pattern A — CO-ATTRIBUTION (gate pilot: 29/29 defensible as a weak association):
    X --pred--> O  and  Y --pred--> O  (same KINSHIP predicate)
    => "X and Y might be related; both <pred> <O>."

The KINSHIP filter is the mechanism (never the naive rule): a shared object counts
only when reached by a capability/authorship/creation predicate. Distribution
predicates (published_by, located_in, …) create spurious cliques and are excluded.

Pattern B (analogical property-transfer) was tested and REJECTED as structurally
unsound; it is intentionally not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable

from worldpgt.reasoning.reflective_reasoning_v1 import Edge, load_edges, _norm

# Sharing an object via these implies genuine kinship (peers / co-creators /
# co-founders). Everything else — notably distribution/location/valuation — is
# excluded because a shared object there is a channel or coincidence, not a link.
KINSHIP_PREDICATES = frozenset({
    "develops", "produces", "created_by", "founded", "developed_by", "provides",
})

SUPPORT_KIND = "speculative_extended"


@dataclass
class CoAttribution:
    """One co-attribution inference: two peers sharing an object via a kinship
    predicate. Both source edges are retained (inspectable premises)."""

    x: str
    y: str
    predicate: str
    shared_object: str
    x_edge: Edge
    y_edge: Edge

    def to_dict(self) -> dict:
        return {
            "support_kind": SUPPORT_KIND,
            "x": self.x,
            "y": self.y,
            "predicate": self.predicate,
            "shared_object": self.shared_object,
            "premise_evidence_ids": [self.x_edge.evidence_id, self.y_edge.evidence_id],
        }


@dataclass
class ExtendedPlan:
    decision: str  # "speculative_extended" | "audit"
    steps: list[CoAttribution] = field(default_factory=list)
    audit_reason: str | None = None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision,
            "support_kind": SUPPORT_KIND if self.decision != "audit" else "missing_knowledge",
            "steps": [s.to_dict() for s in self.steps],
            "audit_reason": self.audit_reason,
        }


def _kinship_edges_to(edges: list[Edge], obj_norm: str) -> dict[str, list[Edge]]:
    """predicate -> edges whose object == obj_norm and predicate is a kinship one."""
    out: dict[str, list[Edge]] = {}
    for e in edges:
        if e.o == obj_norm and e.p in KINSHIP_PREDICATES:
            out.setdefault(e.p, []).append(e)
    return out


def co_attribution_for_pair(edges: list[Edge], x: str, y: str) -> ExtendedPlan:
    """"Why might X and Y be related?" via a shared kinship attribute. Fires only
    when X and Y share an object reached by the SAME kinship predicate."""
    xn, yn = _norm(x), _norm(y)
    if xn == yn:
        return ExtendedPlan("audit", audit_reason="X and Y are the same entity")
    steps: list[CoAttribution] = []
    # objects X reaches via a kinship predicate
    x_by_obj: dict[tuple[str, str], Edge] = {
        (e.o, e.p): e for e in edges if e.s == xn and e.p in KINSHIP_PREDICATES
    }
    for e in edges:
        if e.s == yn and e.p in KINSHIP_PREDICATES and (e.o, e.p) in x_by_obj:
            xe = x_by_obj[(e.o, e.p)]
            steps.append(CoAttribution(
                x=xe.subject, y=e.subject, predicate=xe.predicate,
                shared_object=xe.object, x_edge=xe, y_edge=e,
            ))
    if not steps:
        return ExtendedPlan("audit", audit_reason=(
            "no shared attribute via a kinship predicate; not a defensible "
            "co-attribution (distribution/location links are excluded)"
        ))
    return ExtendedPlan("speculative_extended", steps=steps)


def discover_co_attributions(overlay_items: Iterable[dict]) -> list[CoAttribution]:
    """Enumerate every defensible co-attribution pair in an overlay slice (the
    gated, deduped set — the naive cross-product is never returned)."""
    edges = load_edges(overlay_items)
    by_obj: dict[str, dict[str, list[Edge]]] = {}
    objs = {e.o for e in edges}
    for o in objs:
        km = _kinship_edges_to(edges, o)
        if any(len(v) >= 2 for v in km.values()):
            by_obj[o] = km
    seen: set[tuple] = set()
    out: list[CoAttribution] = []
    for o, km in by_obj.items():
        for pred, es in km.items():
            for a, b in combinations(es, 2):
                if a.s == b.s:
                    continue
                key = tuple(sorted([a.s, b.s])) + (o, pred)
                if key in seen:
                    continue
                seen.add(key)
                out.append(CoAttribution(
                    x=a.subject, y=b.subject, predicate=pred,
                    shared_object=a.object, x_edge=a, y_edge=b,
                ))
    return out


def render_extended(step: CoAttribution) -> str:
    """Structure-driven lower-confidence framing (mirrors uncertainty_note)."""
    return (
        f"{step.x} and {step.y} are not directly linked in the verified relations; "
        f"they are connected here only because both {step.predicate} {step.shared_object}. "
        "Treat this as a broader, less-tested inference than the system's core reasoning "
        "— a similarity, not a stored or directly-derived fact."
    )
