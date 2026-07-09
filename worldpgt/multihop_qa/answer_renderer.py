"""Multi-hop answer renderer for Multi-hop QA v1.

Produces natural-language answers for explicit 2-hop relation chains.
No ML. No network. Pure string formatting.
"""
from __future__ import annotations

from worldpgt.multihop_qa.types import HopEdge, MultihopPlan
from worldpgt.knowledge.temporal_classification import temporal_caveat, weakest_temporal_class

_PRED_VERB: dict[str, str] = {
    "owned_by": "is owned by",
    "service_of": "is a service of",
    "subsidiary_of": "is a subsidiary of",
    "founded_by": "was founded by",
    "founded": "founded",
    "develops": "develops",
    "produces": "produces",
    "is_a": "is a",
    "leader_of": "is the leader of",
    "known_for": "is known for",
    "developed_by": "was developed by",
    "publishes": "publishes",
}


def _clause(edge: HopEdge) -> str:
    verb = _PRED_VERB.get(edge.predicate, f"has relation '{edge.predicate}' to")
    return f"{edge.subject} {verb} {edge.object}"


def _chain_temporal_caveat(plan: MultihopPlan) -> str:
    if plan.path is None:
        return ""
    classes = [
        tc
        for tc in (plan.path.hop1.temporal_class, plan.path.hop2.temporal_class)
        if tc
    ]
    temporal_class = weakest_temporal_class(classes)
    dates = [plan.path.hop1.as_of, plan.path.hop2.as_of]
    caveat = temporal_caveat(temporal_class, dates)
    return f" {caveat}" if caveat else ""


def render(plan: MultihopPlan) -> str:
    """Render a MultihopPlan to a natural-language string."""
    if plan.decision == "audit":
        reason = plan.audit_reason or "no supported 2-hop chain found"
        return f"[audit] {reason}"

    ct = plan.question.chain_type
    ra = plan.render_args

    if ct == "develops_isa":
        a = ra["a"]
        b = ra["b"]
        category = ra["category"]
        return f"{a} develops {b}, which is a {category}.{_chain_temporal_caveat(plan)}"

    hop1: HopEdge = ra["hop1"]
    hop2: HopEdge = ra["hop2"]
    a = ra["a"]
    b = ra["b"]
    c = ra["c"]

    c1 = _clause(hop1)
    c2 = _clause(hop2)
    return f"{a} is connected to {c} through {b}: {c1}, and {c2}.{_chain_temporal_caveat(plan)}"
