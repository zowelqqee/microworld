"""Cross-page answer renderer v1.

Renders a CrossPagePlan into a compact, source-aware, honest answer string.
Paths are shown explicitly. Weak links are always flagged as weak/non-factual.
Source-qualified facts are always flagged as volatile/recheck. Deterministic.
No ML, no network.
"""

from __future__ import annotations

from worldpgt.cross_page_qa.types import CrossPagePlan, PathEdge

_AUDIT_PREFIX = "I cannot answer from the wiki overlay because "


def render(plan: CrossPagePlan) -> str:
    t = plan.render_template
    a = plan.render_args
    if t == "audit":
        return _AUDIT_PREFIX + _lc(str(a.get("reason", plan.audit_reason or "unsupported")))
    if t == "connection_stable":
        return _render_connection_stable(a)
    if t == "connection_source_qualified":
        return _render_connection_source(a)
    if t == "connection_weak":
        return _render_connection_weak(a)
    if t == "relation_chain":
        return _render_relation_chain(a)
    if t == "source_qualified_list":
        return _render_source_list(a)
    if t == "weak_link_policy":
        return _render_weak_policy(a)
    return _AUDIT_PREFIX + "the question is unsupported."


# ------------------------------------------------------------------
# Edge phrasing (always canonical direction — never inverts founders)
# ------------------------------------------------------------------


def _edge_phrase(e: PathEdge) -> str:
    pred = e.predicate
    if pred == "leader_of":
        return f"{e.subject} is linked to {e.object} through leadership"
    if pred == "known_for":
        return f"{e.subject} is known for {e.object}"
    if pred == "founded":
        return f"{e.subject} founded {e.object}"
    if pred == "produces":
        return f"{e.subject} produces {e.object}"
    if pred == "develops":
        return f"{e.subject} develops {e.object}"
    if pred == "publishes":
        return f"{e.subject} publishes {e.object}"
    return f"{e.subject} is linked to {e.object} via {pred}"


def _edge_arrow(e: PathEdge) -> str:
    return f"{e.subject} → {e.predicate} → {e.object}"


def _render_connection_stable(a: dict) -> str:
    edges: list[PathEdge] = a["edges"]
    nodes: list[str] = a["nodes"]
    a_disp, b_disp = a["a"], a["b"]
    phrases = [_edge_phrase(e) for e in edges]
    stabilities = {e.stability for e in edges}
    qual = "semi-stable" if "semi_stable" in stabilities else "stable"
    if len(edges) == 1:
        return (
            f"{a_disp} is connected to {b_disp}: {phrases[0]}. "
            f"This is an explicit {qual} relation."
        )
    intermediates = nodes[1:-1]
    through = ", ".join(intermediates)
    body = _join_clauses(phrases)
    return (
        f"{a_disp} is connected to {b_disp} through {through}: "
        f"{body}. That is an explicit {qual} relation path of "
        f"{len(edges)} steps."
    )


def _render_connection_source(a: dict) -> str:
    facts = a["facts"]
    role = a["role"]
    a_disp, b_disp = a["a"], a["b"]
    examples = "; ".join(
        f"{f['subject']} {f['object']} according to {f['source_name']} as of {f['as_of']}"
        for f in facts[:3]
    )
    if role == "source":
        lead = (
            f"{a_disp} is connected to {b_disp} as a cited source of "
            f"source-qualified estimates"
        )
    else:
        lead = (
            f"{a_disp} is connected to {b_disp} through a source-qualified "
            f"estimate"
        )
    return (
        f"{lead}: {examples}. These are volatile, source-qualified facts that "
        f"require rechecking, not stable factual relations."
    )


def _render_connection_weak(a: dict) -> str:
    a_disp, b_disp = a["a"], a["b"]
    link = a["link"]
    return (
        f"{a_disp} and {b_disp} are connected only by a weak contextual "
        f"link (on the {link['source_page']} page). This is a weak contextual link, not a "
        f"stable factual relation, so it does not prove a factual relationship."
    )


def _render_relation_chain(a: dict) -> str:
    edges: list[PathEdge] = a["edges"]
    a_disp, b_disp = a["a"], a["b"]
    chain = "; ".join(_edge_arrow(e) for e in edges)
    return (
        f"The overlay has an explicit relation path from {a_disp} to {b_disp}: "
        f"{chain}. Each step is an explicit overlay relation."
    )


def _render_source_list(a: dict) -> str:
    facts = a["facts"]
    if not facts:
        return "The overlay contains no source-qualified volatile facts."
    items = "; ".join(
        f"{f['subject']} — {f['object']} according to {f['source_name']} as of {f['as_of']}"
        for f in facts
    )
    return (
        f"The overlay has {len(facts)} volatile source-qualified facts that require "
        f"rechecking: {items}. These are source-qualified estimates, not stable facts."
    )


def _render_weak_policy(a: dict) -> str:
    examples = a.get("examples", [])
    total = a.get("total", 0)
    ex = ""
    if examples:
        ex_str = "; ".join(
            f"{c['source_page']} → {c['target']}" for c in examples
        )
        ex = f" Examples: {ex_str}."
    return (
        f"Weak context links (weak_context_only trust) are contextual "
        f"mentions, not stable factual relations. They cannot prove a factual claim. "
        f"There are currently {total} such weak context links.{ex}"
    )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _join_clauses(clauses: list[str]) -> str:
    if not clauses:
        return ""
    if len(clauses) == 1:
        return clauses[0]
    if len(clauses) == 2:
        return f"{clauses[0]}, and {clauses[1]}"
    return ", ".join(clauses[:-1]) + f", and {clauses[-1]}"


def _lc(s: str) -> str:
    return s[0].lower() + s[1:] if s else s
