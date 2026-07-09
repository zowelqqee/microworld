"""Cross-page answer planner v1.

Builds a small undirected graph from overlay *relations* and finds the shortest
safe path between two endpoints. Definitions/source-facts/weak-context-links are
used only in their dedicated, safe roles:

* relation edges  -> stable / semi-stable path proof (max length 3, prefer <=2)
* source facts    -> source-qualified / volatile answers only
* weak links      -> weak-contextual explanations only (never stable proof)

If no safe explicit path exists, the planner audits. Deterministic. No ML, no
network.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Optional

from worldpgt.cross_page_qa.types import (
    CrossPageEvidence,
    CrossPagePlan,
    CrossPageQuestion,
    PathEdge,
)
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_MAX_PATH_HOPS = 3

_NET_WORTH_TERMS = frozenset({
    "net worth", "net-worth", "net worth estimate", "estimated net worth",
    "networth", "wealth", "net worth claim", "net worth claims",
})

_AUDIT_CURRENT = (
    "this question asks for current/real-time data that is not available as an "
    "accepted fact in the overlay."
)
_AUDIT_UNIVERSAL = (
    "this question asks for a universal generalization that the overlay does not "
    "support."
)
_AUDIT_NO_PATH = (
    "the overlay has no explicit safe relation path or direct contextual link "
    "between these entities."
)
_AUDIT_UNKNOWN_ENDPOINT = (
    "one or both entities are not present in the overlay."
)
_AUDIT_DEFAULT = (
    "the overlay has no explicit safe path supporting this question."
)


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.rstrip("?.").strip()


def _fold(s: str) -> str:
    """Normalized + simple depluralization (trailing 's')."""
    n = _norm(s)
    if len(n) > 3 and n.endswith("s"):
        return n[:-1]
    return n


def _match(a: str, b: str) -> bool:
    if _norm(a) == _norm(b):
        return True
    return _fold(a) == _fold(b)


class CrossPageAnswerPlanner:
    def __init__(self, provider: WikiMemoryOverlayProvider) -> None:
        self._provider = provider
        self._relations = provider.all_relations()
        self._context_links = provider.all_context_links()
        self._source_facts = provider.all_source_facts()
        self._entities = provider.all_entities()
        self._definitions = provider.all_definitions()
        self._adj, self._display = self._build_graph()
        self._known: set[str] = self._build_known_nodes()
        self._known_fold: set[str] = {_fold(k) for k in self._known}

    def _is_known(self, x: str) -> bool:
        cands = {_norm(x), _fold(x)}
        return bool(cands & (self._known | self._known_fold))

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------

    def _build_graph(self):
        adj: dict[str, list[tuple[str, PathEdge]]] = {}
        display: dict[str, str] = {}
        for r in self._relations:
            subj, obj, pred = r.get("subject", ""), r.get("object", ""), r.get("predicate", "")
            if not subj or not obj:
                continue
            ns, no = _norm(subj), _norm(obj)
            display.setdefault(ns, subj)
            display.setdefault(no, obj)
            edge = PathEdge(subject=subj, predicate=pred, object=obj,
                            stability=r.get("stability", "semi_stable"))
            adj.setdefault(ns, []).append((no, edge))
            adj.setdefault(no, []).append((ns, edge))
        return adj, display

    def _build_known_nodes(self) -> set[str]:
        known: set[str] = set(self._adj.keys())
        for e in self._entities:
            known.add(_norm(e.get("label", "")))
            for a in e.get("aliases", []) or []:
                known.add(_norm(a))
        for d in self._definitions:
            known.add(_norm(d.get("subject", "")))
        for c in self._context_links:
            known.add(_norm(c.get("source_page", "")))
            known.add(_norm(c.get("target", "")))
        return {k for k in known if k}

    # ------------------------------------------------------------------
    # Path finding (shortest, undirected over relation edges)
    # ------------------------------------------------------------------

    def _find_path(self, a: str, b: str) -> Optional[list[PathEdge]]:
        start = _norm(a)
        if start not in self._adj:
            return None
        # Trivial: A itself matches B.
        if _match(start, b):
            return []
        parent: dict[str, tuple[str, PathEdge]] = {}
        visited = {start}
        queue: deque[tuple[str, int]] = deque([(start, 0)])
        goal: Optional[str] = None
        while queue:
            node, depth = queue.popleft()
            if depth >= _MAX_PATH_HOPS:
                continue
            for neighbor, edge in self._adj.get(node, []):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                parent[neighbor] = (node, edge)
                if _match(neighbor, b):
                    goal = neighbor
                    queue.clear()
                    break
                queue.append((neighbor, depth + 1))
        if goal is None:
            return None
        # Reconstruct edges from start -> goal.
        edges: list[PathEdge] = []
        cur = goal
        while cur in parent:
            prev, edge = parent[cur]
            edges.append(edge)
            cur = prev
        edges.reverse()
        return edges

    def _node_sequence(self, a: str, edges: list[PathEdge]) -> list[str]:
        """Display names of nodes along the path, starting at A's display."""
        seq = [self._display.get(_norm(a), a)]
        cur = _norm(a)
        for edge in edges:
            ns, no = _norm(edge.subject), _norm(edge.object)
            nxt = no if cur == ns else ns
            seq.append(self._display.get(nxt, nxt))
            cur = nxt
        return seq

    # ------------------------------------------------------------------
    # Weak / source bridges
    # ------------------------------------------------------------------

    def _direct_weak_link(self, a: str, b: str) -> Optional[dict]:
        for c in self._context_links:
            sp, tg, sf = c.get("source_page", ""), c.get("target", ""), c.get("surface", "")
            if _match(sp, a) and (_match(tg, b) or _match(sf, b)):
                return c
            if _match(sp, b) and (_match(tg, a) or _match(sf, a)):
                return c
        return None

    def _source_fact_bridge(self, a: str, b: str) -> Optional[tuple[list[dict], str]]:
        """Return (facts, role) if A links to net-worth-style facts; else None."""
        b_is_net_worth = _norm(b) in _NET_WORTH_TERMS or "net worth" in _norm(b)
        a_is_net_worth = _norm(a) in _NET_WORTH_TERMS or "net worth" in _norm(a)
        if not (b_is_net_worth or a_is_net_worth):
            return None
        other = a if b_is_net_worth else b
        # A is the cited source of estimates.
        as_source = [f for f in self._source_facts if _match(f.get("source_name", ""), other)]
        if as_source:
            return as_source, "source"
        # A is the subject of an estimate.
        as_subject = [f for f in self._source_facts if _match(f.get("subject", ""), other)]
        if as_subject:
            return as_subject, "subject"
        return None

    # ------------------------------------------------------------------
    # Public planning
    # ------------------------------------------------------------------

    def plan(self, q: CrossPageQuestion) -> CrossPagePlan:
        if q.intent == "unsupported_or_too_far":
            reason = _AUDIT_DEFAULT
            if q.is_current_query:
                reason = _AUDIT_CURRENT
            elif q.is_universal_query:
                reason = _AUDIT_UNIVERSAL
            return self._audit(q, reason)
        if q.intent == "source_qualified_list":
            return self._plan_source_list(q)
        if q.intent == "weak_link_list_or_policy":
            return self._plan_weak_policy(q)
        if q.intent == "connection_path":
            return self._plan_connection(q)
        if q.intent == "relation_chain":
            return self._plan_chain(q)
        return self._audit(q, _AUDIT_DEFAULT)

    def _plan_connection(self, q: CrossPageQuestion) -> CrossPagePlan:
        a, b = q.source or "", q.target or ""
        if not a or not b:
            return self._audit(q, _AUDIT_UNKNOWN_ENDPOINT)
        if not self._is_known(a) or not self._is_known(b):
            return self._audit(q, _AUDIT_UNKNOWN_ENDPOINT)

        # 1. Stable / semi-stable relation path.
        edges = self._find_path(a, b)
        if edges:
            ev = CrossPageEvidence(relation_edges_used=[
                f"{e.subject}--{e.predicate}-->{e.object}" for e in edges])
            return CrossPagePlan(
                q, "answer", None, ev, "connection_stable",
                {"a": self._display.get(_norm(a), a),
                 "b": self._display.get(_norm(b), b),
                 "nodes": self._node_sequence(a, edges), "edges": edges},
                confidence=0.9,
            )

        # 2. Source-qualified bridge (net-worth style only).
        bridge = self._source_fact_bridge(a, b)
        if bridge is not None:
            facts, role = bridge
            ev = CrossPageEvidence(source_facts_used=[
                f"{f['subject']}:{f['predicate']}:{f['source_name']}" for f in facts])
            return CrossPagePlan(
                q, "answer", None, ev, "connection_source_qualified",
                {"a": a, "b": b, "facts": facts, "role": role}, confidence=0.85,
            )

        # 3. Direct weak contextual link (explicitly weak, never stable).
        weak = self._direct_weak_link(a, b)
        if weak is not None:
            ev = CrossPageEvidence(weak_links_used=[
                f"{weak['source_page']}->{weak['target']}"])
            return CrossPagePlan(
                q, "answer", None, ev, "connection_weak",
                {"a": a, "b": b, "link": weak}, confidence=0.6,
            )

        return self._audit(q, _AUDIT_NO_PATH)

    def _plan_chain(self, q: CrossPageQuestion) -> CrossPagePlan:
        a, b = q.source or "", q.target or ""
        if not a or not b:
            return self._audit(q, _AUDIT_UNKNOWN_ENDPOINT)
        if not self._is_known(a) or not self._is_known(b):
            return self._audit(q, _AUDIT_UNKNOWN_ENDPOINT)
        edges = self._find_path(a, b)
        if not edges:
            # relation_chain answers only with an explicit relation path.
            return self._audit(q, _AUDIT_NO_PATH)
        ev = CrossPageEvidence(relation_edges_used=[
            f"{e.subject}--{e.predicate}-->{e.object}" for e in edges])
        return CrossPagePlan(
            q, "answer", None, ev, "relation_chain",
            {"a": self._display.get(_norm(a), a),
             "b": self._display.get(_norm(b), b), "edges": edges},
            confidence=0.9,
        )

    def _plan_source_list(self, q: CrossPageQuestion) -> CrossPagePlan:
        facts = list(self._source_facts)
        ev = CrossPageEvidence(source_facts_used=[
            f"{f['subject']}:{f['predicate']}:{f['source_name']}" for f in facts])
        return CrossPagePlan(
            q, "answer", None, ev, "source_qualified_list",
            {"facts": facts}, confidence=0.95,
        )

    def _plan_weak_policy(self, q: CrossPageQuestion) -> CrossPagePlan:
        examples = self._context_links[:3]
        ev = CrossPageEvidence(weak_links_used=[
            f"{c['source_page']}->{c['target']}" for c in examples])
        return CrossPagePlan(
            q, "answer", None, ev, "weak_link_policy",
            {"examples": examples, "total": len(self._context_links)},
            confidence=0.95,
        )

    def _audit(self, q: CrossPageQuestion, reason: str) -> CrossPagePlan:
        return CrossPagePlan(
            q, "audit", reason, CrossPageEvidence(), "audit",
            {"reason": reason, "question": q.question}, confidence=0.0,
        )
