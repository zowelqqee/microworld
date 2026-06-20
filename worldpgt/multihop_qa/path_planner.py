"""Multi-hop path planner for Multi-hop QA v1.

Plans a 2-hop answer from a MultihopQuestion and a RelationGraph.
All hops must come from overlay_relation items. Deterministic. No ML.
"""
from __future__ import annotations

from worldpgt.multihop_qa.path_validator import chain_label, validate_path
from worldpgt.multihop_qa.relation_graph import RelationGraph
from worldpgt.multihop_qa.types import (
    MultihopEvidence,
    MultihopPath,
    MultihopPlan,
    MultihopQuestion,
)

_AUDIT_NO_PATH = "no 2-hop relation path found between these entities in the overlay"
_AUDIT_DIRECT = (
    "this question asks for a direct relation; multi-hop cannot answer using "
    "a transitive chain — single-hop entity QA must handle this directly"
)
_AUDIT_MISSING_HOP = "one or both hops of the requested chain are missing from the overlay"
_AUDIT_UNSUPPORTED = "no recognized multi-hop question pattern matched"
_AUDIT_MISSING_ENDPOINT = "one or both entities are not present in the overlay"


def _make_evidence(path: MultihopPath) -> MultihopEvidence:
    return MultihopEvidence(hops_used=[path.hop1.display(), path.hop2.display()])


def _pick_best(paths: list[MultihopPath]) -> MultihopPath:
    return paths[0]


class MultihopPlanner:
    def __init__(self, graph: RelationGraph) -> None:
        self._graph = graph

    def plan(self, q: MultihopQuestion) -> MultihopPlan:
        ct = q.chain_type

        if q.is_direct_relation or ct == "direct_only":
            return self._audit(q, _AUDIT_DIRECT, "direct_only_audit")

        if ct == "unsupported":
            return self._audit(q, _AUDIT_UNSUPPORTED, "unsupported")

        if ct == "develops_isa":
            return self._plan_develops_isa(q)

        if ct == "through_node":
            return self._plan_through_node(q)

        if ct == "who_through":
            return self._plan_who_through(q)

        if ct == "connection":
            return self._plan_connection(q)

        return self._audit(q, _AUDIT_UNSUPPORTED, "unsupported")

    # ------------------------------------------------------------------ #

    def _plan_connection(self, q: MultihopQuestion) -> MultihopPlan:
        a = q.endpoint_a or ""
        c = q.endpoint_c or ""
        if not a or not c:
            return self._audit(q, _AUDIT_MISSING_ENDPOINT, "missing_endpoint")

        paths = self._graph.find_2hop_paths(a, c)
        reversed_search = False
        if not paths:
            # Try reversed direction: C → B → A. Safety gates apply identically.
            paths = self._graph.find_2hop_paths(c, a)
            reversed_search = bool(paths)

        if not paths:
            return self._audit(q, _AUDIT_NO_PATH, "no_path")

        path = _pick_best(paths)
        valid, reason = validate_path(path, q)
        if not valid:
            return self._audit(q, f"path_blocked: {reason}", "blocked")

        label = chain_label(path.hop1.predicate, path.hop2.predicate)
        # For a reversed search the found path is C→B→A; use the canonical
        # endpoint names from the path so the answer reads "A is connected to C
        # through B" while the edge clauses describe the actual found direction.
        if reversed_search:
            render_a = path.hop2.object   # canonical display name of original A
            render_c = path.hop1.subject  # canonical display name of original C
        else:
            render_a = path.hop1.subject
            render_c = path.hop2.object
        return MultihopPlan(
            question=q, decision="answer", audit_reason=None,
            path=path, chain_label=label,
            render_args={
                "a": render_a, "b": path.via_node, "c": render_c,
                "hop1": path.hop1, "hop2": path.hop2,
            },
            evidence=_make_evidence(path),
        )

    def _plan_develops_isa(self, q: MultihopQuestion) -> MultihopPlan:
        a = q.endpoint_a or ""
        category = q.endpoint_c or ""
        if not a or not category:
            return self._audit(q, _AUDIT_MISSING_ENDPOINT, "missing_endpoint")

        paths = self._graph.find_develops_isa_paths(a, category)
        if not paths:
            return self._audit(q, _AUDIT_MISSING_HOP, "no_develops_isa_path")

        path = _pick_best(paths)
        return MultihopPlan(
            question=q, decision="answer", audit_reason=None,
            path=path, chain_label="develops_then_is_a",
            render_args={
                "a": path.hop1.subject, "b": path.via_node, "category": path.hop2.object,
            },
            evidence=_make_evidence(path),
        )

    def _plan_through_node(self, q: MultihopQuestion) -> MultihopPlan:
        a = q.endpoint_a or ""
        b = q.through_b or ""
        c = q.endpoint_c
        if not a or not b:
            return self._audit(q, _AUDIT_MISSING_ENDPOINT, "missing_endpoint")

        paths = self._graph.find_through_paths(a, b, c)
        if not paths:
            return self._audit(q, _AUDIT_MISSING_HOP, "no_through_path")

        path = _pick_best(paths)
        valid, reason = validate_path(path, q)
        if not valid:
            return self._audit(q, f"path_blocked: {reason}", "blocked")

        label = chain_label(path.hop1.predicate, path.hop2.predicate)
        return MultihopPlan(
            question=q, decision="answer", audit_reason=None,
            path=path, chain_label=label,
            render_args={
                "a": path.hop1.subject, "b": path.via_node, "c": path.hop2.object,
                "hop1": path.hop1, "hop2": path.hop2, "explicit_through": True,
            },
            evidence=_make_evidence(path),
        )

    def _plan_who_through(self, q: MultihopQuestion) -> MultihopPlan:
        a = q.endpoint_a or ""
        b = q.through_b or ""
        if not a or not b:
            return self._audit(q, _AUDIT_MISSING_ENDPOINT, "missing_endpoint")

        paths = self._graph.find_through_paths(a, b)
        if not paths:
            return self._audit(q, _AUDIT_MISSING_HOP, "no_through_path")

        path = _pick_best(paths)
        valid, reason = validate_path(path, q)
        if not valid:
            return self._audit(q, f"path_blocked: {reason}", "blocked")

        label = chain_label(path.hop1.predicate, path.hop2.predicate)
        return MultihopPlan(
            question=q, decision="answer", audit_reason=None,
            path=path, chain_label=label,
            render_args={
                "a": path.hop1.subject, "b": path.via_node, "c": path.hop2.object,
                "hop1": path.hop1, "hop2": path.hop2,
            },
            evidence=_make_evidence(path),
        )

    # ------------------------------------------------------------------ #

    def _audit(self, q: MultihopQuestion, reason: str, label: str) -> MultihopPlan:
        return MultihopPlan(
            question=q, decision="audit", audit_reason=reason,
            path=None, chain_label=label,
            render_args={"reason": reason, "question": q.question},
            evidence=MultihopEvidence(),
        )
