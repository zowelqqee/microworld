"""Read-only relation graph for Multi-hop QA v1.

Builds a directed adjacency structure from overlay_relation items only.
Definitions, entity cards, context links, and source facts are never
included as edges. Deterministic. No ML. No network.
"""
from __future__ import annotations

import re
from typing import Optional

from worldpgt.multihop_qa.types import HopEdge, MultihopPath


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"[''`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


class RelationGraph:
    """Directed graph built exclusively from overlay_relation items.

    No definitions, entity cards, context links, or source facts are
    used as edges. This is the core safety guarantee of the multi-hop layer.
    """

    def __init__(self, relations: list[dict]) -> None:
        # norm(subject) -> list of (predicate, norm(object))
        # Backward-compat format used by outgoing().
        self._out: dict[str, list[tuple[str, str]]] = {}
        # norm(node) -> canonical display name
        self._display: dict[str, str] = {}
        # (norm_subject, predicate, norm_object) -> raw relation dict for metadata lookup
        self._meta: dict[tuple[str, str, str], dict] = {}

        for r in relations:
            # Only include explicit overlay_relation items; skip definitions,
            # entity cards, context links, and source facts even if passed in.
            ot = r.get("overlay_type")
            if ot is not None and ot != "overlay_relation":
                continue
            subj = r.get("subject", "")
            obj = r.get("object", "")
            pred = r.get("predicate", "")
            if not subj or not obj or not pred:
                continue
            ns = _norm(subj)
            no = _norm(obj)
            self._display.setdefault(ns, subj)
            self._display.setdefault(no, obj)
            self._out.setdefault(ns, []).append((pred, no))
            # Store first-seen metadata (duplicates overwrite).
            self._meta[(ns, pred, no)] = r

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    def display(self, node_norm: str) -> str:
        return self._display.get(_norm(node_norm), node_norm)

    def outgoing(self, subject: str) -> list[tuple[str, str]]:
        """Return (predicate, object_norm) pairs for the given subject."""
        return list(self._out.get(_norm(subject), []))

    def is_known(self, entity: str) -> bool:
        return _norm(entity) in self._display

    # ------------------------------------------------------------------
    # Internal: build a HopEdge with full metadata
    # ------------------------------------------------------------------

    def _hop_edge(self, ns: str, pred: str, no: str) -> HopEdge:
        meta = self._meta.get((ns, pred, no), {})
        return HopEdge(
            subject=self.display(ns),
            predicate=pred,
            object=self.display(no),
            overlay_type=meta.get("overlay_type", "overlay_relation"),
            trust=meta.get("trust", ""),
            stability=meta.get("stability", ""),
            risk=meta.get("risk", ""),
            source_page=meta.get("source_page", ""),
            temporal_class=meta.get("temporal_class", ""),
            as_of=meta.get("as_of", ""),
        )

    # ------------------------------------------------------------------
    # Path finders
    # ------------------------------------------------------------------

    def find_2hop_paths(self, a: str, c: str) -> list[MultihopPath]:
        """Find all directed 2-hop paths A → B → C."""
        na, nc = _norm(a), _norm(c)
        paths: list[MultihopPath] = []
        for pred1, nb in self._out.get(na, []):
            for pred2, nc2 in self._out.get(nb, []):
                if nc2 == nc:
                    hop1 = self._hop_edge(na, pred1, nb)
                    hop2 = self._hop_edge(nb, pred2, nc)
                    paths.append(MultihopPath(hop1, hop2, self.display(nb)))
        return paths

    def find_develops_isa_paths(self, a: str, category: str) -> list[MultihopPath]:
        """Find paths: A develops B, B is_a [category]."""
        na = _norm(a)
        cat_norm = _norm(category)
        paths: list[MultihopPath] = []
        for pred1, nb in self._out.get(na, []):
            if pred1 != "develops":
                continue
            for pred2, nc in self._out.get(nb, []):
                if pred2 == "is_a" and _norm(self.display(nc)) == cat_norm:
                    hop1 = self._hop_edge(na, pred1, nb)
                    hop2 = self._hop_edge(nb, pred2, nc)
                    paths.append(MultihopPath(hop1, hop2, self.display(nb)))
        return paths

    def find_through_paths(
        self, a: str, b: str, c: Optional[str] = None
    ) -> list[MultihopPath]:
        """Find directed paths A → B → C with B explicitly specified.

        If c is None, returns all paths from A through B to any C.
        """
        na, nb = _norm(a), _norm(b)
        paths: list[MultihopPath] = []
        for pred1, nb2 in self._out.get(na, []):
            if nb2 != nb:
                continue
            for pred2, nc in self._out.get(nb, []):
                if c is None or _norm(c) == nc:
                    hop1 = self._hop_edge(na, pred1, nb)
                    hop2 = self._hop_edge(nb, pred2, nc)
                    paths.append(MultihopPath(hop1, hop2, self.display(nb)))
        return paths

    def node_count(self) -> int:
        return len(self._display)

    def edge_count(self) -> int:
        return sum(len(v) for v in self._out.values())
