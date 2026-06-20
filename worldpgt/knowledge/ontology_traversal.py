"""Explicit ontology traversal over stable ``is_a`` overlay facts."""

from __future__ import annotations

import re
from collections import deque
from typing import Optional

from worldpgt.multihop_qa.path_validator import validate_hop_safety
from worldpgt.multihop_qa.types import HopEdge


def _norm(s: str) -> str:
    text = (s or "").lower().strip()
    text = re.sub(r"[''`]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _is_a_edge_from_item(item: dict) -> Optional[HopEdge]:
    overlay_type = item.get("overlay_type")
    if overlay_type == "overlay_relation":
        if item.get("predicate") != "is_a":
            return None
        subject = str(item.get("subject") or "").strip()
        obj = str(item.get("object") or "").strip()
    elif overlay_type == "overlay_definition":
        if item.get("predicate", "is_a") != "is_a":
            return None
        subject = str(item.get("subject") or "").strip()
        obj = str(item.get("definition") or "").strip()
    else:
        return None

    if not subject or not obj:
        return None

    return HopEdge(
        subject=subject,
        predicate="is_a",
        object=obj,
        overlay_type=str(overlay_type or ""),
        trust=str(item.get("trust") or ""),
        stability=str(item.get("stability") or ""),
        risk=str(item.get("risk") or ""),
        source_page=str(item.get("source_page") or ""),
    )


def _safe_is_a_edges(overlay_items: list[dict]) -> list[HopEdge]:
    edges: list[HopEdge] = []
    for item in overlay_items:
        if not isinstance(item, dict):
            continue
        edge = _is_a_edge_from_item(item)
        if edge is None:
            continue
        valid, _reason = validate_hop_safety(edge)
        if not valid:
            continue
        # Ontology traversal is stricter than generic multihop: it only walks
        # stable class-membership facts.
        if edge.stability != "stable":
            continue
        edges.append(edge)
    return edges


def find_is_a_path(
    entity_label: str,
    target_class_label: str,
    overlay_items: list[dict],
    ontology_layer_items: list[dict] | None = None,
    max_depth: int = 6,
) -> Optional[list[HopEdge]]:
    """Return an explicit ``is_a`` path from entity to target, or None.

    The traversal is exact after normalization, directed only from subject to
    class, and deterministic. When an ontology layer is supplied, outgoing
    edges from the main overlay are considered before layer edges.
    """

    start = _norm(entity_label)
    target = _norm(target_class_label)
    if not start or not target or max_depth <= 0:
        return None

    adjacency: dict[str, list[tuple[int, HopEdge]]] = {}
    for priority, items in enumerate((overlay_items, ontology_layer_items or [])):
        for edge in _safe_is_a_edges(items):
            adjacency.setdefault(_norm(edge.subject), []).append((priority, edge))
    for edges in adjacency.values():
        edges.sort(key=lambda row: (row[0], _norm(row[1].object), row[1].object, row[1].subject))

    queue: deque[tuple[str, list[HopEdge]]] = deque([(start, [])])
    visited_depth: dict[str, int] = {start: 0}

    while queue:
        node, path = queue.popleft()
        if len(path) >= max_depth:
            continue
        for _priority, edge in adjacency.get(node, []):
            next_node = _norm(edge.object)
            if not next_node:
                continue
            next_path = [*path, edge]
            if next_node == target:
                return next_path
            next_depth = len(next_path)
            prev_depth = visited_depth.get(next_node)
            if prev_depth is not None and prev_depth <= next_depth:
                continue
            visited_depth[next_node] = next_depth
            queue.append((next_node, next_path))

    return None
