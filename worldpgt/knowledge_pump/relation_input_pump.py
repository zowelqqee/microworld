"""Proposal-only training pump for the relation-input semantic graph.

The pump consumes reviewed or benchmark-labelled question examples.  It learns
only candidate ``phrase --denotes--> predicate`` edges, never facts.  Its output
is a separate proposal graph: runtime graph data and accepted memory are never
rewritten by this module.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable

from worldpgt.reasoning.relation_input_graph import question_frame_tokens


def _spans_for_surface(question: str, surface: str) -> tuple[tuple[int, int], ...]:
    if not surface:
        return ()
    pattern = re.compile(re.escape(surface), re.IGNORECASE)
    return tuple((match.start(), match.end()) for match in pattern.finditer(question))


def _predicate_values(row: dict[str, Any]) -> tuple[str, ...]:
    value = row.get("expected_predicate", row.get("predicate"))
    if isinstance(value, str) and value:
        return (value,)
    if isinstance(value, list) and all(isinstance(item, str) and item for item in value):
        return tuple(value)
    return ()


def build_relation_input_proposal(
    examples: Iterable[dict[str, Any]],
    *,
    min_support: int = 2,
) -> dict[str, Any]:
    """Aggregate labelled question frames into a reviewable proposal graph.

    A candidate is eligible only when a single canonical predicate is attached
    to the frame in every observed example and it has at least ``min_support``
    independent rows.  Ambiguous or unlabelled rows are reported, never
    promoted into a usable graph edge.
    """

    if min_support < 1:
        raise ValueError("min_support must be at least 1")
    counts: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
    examples_by_signature: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    rejected: list[dict[str, Any]] = []
    seen = 0
    for row in examples:
        if not isinstance(row, dict):
            continue
        question = row.get("question")
        predicates = _predicate_values(row)
        subject = row.get("expected_subject", row.get("subject", ""))
        if not isinstance(question, str) or not question.strip() or len(predicates) != 1:
            rejected.append({"reason": "missing_or_ambiguous_label", "id": row.get("id")})
            continue
        signature = question_frame_tokens(question, entity_spans=_spans_for_surface(question, str(subject)))
        if not signature:
            rejected.append({"reason": "empty_question_frame", "id": row.get("id")})
            continue
        predicate = predicates[0]
        counts[signature][predicate] += 1
        examples_by_signature[signature].append({"id": str(row.get("id", "")), "question": question})
        seen += 1

    nodes: list[dict[str, str]] = []
    edges: list[dict[str, Any]] = []
    proposals: list[dict[str, Any]] = []
    predicate_nodes: set[str] = set()
    for signature in sorted(counts):
        support = counts[signature]
        if len(support) != 1:
            rejected.append({
                "reason": "conflicting_predicate_labels",
                "phrase": " ".join(signature),
                "predicates": dict(sorted(support.items())),
            })
            continue
        predicate, count = next(iter(support.items()))
        if count < min_support:
            rejected.append({
                "reason": "insufficient_support",
                "phrase": " ".join(signature),
                "predicate": predicate,
                "support": count,
            })
            continue
        phrase = " ".join(signature)
        phrase_id = "phrase:" + "_".join(signature)
        predicate_id = "predicate:" + predicate
        nodes.append({"id": phrase_id, "kind": "phrase", "surface": phrase})
        predicate_nodes.add(predicate)
        edges.append({
            "source": phrase_id,
            "predicate": "denotes",
            "target": predicate_id,
            "support": count,
            "evidence_ids": [item["id"] for item in examples_by_signature[signature]],
        })
        proposals.append({"phrase": phrase, "predicate": predicate, "support": count})
    nodes.extend(
        {"id": "predicate:" + predicate, "kind": "predicate", "predicate": predicate}
        for predicate in sorted(predicate_nodes)
    )
    return {
        "version": 1,
        "proposal_only": True,
        "accepted_memory_modified": False,
        "runtime_graph_modified": False,
        "training_examples_seen": seen,
        "min_support": min_support,
        "nodes": nodes,
        "edges": edges,
        "proposals": proposals,
        "rejected": rejected,
    }


def write_relation_input_proposal(
    examples: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    min_support: int = 2,
) -> dict[str, Any]:
    """Write proposal and audit artifacts; never write the runtime graph."""

    proposal = build_relation_input_proposal(examples, min_support=min_support)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    graph = {key: proposal[key] for key in ("version", "proposal_only", "accepted_memory_modified", "runtime_graph_modified", "nodes", "edges")}
    report = {key: proposal[key] for key in ("training_examples_seen", "min_support", "proposals", "rejected")}
    (root / "relation_input_graph_proposal.json").write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")
    (root / "relation_input_graph_training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return proposal
