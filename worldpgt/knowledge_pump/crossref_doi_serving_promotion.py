"""User-authorized serving promotion for precision-accepted Crossref DOI edges.

This creates a discovered *experimental serving graph* only.  It deliberately
does not alter accepted memory or the trusted promoted wiki overlay.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


_SERVING_TIER = "evidence_grounded_structured_relation_v1"
_EXTRACTION = "crossref_doi_structured_metadata_v1"


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(" ".join(str(row.get(field) or "").casefold().split()) for field in ("subject", "predicate", "object"))


def build_crossref_doi_serving_overlay(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Convert accepted Crossref DOI proposal edges into a queryable campaign graph."""

    input_rows = [dict(row) for row in rows]
    selected: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    rejected: list[dict[str, str]] = []
    for row in input_rows:
        key = _key(row)
        if (
            row.get("overlay_type") != "overlay_relation"
            or row.get("open_web_extraction") != _EXTRACTION
            or str(row.get("source_kind") or "") != "crossref_doi"
            or not all(key)
            or not str(row.get("evidence_text") or "").strip()
            or not str(row.get("canonical_doi") or "").strip()
        ):
            rejected.append({"reason": "not_precision_accepted_crossref_doi_relation"})
            continue
        if key in seen:
            continue
        seen.add(key)
        row.update({
            "experimental_tier": _SERVING_TIER,
            "trust": "user_authorized_crossref_doi_serving",
            "serving_status": "user_authorized_experimental",
            "promotion_source": "crossref_doi_precision_gate_v1",
            "experimental_query_only": True,
            # This layer is queryable in the main runtime but intentionally
            # remains distinct from accepted/general-purpose memory.
            "safe_for_general_runtime": False,
        })
        selected.append(row)
    summary = {
        "promotion_scope": "experimental_serving_graph_only",
        "accepted_memory_modified": False,
        "promoted_wiki_overlay_modified": False,
        "serving_experimental_overlay_modified": True,
        "input_precision_accepted_relations": len(input_rows),
        "serving_relation_count": len(selected),
        "serving_predicate_distribution": dict(sorted(Counter(str(row["predicate"]) for row in selected).items())),
        "deduplicated_or_invalid_input_count": len(rejected),
        "serving_tier": _SERVING_TIER,
    }
    return selected, summary
