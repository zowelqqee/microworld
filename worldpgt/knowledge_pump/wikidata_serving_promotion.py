"""User-authorized, reversible serving campaign for Wikidata seed proposals."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable


_EXTRACTION = "wikidata_api_structured_property_v1"
_SERVING_TIER = "evidence_grounded_wikidata_relation_v1"


def _key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(" ".join(str(row.get(field) or "").casefold().split()) for field in ("subject", "predicate", "object"))


def build_wikidata_serving_overlay(rows: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Make precision-accepted Wikidata edges queryable without trusting memory."""
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    input_rows = [dict(row) for row in rows]
    for row in input_rows:
        key = _key(row)
        if (
            row.get("overlay_type") != "overlay_relation"
            or row.get("open_web_extraction") != _EXTRACTION
            or row.get("source_kind") != "wikidata_api"
            or not str(row.get("canonical_qid") or "").startswith("Q")
            or not all(key)
            or not str(row.get("evidence_text") or "").strip()
        ):
            rejected.append({"reason": "not_precision_accepted_wikidata_relation"})
            continue
        if key in seen:
            continue
        seen.add(key)
        row.update({
            "experimental_tier": _SERVING_TIER,
            "trust": "user_authorized_wikidata_serving",
            "serving_status": "user_authorized_experimental",
            "promotion_source": "wikidata_seed_v1_precision_gate",
            "experimental_query_only": True,
            "safe_for_general_runtime": False,
        })
        selected.append(row)
    return selected, {
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
