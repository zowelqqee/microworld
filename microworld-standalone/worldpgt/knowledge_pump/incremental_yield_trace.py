"""Incremental yield trace for Knowledge Pump v1.9.

Measures *incremental* (newly added) answerable knowledge per batch, as
opposed to cumulative totals which are misleading. The core problem this
solves: raw fetch volume kept growing across batches while the number of
answerable facts stayed flat (or shrank), meaning recent batches were
unproductive.

Attribution model
-----------------
Per-batch answerable-fact attribution is not stored directly anywhere. We
reconstruct it deterministically by *first-fetch attribution*: each
answerable fact carries a ``source_page``; we credit that fact to the first
batch (in history order) whose ``fetch_success`` list contains the matching
title. A fact whose source page was never fetched by a pump batch (e.g. it
came from the original seed snapshot manifest) is counted as ``baseline`` and
never attributed to a pump batch.

This makes re-fetching an already-fetched title yield exactly zero new facts,
which is the honest accounting we want.

No network calls. No writes to accepted memory, overlays, or snapshots.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

# A fact is "answerable" only if it is a relation or a definition. Entity
# cards (overlay_entity) are world-model items but NOT answerable facts.
_ANSWERABLE_TYPES = ("overlay_relation", "overlay_definition")
_WORLD_MODEL_TYPES = ("overlay_relation", "overlay_definition", "overlay_entity")

DEFAULT_RECENT_WINDOW = 4


def normalize_page(value: str) -> str:
    """Normalize a title / source_page for cross-artifact matching."""
    return (value or "").replace("_", " ").strip().lower()


# ---------------------------------------------------------------------------
# First-fetch attribution
# ---------------------------------------------------------------------------

def _first_fetch_positions(history: list[dict]) -> dict[str, int]:
    """Map normalized fetched title -> first history position that fetched it."""
    first_pos: dict[str, int] = {}
    for pos, batch in enumerate(history):
        for title in (batch.get("fetch_success") or []):
            key = normalize_page(title)
            if key and key not in first_pos:
                first_pos[key] = pos
    return first_pos


def _attribute_by_source_page(
    items: list[dict],
    first_pos: dict[str, int],
    page_key: str,
) -> tuple[dict[int, int], int, int]:
    """Attribute a list of items to history positions by their source page.

    Returns (counts_by_position, attributed_total, baseline_total).
    """
    by_pos: dict[int, int] = defaultdict(int)
    attributed = 0
    baseline = 0
    for item in items:
        page = normalize_page(_extract_page(item, page_key))
        pos = first_pos.get(page)
        if pos is None:
            baseline += 1
            continue
        by_pos[pos] += 1
        attributed += 1
    return by_pos, attributed, baseline


def _extract_page(item: dict, primary_key: str) -> str:
    for key in (primary_key, "source_page", "source_doc_title", "source_title"):
        val = item.get(key)
        if val:
            return str(val)
    return ""


# ---------------------------------------------------------------------------
# Batch trace
# ---------------------------------------------------------------------------

def build_batch_trace(
    history: list[dict],
    answerable_facts: list[dict],
    *,
    ingestion_candidates: list[dict] | None = None,
    relation_candidates: list[dict] | None = None,
    safe_delta: list[dict] | None = None,
    weak_context: list[dict] | None = None,
    entity_items: list[dict] | None = None,
) -> tuple[list[dict], dict[str, Any]]:
    """Build a per-batch incremental yield trace.

    Each row carries newly-added (delta) counts attributed to that batch via
    first-fetch attribution — never cumulative totals.
    """
    ingestion_candidates = ingestion_candidates or []
    relation_candidates = relation_candidates or []
    safe_delta = safe_delta or []
    weak_context = weak_context or []

    answerable = [f for f in answerable_facts if f.get("overlay_type") in _ANSWERABLE_TYPES]
    relations = [f for f in answerable if f.get("overlay_type") == "overlay_relation"]
    definitions = [f for f in answerable if f.get("overlay_type") == "overlay_definition"]
    if entity_items is None:
        entity_items = [f for f in answerable_facts if f.get("overlay_type") == "overlay_entity"]

    first_pos = _first_fetch_positions(history)

    rel_by_pos, rel_attr, rel_base = _attribute_by_source_page(relations, first_pos, "source_page")
    def_by_pos, def_attr, def_base = _attribute_by_source_page(definitions, first_pos, "source_page")
    ent_by_pos, _ent_attr, _ent_base = _attribute_by_source_page(entity_items, first_pos, "source_page")
    ing_by_pos, _ing_attr, _ing_base = _attribute_by_source_page(ingestion_candidates, first_pos, "source_doc_title")
    relc_by_pos, _rc_attr, _rc_base = _attribute_by_source_page(relation_candidates, first_pos, "source_title")
    safe_by_pos, _s_attr, _s_base = _attribute_by_source_page(safe_delta, first_pos, "source_page")
    weak_by_pos, _w_attr, _w_base = _attribute_by_source_page(weak_context, first_pos, "source_page")

    total_fact_pages = len({normalize_page(f.get("source_page", "")) for f in answerable})
    mapped_fact_pages = len(
        {normalize_page(f.get("source_page", "")) for f in answerable
         if normalize_page(f.get("source_page", "")) in first_pos}
    )
    if total_fact_pages == 0:
        confidence = "low"
    else:
        ratio = mapped_fact_pages / total_fact_pages
        confidence = "high" if ratio >= 0.9 else "medium" if ratio >= 0.6 else "low"

    rows: list[dict] = []
    for pos, batch in enumerate(history):
        rels_n = rel_by_pos.get(pos, 0)
        defs_n = def_by_pos.get(pos, 0)
        facts_n = rels_n + defs_n
        ents_n = ent_by_pos.get(pos, 0)
        ready = int(batch.get("ready_count") or 0)
        rows.append({
            "history_position": pos,
            "batch_index": batch.get("batch_index"),
            "status": batch.get("status", ""),
            "titles_selected": len(batch.get("titles_planned") or []),
            "titles_fetched": len(batch.get("titles_fetched") or []),
            "fetch_success": len(batch.get("fetch_success") or []),
            "ready_docs": ready,
            "ingestion_candidates": ing_by_pos.get(pos, 0),
            "relation_candidates": relc_by_pos.get(pos, 0),
            "safe_delta_items": safe_by_pos.get(pos, 0),
            "weak_context_items": weak_by_pos.get(pos, 0),
            "precision_accepted_relations": rels_n,
            "precision_accepted_definitions": defs_n,
            "new_answerable_facts": facts_n,
            "new_world_model_items": facts_n + ents_n,
            "yield_score": round(facts_n / ready, 6) if ready else 0.0,
            "attribution_mode": "title_source_first_fetch",
        })

    meta = {
        "attribution_mode": "title_source_first_fetch",
        "attribution_confidence": confidence,
        "baseline_relation_facts": rel_base,
        "baseline_definition_facts": def_base,
        "baseline_answerable_facts": rel_base + def_base,
        "total_answerable_fact_source_pages": total_fact_pages,
        "mapped_answerable_fact_source_pages": mapped_fact_pages,
        "cumulative_answerable_facts": len(answerable),
        "cumulative_relations": len(relations),
        "cumulative_definitions": len(definitions),
    }
    return rows, meta


# ---------------------------------------------------------------------------
# Recent-window metrics
# ---------------------------------------------------------------------------

def recent_window_metrics(
    rows: list[dict],
    *,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    prev_cumulative_answerable: int | None = None,
    cumulative_answerable: int | None = None,
) -> dict[str, Any]:
    """Summarize the most recent ``recent_window`` batches.

    ``negative_yield_detected`` is True when either a prior cumulative count is
    known and the current count dropped, OR when recent batches consumed ready
    docs but produced zero new answerable facts (raw fetch grew, knowledge did
    not).
    """
    window = recent_window if recent_window and recent_window > 0 else len(rows)
    recent = rows[-window:] if rows else []

    def _sum(key: str) -> int:
        return sum(int(r.get(key, 0)) for r in recent)

    titles_fetched = _sum("titles_fetched")
    fetch_success = _sum("fetch_success")
    ready_docs = _sum("ready_docs")
    new_facts = _sum("new_answerable_facts")
    new_relations = _sum("precision_accepted_relations")
    new_definitions = _sum("precision_accepted_definitions")
    zero_yield_batches = sum(1 for r in recent if int(r.get("new_answerable_facts", 0)) == 0)

    yield_per_ready = round(new_facts / ready_docs, 6) if ready_docs else 0.0
    yield_per_success = round(new_facts / fetch_success, 6) if fetch_success else 0.0

    dropped = (
        prev_cumulative_answerable is not None
        and cumulative_answerable is not None
        and cumulative_answerable < prev_cumulative_answerable
    )
    raw_grew_no_facts = ready_docs > 0 and new_facts == 0
    negative_yield_detected = bool(dropped or raw_grew_no_facts)

    return {
        "recent_window": window,
        "recent_batches_analyzed": len(recent),
        "recent_titles_fetched": titles_fetched,
        "recent_fetch_success": fetch_success,
        "recent_ready_docs": ready_docs,
        "recent_new_answerable_facts": new_facts,
        "recent_new_relations": new_relations,
        "recent_new_definitions": new_definitions,
        "recent_answerable_yield_per_ready_doc": yield_per_ready,
        "recent_answerable_yield_per_fetch_success": yield_per_success,
        "zero_yield_batch_count": zero_yield_batches,
        "negative_yield_detected": negative_yield_detected,
        "cumulative_answerable_drop_detected": bool(dropped),
        "raw_fetch_grew_without_facts": raw_grew_no_facts,
    }
