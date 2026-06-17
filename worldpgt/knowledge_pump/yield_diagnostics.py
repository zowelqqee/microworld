"""Yield diagnostics for Knowledge Pump v1.

Reads existing pump artifacts and produces a deterministic funnel report
explaining where each batch loses usable knowledge.

No network calls. No writes to accepted memory or overlays.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from worldpgt.knowledge_pump.frontier_title_extractor import _has_prose_period_break, _PROSE_TERMINAL_WORDS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _title_rejected_by_hygiene(title: str) -> tuple[bool, str]:
    """Return (rejected, reason) using the strengthened frontier filter."""
    words = title.split()
    if not words:
        return True, "empty"
    if words[-1] in _PROSE_TERMINAL_WORDS:
        return True, "prose_terminal_word"
    if _has_prose_period_break(title):
        return True, "prose_period_break"
    return False, ""


# ---------------------------------------------------------------------------
# Batch history sanity
# ---------------------------------------------------------------------------

def _batch_history_sanity(history: list[dict]) -> dict[str, Any]:
    total = len(history)
    with_title_lists = sum(1 for b in history if b.get("titles_planned"))
    legacy = total - with_title_lists
    indices = [b.get("batch_index") for b in history]
    has_fetch_success_titles = any(b.get("fetch_success") for b in history)
    has_ready_titles = any(b.get("titles_ready") or b.get("ready_count") for b in history)

    seen: set = set()
    dup_count = 0
    for idx in indices:
        k = (idx,)
        if k in seen:
            dup_count += 1
        seen.add(k)

    # Sequence: within each "run" (grouped by restart) indices should be monotone
    sequence_valid = True
    prev = None
    for b in history:
        idx = b.get("batch_index")
        if prev is not None and idx is not None and idx < prev:
            sequence_valid = False
            break
        prev = idx

    return {
        "batch_history_total_entries": total,
        "batch_history_entries_with_title_lists": with_title_lists,
        "batch_history_legacy_entries": legacy,
        "batch_index_sequence_valid": sequence_valid,
        "duplicate_batch_index_count": dup_count,
        "batch_history_has_fetch_success_titles": has_fetch_success_titles,
        "batch_history_has_ready_titles": has_ready_titles,
    }


# ---------------------------------------------------------------------------
# Frontier title quality preview
# ---------------------------------------------------------------------------

def _frontier_quality(frontier: list[dict]) -> dict[str, Any]:
    rejected: list[dict] = []
    kept: list[dict] = []
    reason_counts: Counter = Counter()

    for item in frontier:
        title = item.get("title", "")
        is_rejected, reason = _title_rejected_by_hygiene(title)
        if is_rejected:
            rejected.append({"title": title, "reason": reason, "source": item.get("source", "")})
            reason_counts[reason] += 1
        else:
            kept.append({"title": title, "source": item.get("source", "")})

    return {
        "frontier_title_total": len(frontier),
        "frontier_title_rejected_by_hygiene_count": len(rejected),
        "frontier_title_kept_by_hygiene_count": len(kept),
        "frontier_title_rejection_by_reason": dict(reason_counts),
        "frontier_title_kept_samples": [x["title"] for x in kept[:20]],
        "frontier_title_rejected_samples": [x["title"] for x in rejected[:20]],
    }, rejected, kept


# ---------------------------------------------------------------------------
# Source-level candidate survival
# ---------------------------------------------------------------------------

def _candidate_survival(
    ingestion_candidates: list[dict],
    relation_candidates: list[dict],
    safe_delta: list[dict],
    weak_context: list[dict],
    precision_answerable: list[dict],
    precision_rejected: list[dict],
    precision_quarantine: list[dict],
    precision_property: list[dict],
) -> dict[str, dict]:
    """Per source_page survival counters."""
    pages: dict[str, dict] = defaultdict(lambda: {
        "source_page": "",
        "ingestion_candidate_count": 0,
        "relation_candidate_count": 0,
        "safe_delta_count": 0,
        "weak_context_count": 0,
        "entity_count": 0,
        "answerable_before_precision_count": 0,
        "precision_accepted_count": 0,
        "precision_rejected_count": 0,
        "precision_quarantined_count": 0,
        "property_candidate_count": 0,
        "final_fact_count": 0,
    })

    for item in ingestion_candidates:
        pg = item.get("source_doc_title") or item.get("source_page") or "unknown"
        pages[pg]["source_page"] = pg
        pages[pg]["ingestion_candidate_count"] += 1

    for item in relation_candidates:
        pg = item.get("source_title") or item.get("source_page") or item.get("source_doc_title") or "unknown"
        pages[pg]["source_page"] = pg
        pages[pg]["relation_candidate_count"] += 1

    for item in safe_delta:
        pg = item.get("source_page") or "unknown"
        pages[pg]["source_page"] = pg
        pages[pg]["safe_delta_count"] += 1
        if item.get("trust") == "weak_context_only":
            pages[pg]["weak_context_count"] += 1

    for item in weak_context:
        pg = item.get("source_page") or "unknown"
        pages[pg]["source_page"] = pg
        # already counted above via safe_delta; avoid double-count

    # precision answerable: before precision (entity_delta contains pre-precision items)
    for item in precision_answerable:
        pg = item.get("source_page") or "unknown"
        pages[pg]["source_page"] = pg
        pages[pg]["precision_accepted_count"] += 1
        ot = item.get("overlay_type", "")
        if ot == "overlay_entity":
            pages[pg]["entity_count"] += 1
        elif ot in ("overlay_relation", "overlay_definition"):
            pages[pg]["final_fact_count"] += 1

    for item in precision_rejected:
        pg = (item.get("item") or {}).get("source_page") or item.get("source_page") or "unknown"
        pages[pg]["source_page"] = pg
        pages[pg]["precision_rejected_count"] += 1

    for item in precision_quarantine:
        pg = (item.get("item") or {}).get("source_page") or item.get("source_page") or "unknown"
        pages[pg]["source_page"] = pg
        pages[pg]["precision_quarantined_count"] += 1

    for item in precision_property:
        pg = (item.get("item") or {}).get("source_page") or item.get("source_page") or "unknown"
        pages[pg]["source_page"] = pg
        pages[pg]["property_candidate_count"] += 1

    # answerable_before_precision: sum of accepted + rejected + quarantined + property
    for pg in pages:
        d = pages[pg]
        d["answerable_before_precision_count"] = (
            d["precision_accepted_count"]
            + d["precision_rejected_count"]
            + d["precision_quarantined_count"]
            + d["property_candidate_count"]
        )

    return dict(pages)


def _survival_surfaces(survival: dict[str, dict]) -> dict[str, Any]:
    entries = list(survival.values())

    def top(key: str, n: int = 20) -> list[dict]:
        return sorted(entries, key=lambda x: -x.get(key, 0))[:n]

    no_facts = [
        e for e in entries
        if e.get("ingestion_candidate_count", 0) > 0 and e.get("final_fact_count", 0) == 0
    ]
    return {
        "top_sources_by_ingestion_candidates": top("ingestion_candidate_count"),
        "top_sources_by_weak_context": top("weak_context_count"),
        "top_sources_by_rejected": top("precision_rejected_count"),
        "top_sources_by_final_facts": top("final_fact_count"),
        "sources_with_candidates_but_no_final_facts": no_facts,
    }


# ---------------------------------------------------------------------------
# Sink classification
# ---------------------------------------------------------------------------

_SINK_CATEGORIES = [
    "frontier_low_quality",
    "fetch_failed",
    "not_ready",
    "weak_context_only",
    "duplicate_entity_only",
    "duplicate_relation_or_definition",
    "precision_rejected",
    "precision_quarantined",
    "property_candidate_only",
    "candidate_not_mapped_to_supported_relation",
    "batch_attribution_unavailable",
]


def _classify_sinks(summary: dict, safe_delta: list[dict], precision_rejected: list[dict],
                    precision_quarantine: list[dict], fresh_quarantine: list[dict]) -> dict[str, Any]:
    fetched = summary.get("fetched_count_total", 0)
    fetch_success = summary.get("fetch_success_count_total", 0)
    ready = summary.get("ready_for_ingestion_count_total", 0)

    fetch_failed_count = fetched - fetch_success if fetched and fetch_success else 0
    not_ready_count = fetch_success - ready if fetch_success and ready else 0

    weak_count = sum(1 for x in safe_delta if x.get("trust") == "weak_context_only")
    entity_only = sum(1 for x in safe_delta if x.get("overlay_type") == "overlay_entity")

    quarantine_by_reason: Counter = Counter()
    for item in fresh_quarantine:
        quarantine_by_reason[item.get("reason", "unknown")] += 1

    rejected_by_reason: Counter = Counter()
    for item in precision_rejected:
        rejected_by_reason[item.get("reason", "unknown")] += 1

    precision_rej_count = len(precision_rejected)
    precision_quar_count = len(precision_quarantine)

    answerable_before = summary.get("answerable_delta_before_precision", 0)
    answerable_after = summary.get("answerable_delta_after_precision", 0)
    fact_count = summary.get("pump_answerable_fact_delta_count", 0)
    qa_fact_count = summary.get("pump_fact_qa_fact_count") or summary.get("pump_fact_qa_enabled") and fact_count or 0

    fresh_recompute_uses_new_ready = bool(
        summary.get("fresh_ingestion_candidates_total", 0) > summary.get("ingestion_candidates_total", 0)
    )
    merge_includes_fresh = bool(
        summary.get("fresh_merged_safe_delta_count", 0) > summary.get("merged_safe_delta_count", 0)
    )
    prev_fact_count = summary.get("pump_answerable_fact_delta_count_before_v2")
    if prev_fact_count is None:
        prev_fact_count = summary.get("pump_answerable_fact_delta_count", 0)
    fact_count_increased = bool(fact_count > prev_fact_count)

    prev_qa_fact_count = summary.get("pump_fact_qa_fact_count_before_v2")
    if prev_qa_fact_count is None:
        prev_qa_fact_count = prev_fact_count
    qa_fact_count_increased = bool(qa_fact_count > prev_qa_fact_count)

    sinks = {
        "fetch_failed": fetch_failed_count,
        "not_ready": not_ready_count,
        "weak_context_only": weak_count,
        "duplicate_entity_only": entity_only,
        "precision_rejected": precision_rej_count,
        "precision_quarantined": precision_quar_count,
        "missing_explicit_evidence_quarantine": quarantine_by_reason.get("missing_explicit_evidence", 0),
    }

    dominant = sorted(sinks.items(), key=lambda kv: -kv[1])[:3]

    return {
        "sink_counts": sinks,
        "dominant_sinks": [k for k, v in dominant],
        "fresh_recompute_appears_to_use_new_ready_docs": fresh_recompute_uses_new_ready,
        "merge_appears_to_include_fresh_candidates": merge_includes_fresh,
        "precision_accepted_fact_count_increased": fact_count_increased,
        "qa_fact_count_increased": qa_fact_count_increased,
        "answerable_fact_delta_count_before_v2": prev_fact_count,
        "qa_fact_count_before_v2": prev_qa_fact_count,
        "answerable_delta_before_precision": answerable_before,
        "answerable_delta_after_precision": answerable_after,
        "answerable_fact_delta_count": fact_count,
        "quarantine_by_reason": dict(quarantine_by_reason),
        "precision_rejected_by_reason": dict(rejected_by_reason),
        "attribution_mode": "inferred",
    }


# ---------------------------------------------------------------------------
# Batch funnel
# ---------------------------------------------------------------------------

def _batch_funnel(history: list[dict], summary: dict) -> list[dict]:
    rows = []
    for b in history:
        planned = b.get("titles_planned") or []
        fetched = b.get("titles_fetched") or []
        success = b.get("fetch_success") or []
        ready_count = b.get("ready_count") or 0
        not_ready_count = b.get("not_ready_count") or 0
        row = {
            "batch_index": b.get("batch_index"),
            "started_at": b.get("started_at", ""),
            "finished_at": b.get("finished_at", ""),
            "status": b.get("status", ""),
            "titles_planned": len(planned),
            "titles_fetched": len(fetched),
            "fetch_success": len(success) if success else b.get("fetch_success", 0) or 0,
            "fetch_failed": b.get("fetch_failed") if isinstance(b.get("fetch_failed"), int)
                            else len(b.get("fetch_failed") or []),
            "ready_count": ready_count,
            "not_ready_count": not_ready_count,
            "attribution_mode": "inferred" if not success else "exact",
        }
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_yield_diagnostics(
    pump_dir: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    """Compute yield diagnostics from existing pump artifacts.

    Reads from pump_dir (knowledge_pump_v1/), writes to out_dir.
    Does NOT modify any accepted memory or overlay artifacts.
    Does NOT make network calls.

    Returns a summary dict.
    """
    pump_dir = Path(pump_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Load artifacts ---
    summary = _load_json(pump_dir / "pump_summary.json", {})
    history = _load_json(pump_dir / "pump_batch_history.json", [])
    frontier = _load_json(pump_dir / "frontier_titles.json", [])
    ingestion_candidates = _load_json(pump_dir / "pump_fresh_ingestion_candidates.json", [])
    relation_candidates = _load_json(pump_dir / "pump_fresh_relation_candidates.json", [])
    safe_delta = _load_json(pump_dir / "pump_safe_delta.json", [])
    fresh_safe_delta = _load_json(pump_dir / "pump_fresh_safe_delta.json", safe_delta)
    weak_context = _load_json(pump_dir / "pump_weak_context_delta.json", [])
    precision_answerable = _load_json(pump_dir / "pump_precision_answerable_delta.json", [])
    precision_rejected = _load_json(pump_dir / "pump_precision_rejected_delta.json", [])
    precision_quarantine = _load_json(pump_dir / "pump_precision_quarantine.json", [])
    precision_property = _load_json(pump_dir / "pump_precision_property_candidates.json", [])
    fresh_quarantine = _load_json(pump_dir / "pump_fresh_quarantine.json", [])
    entity_delta = _load_json(pump_dir / "pump_entity_delta.json", [])
    answerable_delta = _load_json(pump_dir / "pump_answerable_delta.json", [])
    fact_qa_summary_path = pump_dir / "pump_fact_qa_v1" / "pump_fact_qa_summary.json"
    fact_qa_summary = _load_json(fact_qa_summary_path, {})

    # --- Funnel counts ---
    funnel = _build_funnel(summary, history, ingestion_candidates, relation_candidates,
                           fresh_safe_delta or safe_delta, weak_context,
                           precision_answerable, precision_rejected,
                           precision_quarantine, precision_property,
                           entity_delta, answerable_delta, fact_qa_summary)

    # --- Batch history sanity ---
    history_sanity = _batch_history_sanity(history)

    # --- Batch-level funnel ---
    batch_funnel = _batch_funnel(history, summary)

    # --- Frontier quality ---
    fq, rejected_titles, kept_titles = _frontier_quality(frontier)

    # --- Candidate survival ---
    survival = _candidate_survival(
        ingestion_candidates, relation_candidates,
        fresh_safe_delta or safe_delta, weak_context,
        precision_answerable, precision_rejected,
        precision_quarantine, precision_property,
    )
    survival_surfaces = _survival_surfaces(survival)

    # --- Sink classification ---
    sink_report = _classify_sinks(
        summary, fresh_safe_delta or safe_delta,
        precision_rejected, precision_quarantine, fresh_quarantine,
    )

    # --- Write artifacts ---
    _write_json(out_dir / "batch_yield_summary.json", {
        "funnel": funnel,
        "batch_history_sanity": history_sanity,
        "frontier_quality": fq,
        "sink_classification": sink_report,
        "attribution_mode": "inferred",
    })

    _write_json(out_dir / "batch_yield_report.json", {
        "funnel": funnel,
        "batch_history_sanity": history_sanity,
        "batch_funnel_by_batch": batch_funnel,
        "frontier_quality": fq,
        "sink_classification": sink_report,
        "attribution_mode": "inferred",
        "confirmations": {
            "network_calls": False,
            "trusted_memory_modified": False,
            "accepted_overlay_modified": False,
            "promoted_overlay_modified": False,
            "snapshot_dry_run_overlay_modified": False,
            "weak_context_excluded_from_answerable_facts": True,
            "entity_cards_not_counted_as_answerable_facts": True,
        },
    })

    _write_csv(
        out_dir / "batch_yield_by_batch.csv",
        batch_funnel,
        ["batch_index", "started_at", "finished_at", "status",
         "titles_planned", "titles_fetched", "fetch_success",
         "fetch_failed", "ready_count", "not_ready_count", "attribution_mode"],
    )

    # Source-page CSV
    source_rows = sorted(survival.values(), key=lambda x: -x.get("ingestion_candidate_count", 0))
    _write_csv(
        out_dir / "batch_yield_by_source_page.csv",
        source_rows,
        ["source_page", "ingestion_candidate_count", "relation_candidate_count",
         "safe_delta_count", "weak_context_count", "entity_count",
         "answerable_before_precision_count", "precision_accepted_count",
         "precision_rejected_count", "precision_quarantined_count",
         "property_candidate_count", "final_fact_count"],
    )

    _write_json(out_dir / "frontier_title_quality_report.json", {
        "frontier_quality": fq,
        "rejected_samples": rejected_titles[:50],
    })

    _write_csv(
        out_dir / "frontier_title_quality_report.csv",
        rejected_titles,
        ["title", "reason", "source"],
    )

    low_yield = [
        e for e in survival.values()
        if e.get("ingestion_candidate_count", 0) >= 3 and e.get("final_fact_count", 0) == 0
    ]
    high_yield = [
        e for e in survival.values()
        if e.get("final_fact_count", 0) >= 2
    ]
    _write_json(out_dir / "low_yield_source_pages.json",
                sorted(low_yield, key=lambda x: -x.get("ingestion_candidate_count", 0)))
    _write_json(out_dir / "high_yield_source_pages.json",
                sorted(high_yield, key=lambda x: -x.get("final_fact_count", 0)))

    _write_json(out_dir / "candidate_survival_report.json", {
        **survival_surfaces,
        "total_source_pages": len(survival),
    })
    _write_json(out_dir / "candidate_survival_by_reason.json", {
        "quarantine_by_reason": sink_report.get("quarantine_by_reason", {}),
        "precision_rejected_by_reason": sink_report.get("precision_rejected_by_reason", {}),
    })

    # Pages with candidates but no final facts
    no_fact_pages = [
        e for e in survival.values()
        if e.get("ingestion_candidate_count", 0) > 0 and e.get("final_fact_count", 0) == 0
    ]
    _write_json(out_dir / "new_candidate_no_fact_report.json", {
        "count": len(no_fact_pages),
        "pages": sorted(no_fact_pages, key=lambda x: -x.get("ingestion_candidate_count", 0))[:100],
    })

    result = {
        "funnel": funnel,
        "batch_history_sanity": history_sanity,
        "frontier_quality": fq,
        "sink_classification": sink_report,
        "candidate_survival_pages_total": len(survival),
        "low_yield_pages_count": len(low_yield),
        "high_yield_pages_count": len(high_yield),
        "no_fact_pages_count": len(no_fact_pages),
        "artifacts_written": [
            str(out_dir / "batch_yield_summary.json"),
            str(out_dir / "batch_yield_report.json"),
            str(out_dir / "batch_yield_by_batch.csv"),
            str(out_dir / "batch_yield_by_source_page.csv"),
            str(out_dir / "frontier_title_quality_report.json"),
            str(out_dir / "frontier_title_quality_report.csv"),
            str(out_dir / "low_yield_source_pages.json"),
            str(out_dir / "high_yield_source_pages.json"),
            str(out_dir / "candidate_survival_report.json"),
            str(out_dir / "candidate_survival_by_reason.json"),
            str(out_dir / "new_candidate_no_fact_report.json"),
        ],
        "network_calls": False,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "snapshot_dry_run_overlay_modified": False,
    }
    return result


def _build_funnel(
    summary: dict,
    history: list[dict],
    ingestion_candidates: list[dict],
    relation_candidates: list[dict],
    safe_delta: list[dict],
    weak_context: list[dict],
    precision_answerable: list[dict],
    precision_rejected: list[dict],
    precision_quarantine: list[dict],
    precision_property: list[dict],
    entity_delta: list[dict],
    answerable_delta: list[dict],
    fact_qa_summary: dict,
) -> dict[str, Any]:
    # Titles planned: sum across all batches
    titles_planned = sum(len(b.get("titles_planned") or []) for b in history)
    titles_fetched = sum(len(b.get("titles_fetched") or []) for b in history)

    fetch_success = summary.get("fetch_success_count_total", 0)
    ready_docs = summary.get("ready_for_ingestion_count_total", 0)

    ingestion_count = len(ingestion_candidates)
    relation_count = len(relation_candidates)

    safe_delta_total = len(safe_delta)
    # Count weak context from safe_delta directly (they appear there as overlay_context_link items)
    weak_context_total = sum(1 for x in safe_delta if x.get("trust") == "weak_context_only")

    entity_items = sum(1 for x in safe_delta if x.get("overlay_type") == "overlay_entity")
    answerable_before_precision = summary.get("answerable_delta_before_precision", 0)

    # precision results
    precision_accepted = len(precision_answerable)
    precision_rel = sum(1 for x in precision_answerable if x.get("overlay_type") == "overlay_relation")
    precision_def = sum(1 for x in precision_answerable if x.get("overlay_type") == "overlay_definition")
    precision_ent = sum(1 for x in precision_answerable if x.get("overlay_type") == "overlay_entity")
    precision_rejected_count = len(precision_rejected)
    precision_quarantined_count = len(precision_quarantine)
    property_candidates = len(precision_property)

    answerable_facts = summary.get("pump_answerable_fact_delta_count", 0)
    world_model_items = summary.get("pump_world_model_delta_count", 0)
    qa_fact_count = fact_qa_summary.get("pump_fact_qa_fact_count", answerable_facts)

    return {
        "titles_planned": titles_planned,
        "titles_fetched": titles_fetched,
        "fetch_success": fetch_success,
        "ready_docs": ready_docs,
        "ingestion_candidates": ingestion_count,
        "relation_candidates": relation_count,
        "safe_delta_items": safe_delta_total,
        "weak_context_items": weak_context_total,
        "entity_items": entity_items,
        "answerable_before_precision": answerable_before_precision,
        "precision_accepted_relations": precision_rel,
        "precision_accepted_definitions": precision_def,
        "precision_accepted_entities": precision_ent,
        "precision_rejected": precision_rejected_count,
        "precision_quarantined": precision_quarantined_count,
        "property_candidates": property_candidates,
        "new_answerable_facts": answerable_facts,
        "new_world_model_items": world_model_items,
        "qa_fact_count": qa_fact_count,
        "attribution_mode": "inferred",
    }


def update_pump_summary_with_yield(pump_dir: Path, yield_result: dict) -> None:
    """Additively update pump_summary.json with a compact yield diagnostics section.

    This is a proposal artifact — never accepted memory.
    """
    summary_path = pump_dir / "pump_summary.json"
    if not summary_path.exists():
        return
    try:
        data = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return
    data["yield_diagnostics"] = {
        "funnel": yield_result.get("funnel"),
        "dominant_sinks": yield_result.get("sink_classification", {}).get("dominant_sinks"),
        "frontier_rejected_by_hygiene": yield_result.get("frontier_quality", {}).get(
            "frontier_title_rejected_by_hygiene_count"
        ),
        "low_yield_pages_count": yield_result.get("low_yield_pages_count"),
        "no_fact_pages_count": yield_result.get("no_fact_pages_count"),
    }
    summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def update_pump_report_with_yield(pump_dir: Path, yield_result: dict) -> None:
    """Additively update pump_report.json with a compact yield diagnostics section.

    This is a proposal artifact — never accepted memory.
    """
    report_path = pump_dir / "pump_report.json"
    if not report_path.exists():
        return
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(data, dict):
        return
    if "summary" in data:
        data["summary"]["yield_diagnostics"] = {
            "funnel": yield_result.get("funnel"),
            "sink_classification": yield_result.get("sink_classification"),
            "frontier_quality": yield_result.get("frontier_quality"),
        }
    else:
        data["yield_diagnostics"] = yield_result.get("sink_classification")
    report_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
