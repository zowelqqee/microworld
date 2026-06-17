"""Yield-ranked frontier + incremental yield gate for Knowledge Pump v1.9.

Blind pumping has stopped producing answerable knowledge: raw fetch volume
grows while answerable facts stay flat. This module replaces blind frontier
selection with a *yield-aware* control layer that:

1. Measures incremental yield per batch (via incremental_yield_trace).
2. Detects zero / negative answerable-fact growth.
3. Ranks future frontier titles by expected answerable yield.
4. Downranks / blocks historically low-yield sources and titles.
5. Emits a safety gate recommendation (blind vs yield-ranked, force required).

It is strictly selection/control. It does NOT fetch, does NOT mutate accepted
memory or any overlay, does NOT weaken validators or precision gates, and does
NOT promote weak context or entity cards into answerable facts.
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from worldpgt.knowledge_pump.incremental_yield_trace import (
    DEFAULT_RECENT_WINDOW,
    build_batch_trace,
    normalize_page,
    recent_window_metrics,
)
from worldpgt.knowledge_pump.yield_diagnostics import _title_rejected_by_hygiene
from worldpgt.knowledge_pump.yield_gate import (
    DEFAULT_MIN_YIELD_PER_READY_DOC,
    evaluate_yield_gate,
)

_ANSWERABLE_TYPES = ("overlay_relation", "overlay_definition")
_HIGH_YIELD_SIGNAL_SOURCES = {
    "relation_v2", "overlay_relation", "overlay_entity", "knowledge_requests",
}
_LOW_YIELD_INGESTION_THRESHOLD = 3  # >= this many candidates and 0 facts => low yield
_STOPWORD_TOKENS = {
    "the", "a", "an", "of", "in", "on", "and", "or", "for", "to", "with",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
}
_MAX_BATCH_PLAN = 250


# ---------------------------------------------------------------------------
# IO helpers
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
        for row in rows:
            out = dict(row)
            for key, val in out.items():
                if isinstance(val, (list, dict)):
                    out[key] = "; ".join(str(v) for v in val) if isinstance(val, list) else json.dumps(val)
            writer.writerow(out)


# ---------------------------------------------------------------------------
# Source-page hint parsing
# ---------------------------------------------------------------------------

_PAGE_IN_REASON = re.compile(r" in (.+?)\.md$")


def _source_hint(reason: str) -> str:
    """Extract the originating source page from a frontier reason string."""
    match = _PAGE_IN_REASON.search(reason or "")
    if not match:
        return ""
    raw = match.group(1)
    return unquote(raw).replace("_", " ").strip()


def _is_metadata_token(title: str) -> bool:
    words = (title or "").split()
    if not words:
        return True
    if len(words) == 1 and (title.lower() in _STOPWORD_TOKENS or len(title) < 3):
        return True
    if all(w.lower() in _STOPWORD_TOKENS for w in words):
        return True
    return False


def _object_terms(facts: list[dict]) -> set[str]:
    terms: set[str] = set()
    for fact in facts:
        obj = str(fact.get("object", ""))
        for word in re.findall(r"[A-Za-z0-9]+", obj):
            if len(word) > 3:
                terms.add(word.lower())
    return terms


# ---------------------------------------------------------------------------
# Yield signal sets
# ---------------------------------------------------------------------------

def build_yield_signals(
    answerable_facts: list[dict],
    *,
    ingestion_candidates: list[dict] | None = None,
    history: list[dict] | None = None,
    manifest: list[dict] | None = None,
    high_yield_pages: list[dict] | None = None,
    low_yield_pages: list[dict] | None = None,
) -> dict[str, Any]:
    """Compute the positive/negative signal sets used to score titles."""
    ingestion_candidates = ingestion_candidates or []
    history = history or []
    manifest = manifest or []

    answerable = [f for f in answerable_facts if f.get("overlay_type") in _ANSWERABLE_TYPES]

    fact_pages = {normalize_page(f.get("source_page", "")) for f in answerable}
    fact_pages.discard("")

    fact_subjects = {normalize_page(f.get("subject", "")) for f in answerable}
    fact_subjects.discard("")
    fact_object_pages = {
        normalize_page(f.get("object", "")) for f in answerable
        if f.get("overlay_type") == "overlay_relation"
    }
    fact_object_pages.discard("")
    object_terms = _object_terms([f for f in answerable if f.get("overlay_type") == "overlay_relation"])

    # facts per page -> high-yield producers (>=2 facts) computed self-sufficiently.
    facts_per_page: dict[str, int] = defaultdict(int)
    for fact in answerable:
        facts_per_page[normalize_page(fact.get("source_page", ""))] += 1
    high_yield_self = {pg for pg, n in facts_per_page.items() if n >= 2 and pg}

    # Enrich with diagnostics outputs if provided.
    if high_yield_pages:
        for entry in high_yield_pages:
            high_yield_self.add(normalize_page(entry.get("source_page", "")))
    high_yield_self.discard("")

    # ingestion candidates per page with zero facts => low-yield source page.
    ingestion_per_page: dict[str, int] = defaultdict(int)
    for cand in ingestion_candidates:
        page = normalize_page(
            cand.get("source_doc_title") or cand.get("source_page") or ""
        )
        if page:
            ingestion_per_page[page] += 1
    low_yield_self = {
        pg for pg, n in ingestion_per_page.items()
        if n >= _LOW_YIELD_INGESTION_THRESHOLD and facts_per_page.get(pg, 0) == 0 and pg
    }
    if low_yield_pages:
        for entry in low_yield_pages:
            low_yield_self.add(normalize_page(entry.get("source_page", "")))
    low_yield_self.discard("")

    # already-fetched and failed-fetch sets from history + manifest.
    already_fetched: set[str] = set()
    failed_fetch: set[str] = set()
    for batch in history:
        for title in (batch.get("fetch_success") or []):
            already_fetched.add(normalize_page(title))
        for title in (batch.get("titles_fetched") or []):
            already_fetched.add(normalize_page(title))
        for title in (batch.get("fetch_failed") or []):
            failed_fetch.add(normalize_page(title))
    for row in manifest:
        already_fetched.add(normalize_page(row.get("normalized_title") or row.get("title") or ""))
    already_fetched.discard("")
    failed_fetch.discard("")

    return {
        "fact_pages": fact_pages,
        "fact_subjects": fact_subjects,
        "fact_object_pages": fact_object_pages,
        "object_terms": object_terms,
        "high_yield_pages": high_yield_self,
        "low_yield_pages": low_yield_self,
        "already_fetched": already_fetched,
        "failed_fetch": failed_fetch,
        "facts_per_page": dict(facts_per_page),
        "ingestion_per_page": dict(ingestion_per_page),
    }


# ---------------------------------------------------------------------------
# Title scoring
# ---------------------------------------------------------------------------

def score_title(title: str, source: str, reason: str, weight: int, signals: dict) -> dict:
    """Score a single frontier title for expected answerable yield."""
    ntitle = normalize_page(title)
    hint = _source_hint(reason)
    hint_n = normalize_page(hint)
    source_tags = set((source or "").split("|"))

    rejected, hygiene_reason = _title_rejected_by_hygiene(title)
    already = ntitle in signals["already_fetched"]

    score = 0
    positive: list[str] = []
    negative: list[str] = []

    # --- Positive signals ---
    if ntitle in signals["fact_pages"]:
        score += 60
        positive.append("source_page_produced_accepted_fact")
    if ntitle in signals["high_yield_pages"]:
        score += 40
        positive.append("in_high_yield_source_pages")
    if ntitle in signals["fact_subjects"]:
        score += 30
        positive.append("subject_of_accepted_fact")
    if ntitle in signals["fact_object_pages"]:
        score += 25
        positive.append("object_of_accepted_fact")
    if hint_n and hint_n in signals["high_yield_pages"] and not already:
        score += 20
        positive.append("adjacent_to_high_yield_source_page")
    if not rejected and not _is_metadata_token(title):
        score += 8
        positive.append("encyclopedic_entity_form")
    if signals["object_terms"] and any(
        w.lower() in signals["object_terms"] for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) > 3
    ):
        score += 10
        positive.append("term_in_accepted_fact_objects")
    if source_tags & _HIGH_YIELD_SIGNAL_SOURCES:
        score += 6
        positive.append("relation_or_overlay_source")
    # small deterministic weight tie-break (capped, never dominates signals).
    score += min(int(weight or 0), 50) / 50.0

    # --- Negative signals ---
    if already:
        score -= 100
        negative.append("already_fetched")
    if rejected:
        score -= 80
        negative.append(f"hygiene_rejected:{hygiene_reason}")
    if ntitle in signals["low_yield_pages"]:
        score -= 50
        negative.append("low_yield_source_page")
    if ntitle in signals["failed_fetch"]:
        score -= 40
        negative.append("historically_failed_fetch")
    if _is_metadata_token(title):
        score -= 30
        negative.append("metadata_or_stopword_token")
    if hint_n and hint_n in signals["low_yield_pages"] and hint_n not in signals["high_yield_pages"]:
        score -= 15
        negative.append("adjacent_to_low_yield_source_page")

    blocked = bool(already or rejected or ntitle in signals["low_yield_pages"] or _is_metadata_token(title))
    if blocked:
        expected_yield_class = "blocked"
    elif score >= 60:
        expected_yield_class = "high"
    elif score >= 20:
        expected_yield_class = "medium"
    else:
        expected_yield_class = "low"

    return {
        "title": title,
        "normalized_title": ntitle,
        "score": round(score, 4),
        "positive_reasons": positive,
        "negative_reasons": negative,
        "already_fetched": already,
        "source_hint": hint,
        "expected_yield_class": expected_yield_class,
        "blocked": blocked,
    }


def score_frontier(frontier: list[dict], signals: dict) -> list[dict]:
    """Score and de-duplicate frontier titles (highest score per title kept)."""
    best: dict[str, dict] = {}
    for item in frontier:
        title = item.get("title", "")
        if not title:
            continue
        scored = score_title(
            title,
            item.get("source", ""),
            item.get("reason", ""),
            item.get("weight", 0),
            signals,
        )
        key = scored["normalized_title"]
        prior = best.get(key)
        if prior is None or scored["score"] > prior["score"]:
            best[key] = scored
    return sorted(best.values(), key=lambda x: (-x["score"], x["title"].casefold()))


# ---------------------------------------------------------------------------
# Plan / blocklist construction
# ---------------------------------------------------------------------------

def build_batch_plan(scored: list[dict], *, limit: int = _MAX_BATCH_PLAN) -> list[dict]:
    """Top eligible (non-blocked, not-fetched) titles, ranked by score."""
    plan: list[dict] = []
    for entry in scored:
        if entry["blocked"] or entry["already_fetched"]:
            continue
        plan.append({
            "title": entry["title"],
            "score": entry["score"],
            "positive_reasons": entry["positive_reasons"],
            "negative_reasons": entry["negative_reasons"],
            "already_fetched": False,
            "source_hint": entry["source_hint"],
            "expected_yield_class": entry["expected_yield_class"],
        })
        if len(plan) >= limit:
            break
    return plan


def build_blocklist(scored: list[dict]) -> list[dict]:
    """Low-yield / fragment / metadata / failed titles that should not be fetched.

    Already-fetched-only titles are excluded — they are not a yield problem,
    just exhausted.
    """
    block: list[dict] = []
    for entry in scored:
        block_reasons = [
            r for r in entry["negative_reasons"]
            if r != "already_fetched"
            and (
                r == "low_yield_source_page"
                or r == "historically_failed_fetch"
                or r == "metadata_or_stopword_token"
                or r.startswith("hygiene_rejected")
                or r == "adjacent_to_low_yield_source_page"
            )
        ]
        if not block_reasons:
            continue
        block.append({
            "title": entry["title"],
            "score": entry["score"],
            "block_reasons": block_reasons,
            "source_hint": entry["source_hint"],
            "expected_yield_class": entry["expected_yield_class"],
        })
    return sorted(block, key=lambda x: (x["score"], x["title"].casefold()))


def high_yield_frontier(scored: list[dict]) -> list[dict]:
    """Titles expected to bear answerable facts (high/medium class)."""
    out: list[dict] = []
    for entry in scored:
        if entry["blocked"] or entry["already_fetched"]:
            continue
        if entry["expected_yield_class"] in ("high", "medium"):
            out.append({
                "title": entry["title"],
                "score": entry["score"],
                "positive_reasons": entry["positive_reasons"],
                "source_hint": entry["source_hint"],
                "expected_yield_class": entry["expected_yield_class"],
            })
    return out


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_yield_ranked_frontier(
    pump_dir: str | Path,
    out_dir: str | Path,
    *,
    snapshots_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    recent_window: int = DEFAULT_RECENT_WINDOW,
    min_yield_per_ready_doc: float = DEFAULT_MIN_YIELD_PER_READY_DOC,
    batch_plan_limit: int = _MAX_BATCH_PLAN,
    force: bool = False,
    frontier_policy: str = "yield-ranked",
) -> dict[str, Any]:
    """Compute the yield-ranked frontier and gate from existing pump artifacts.

    Reads from ``pump_dir`` (and optional sibling dirs); writes only under
    ``out_dir``. No network calls; no mutation of protected files.
    """
    pump_dir = Path(pump_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if diagnostics_dir is None:
        diagnostics_dir = pump_dir / "yield_diagnostics_v1"
    diagnostics_dir = Path(diagnostics_dir)

    # --- Load artifacts ---
    summary = _load_json(pump_dir / "pump_summary.json", {}) or {}
    history = _load_json(pump_dir / "pump_batch_history.json", []) or []
    frontier = _load_json(pump_dir / "frontier_titles.json", []) or []
    answerable = _load_json(pump_dir / "pump_precision_answerable_delta.json", []) or []
    ingestion_candidates = _load_json(pump_dir / "pump_fresh_ingestion_candidates.json", []) or []
    relation_candidates = _load_json(pump_dir / "pump_fresh_relation_candidates.json", []) or []
    safe_delta = _load_json(pump_dir / "pump_fresh_safe_delta.json", None)
    if safe_delta is None:
        safe_delta = _load_json(pump_dir / "pump_safe_delta.json", []) or []
    weak_context = _load_json(pump_dir / "pump_weak_context_delta.json", []) or []
    high_yield_pages = _load_json(diagnostics_dir / "high_yield_source_pages.json", []) or []
    low_yield_pages = _load_json(diagnostics_dir / "low_yield_source_pages.json", []) or []

    manifest: list[dict] = []
    if snapshots_dir is not None:
        manifest = _load_json(Path(snapshots_dir) / "snapshot_manifest.json", []) or []

    # --- Incremental yield trace ---
    trace_rows, trace_meta = build_batch_trace(
        history,
        answerable,
        ingestion_candidates=ingestion_candidates,
        relation_candidates=relation_candidates,
        safe_delta=safe_delta,
        weak_context=weak_context,
    )
    prev_summary = _load_json(out_dir / "incremental_yield_summary.json", {}) or {}
    metrics = recent_window_metrics(
        trace_rows,
        recent_window=recent_window,
        prev_cumulative_answerable=prev_summary.get("cumulative_answerable_facts"),
        cumulative_answerable=trace_meta["cumulative_answerable_facts"],
    )

    # --- Yield signals + scoring ---
    signals = build_yield_signals(
        answerable,
        ingestion_candidates=ingestion_candidates,
        history=history,
        manifest=manifest,
        high_yield_pages=high_yield_pages,
        low_yield_pages=low_yield_pages,
    )
    scored = score_frontier(frontier, signals)
    plan = build_batch_plan(scored, limit=batch_plan_limit)
    blocklist = build_blocklist(scored)
    high_yield = high_yield_frontier(scored)

    yield_ranked_available = len(plan) > 0

    # --- Gate ---
    gate = evaluate_yield_gate(
        metrics,
        min_yield_per_ready_doc=min_yield_per_ready_doc,
        yield_ranked_available=yield_ranked_available,
        frontier_policy=frontier_policy,
        force=force,
    )

    # --- Compose summary ---
    missing_data = _missing_data_report(history, answerable, frontier, summary)
    incremental_summary = {
        "recent_batches_analyzed": metrics["recent_batches_analyzed"],
        "recent_titles_fetched": metrics["recent_titles_fetched"],
        "recent_fetch_success": metrics["recent_fetch_success"],
        "recent_ready_docs": metrics["recent_ready_docs"],
        "recent_new_answerable_facts": metrics["recent_new_answerable_facts"],
        "recent_new_relations": metrics["recent_new_relations"],
        "recent_new_definitions": metrics["recent_new_definitions"],
        "recent_answerable_yield_per_ready_doc": metrics["recent_answerable_yield_per_ready_doc"],
        "recent_answerable_yield_per_fetch_success": metrics["recent_answerable_yield_per_fetch_success"],
        "zero_yield_batch_count": metrics["zero_yield_batch_count"],
        "negative_yield_detected": metrics["negative_yield_detected"],
        "blind_fetch_recommended": gate["blind_fetch_recommended"],
        "yield_ranked_fetch_recommended": gate["yield_ranked_fetch_recommended"],
        "force_required_for_blind_fetch": gate["force_required_for_blind_fetch"],
        "cumulative_answerable_facts": trace_meta["cumulative_answerable_facts"],
        "cumulative_relations": trace_meta["cumulative_relations"],
        "cumulative_definitions": trace_meta["cumulative_definitions"],
        "baseline_answerable_facts": trace_meta["baseline_answerable_facts"],
        "attribution_mode": trace_meta["attribution_mode"],
        "attribution_confidence": trace_meta["attribution_confidence"],
        "recent_window": metrics["recent_window"],
        "yield_ranked_selected_count": len(plan),
        "yield_ranked_available_count": len([s for s in scored if not s["blocked"] and not s["already_fetched"]]),
        "high_yield_frontier_count": len(high_yield),
        "low_yield_blocklist_count": len(blocklist),
        "missing_data": missing_data,
    }

    # --- Write artifacts ---
    _write_json(out_dir / "incremental_yield_summary.json", incremental_summary)

    _write_csv(
        out_dir / "incremental_yield_by_batch.csv",
        trace_rows,
        ["history_position", "batch_index", "status", "titles_selected",
         "titles_fetched", "fetch_success", "ready_docs", "ingestion_candidates",
         "relation_candidates", "safe_delta_items", "weak_context_items",
         "precision_accepted_relations", "precision_accepted_definitions",
         "new_answerable_facts", "new_world_model_items", "yield_score",
         "attribution_mode"],
    )

    title_rows = _title_yield_rows(answerable, signals)
    _write_csv(
        out_dir / "incremental_yield_by_title.csv",
        title_rows,
        ["source_page", "answerable_facts", "relations", "definitions",
         "ingestion_candidates", "is_high_yield", "is_low_yield"],
    )

    source_page_rows = _source_page_rows(signals)
    _write_csv(
        out_dir / "incremental_yield_by_source_page.csv",
        source_page_rows,
        ["source_page", "answerable_facts", "ingestion_candidates",
         "is_high_yield", "is_low_yield"],
    )

    _write_json(out_dir / "high_yield_frontier_titles.json", high_yield)
    _write_csv(
        out_dir / "high_yield_frontier_titles.csv",
        high_yield,
        ["title", "score", "expected_yield_class", "source_hint", "positive_reasons"],
    )

    _write_json(out_dir / "low_yield_blocklist.json", blocklist)
    _write_csv(
        out_dir / "low_yield_blocklist.csv",
        blocklist,
        ["title", "score", "expected_yield_class", "source_hint", "block_reasons"],
    )

    _write_json(out_dir / "yield_ranked_batch_plan.json", {
        "frontier_policy": frontier_policy,
        "selected_count": len(plan),
        "batch_plan_limit": batch_plan_limit,
        "titles": plan,
    })
    _write_csv(
        out_dir / "yield_ranked_batch_plan.csv",
        plan,
        ["title", "score", "expected_yield_class", "source_hint",
         "already_fetched", "positive_reasons", "negative_reasons"],
    )

    gate_report = {
        **gate,
        "metrics": metrics,
        "attribution_mode": trace_meta["attribution_mode"],
        "attribution_confidence": trace_meta["attribution_confidence"],
        "yield_ranked_selected_count": len(plan),
        "yield_ranked_available_count": incremental_summary["yield_ranked_available_count"],
        "missing_data": missing_data,
        "confirmations": _confirmations(),
    }
    _write_json(out_dir / "yield_gate_report.json", gate_report)

    return {
        "incremental_summary": incremental_summary,
        "gate": gate,
        "metrics": metrics,
        "trace_rows": trace_rows,
        "trace_meta": trace_meta,
        "scored": scored,
        "plan": plan,
        "blocklist": blocklist,
        "high_yield": high_yield,
        "yield_ranked_available": yield_ranked_available,
        "signals_summary": {
            "fact_pages": len(signals["fact_pages"]),
            "high_yield_pages": len(signals["high_yield_pages"]),
            "low_yield_pages": len(signals["low_yield_pages"]),
            "already_fetched": len(signals["already_fetched"]),
        },
        "artifacts_written": [
            "incremental_yield_summary.json",
            "incremental_yield_by_batch.csv",
            "incremental_yield_by_title.csv",
            "incremental_yield_by_source_page.csv",
            "high_yield_frontier_titles.json",
            "high_yield_frontier_titles.csv",
            "low_yield_blocklist.json",
            "low_yield_blocklist.csv",
            "yield_ranked_batch_plan.json",
            "yield_ranked_batch_plan.csv",
            "yield_gate_report.json",
        ],
        "network_calls": False,
    }


def select_yield_ranked_titles(
    pump_dir: str | Path,
    *,
    snapshots_dir: str | Path | None = None,
    diagnostics_dir: str | Path | None = None,
    limit: int = _MAX_BATCH_PLAN,
) -> list[str]:
    """Return an ordered list of yield-ranked, not-yet-fetched titles.

    Used by the pump runner to select titles without writing artifacts or
    touching the network.
    """
    pump_dir = Path(pump_dir)
    frontier = _load_json(pump_dir / "frontier_titles.json", []) or []
    answerable = _load_json(pump_dir / "pump_precision_answerable_delta.json", []) or []
    ingestion_candidates = _load_json(pump_dir / "pump_fresh_ingestion_candidates.json", []) or []
    history = _load_json(pump_dir / "pump_batch_history.json", []) or []
    if diagnostics_dir is None:
        diagnostics_dir = pump_dir / "yield_diagnostics_v1"
    high_yield_pages = _load_json(Path(diagnostics_dir) / "high_yield_source_pages.json", []) or []
    low_yield_pages = _load_json(Path(diagnostics_dir) / "low_yield_source_pages.json", []) or []
    manifest: list[dict] = []
    if snapshots_dir is not None:
        manifest = _load_json(Path(snapshots_dir) / "snapshot_manifest.json", []) or []

    signals = build_yield_signals(
        answerable,
        ingestion_candidates=ingestion_candidates,
        history=history,
        manifest=manifest,
        high_yield_pages=high_yield_pages,
        low_yield_pages=low_yield_pages,
    )
    scored = score_frontier(frontier, signals)
    plan = build_batch_plan(scored, limit=limit)
    return [entry["title"] for entry in plan]


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _title_yield_rows(answerable: list[dict], signals: dict) -> list[dict]:
    facts_per_page = signals["facts_per_page"]
    ingestion_per_page = signals["ingestion_per_page"]
    per_page: dict[str, dict] = defaultdict(lambda: {"relations": 0, "definitions": 0})
    display: dict[str, str] = {}
    for fact in answerable:
        if fact.get("overlay_type") not in _ANSWERABLE_TYPES:
            continue
        page = fact.get("source_page", "")
        key = normalize_page(page)
        display.setdefault(key, page)
        if fact.get("overlay_type") == "overlay_relation":
            per_page[key]["relations"] += 1
        else:
            per_page[key]["definitions"] += 1
    rows = []
    for key, counts in per_page.items():
        total = counts["relations"] + counts["definitions"]
        rows.append({
            "source_page": display.get(key, key),
            "answerable_facts": total,
            "relations": counts["relations"],
            "definitions": counts["definitions"],
            "ingestion_candidates": ingestion_per_page.get(key, 0),
            "is_high_yield": key in signals["high_yield_pages"],
            "is_low_yield": key in signals["low_yield_pages"],
        })
    return sorted(rows, key=lambda x: (-x["answerable_facts"], x["source_page"].casefold()))


def _source_page_rows(signals: dict) -> list[dict]:
    facts_per_page = signals["facts_per_page"]
    ingestion_per_page = signals["ingestion_per_page"]
    pages = set(facts_per_page) | set(ingestion_per_page)
    rows = []
    for key in pages:
        if not key:
            continue
        rows.append({
            "source_page": key,
            "answerable_facts": facts_per_page.get(key, 0),
            "ingestion_candidates": ingestion_per_page.get(key, 0),
            "is_high_yield": key in signals["high_yield_pages"],
            "is_low_yield": key in signals["low_yield_pages"],
        })
    return sorted(rows, key=lambda x: (-x["answerable_facts"], -x["ingestion_candidates"], x["source_page"]))


def _missing_data_report(history, answerable, frontier, summary) -> dict[str, Any]:
    """Identify data that would improve attribution precision."""
    missing: list[str] = []
    if not any(b.get("fetch_success") for b in history):
        missing.append("batch_history_lacks_fetch_success_title_lists")
    if not answerable:
        missing.append("no_precision_answerable_delta_artifact")
    if not frontier:
        missing.append("no_frontier_titles_artifact")
    if not any("source_page" in f for f in answerable[:1]):
        missing.append("answerable_facts_lack_source_page")
    return {
        "missing_signals": missing,
        "exact_per_batch_fact_attribution_available": False,
        "attribution_basis": "first_fetch_by_source_page_title_match",
        "note": (
            "Per-batch answerable-fact attribution is reconstructed, not "
            "directly recorded. Facts are credited to the first batch that "
            "fetched the matching source page; re-fetches of an already-fetched "
            "title therefore yield zero, which is the intended honest accounting."
        ),
    }


def _confirmations() -> dict[str, bool]:
    return {
        "network_calls": False,
        "accepted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "snapshot_dry_run_overlay_modified": False,
        "sense_memory_modified": False,
        "validators_weakened": False,
        "precision_gates_weakened": False,
        "weak_context_promoted_to_answerable_fact": False,
        "entity_cards_counted_as_answerable_facts": False,
    }
