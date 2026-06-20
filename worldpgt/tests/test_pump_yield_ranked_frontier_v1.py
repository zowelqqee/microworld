"""Tests for Knowledge Pump v1.9 Yield-Ranked Frontier + Incremental Yield Gate.

These tests verify the pump *selection/control* layer:
  * incremental (per-batch) yield is measured as delta, not cumulative,
  * zero/negative answerable-fact growth flips the gate away from blind fetch,
  * low-yield titles are blocklisted and high-yield titles up-ranked,
  * already-fetched titles never re-enter the next batch,
  * the yield-ranked policy changes selection vs the default policy.

No network calls. No mutation of accepted memory or any overlay.
"""

from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from worldpgt.knowledge_pump.incremental_yield_trace import (
    build_batch_trace,
    recent_window_metrics,
)
from worldpgt.knowledge_pump.yield_gate import evaluate_yield_gate
from worldpgt.knowledge_pump.yield_ranked_frontier import (
    build_yield_signals,
    run_yield_ranked_frontier,
    score_frontier,
    score_title,
    select_yield_ranked_titles,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_EXP = _REPO / "worldpgt" / "experiments"
_PUMP_DIR = _EXP / "knowledge_pump_v1"
_SNAPSHOTS = _EXP / "wiki_snapshots_v1"

_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
    _REPO / "worldpgt" / "continuation" / "sense_memory.py",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Synthetic fixtures — deterministic, no real artifacts required
# ---------------------------------------------------------------------------

def _history_with_zero_recent_yield() -> list[dict]:
    """Two productive early batches, then two re-fetch batches that add nothing.

    The later batches fetch titles already fetched earlier, so first-fetch
    attribution credits them zero new facts despite growing ready docs."""
    return [
        {"batch_index": 0, "titles_planned": ["A", "B"], "titles_fetched": ["Alpha", "Beta"],
         "fetch_success": ["Alpha", "Beta"], "fetch_failed": [], "ready_count": 2,
         "not_ready_count": 0, "status": "completed"},
        {"batch_index": 1, "titles_planned": ["C", "D"], "titles_fetched": ["Gamma", "Delta"],
         "fetch_success": ["Gamma", "Delta"], "fetch_failed": [], "ready_count": 2,
         "not_ready_count": 0, "status": "completed"},
        # Re-fetch of already-fetched titles -> zero new attributed facts.
        {"batch_index": 2, "titles_planned": ["A", "B"], "titles_fetched": ["Alpha", "Beta"],
         "fetch_success": ["Alpha", "Beta"], "fetch_failed": [], "ready_count": 2,
         "not_ready_count": 0, "status": "completed"},
        {"batch_index": 3, "titles_planned": ["C", "D"], "titles_fetched": ["Gamma", "Delta"],
         "fetch_success": ["Gamma", "Delta"], "fetch_failed": [], "ready_count": 2,
         "not_ready_count": 0, "status": "completed"},
    ]


def _answerable_facts() -> list[dict]:
    return [
        {"overlay_type": "overlay_relation", "subject": "Alpha", "predicate": "develops",
         "object": "Widget", "source_page": "Alpha"},
        {"overlay_type": "overlay_definition", "subject": "Beta", "definition": "a thing",
         "source_page": "Beta"},
        {"overlay_type": "overlay_relation", "subject": "Gamma", "predicate": "founded_by",
         "object": "Someone", "source_page": "Gamma"},
        # An entity card must NOT count as an answerable fact.
        {"overlay_type": "overlay_entity", "subject": "Delta", "source_page": "Delta"},
        # A weak-context link must NOT count as an answerable fact.
        {"overlay_type": "overlay_context_link", "trust": "weak_context_only",
         "source_page": "Alpha"},
    ]


# ---------------------------------------------------------------------------
# Test 1 + 2: incremental (delta) yield, cumulative is not new yield
# ---------------------------------------------------------------------------

def test_incremental_summary_detects_flat_answerable_growth() -> None:
    history = _history_with_zero_recent_yield()
    rows, meta = build_batch_trace(history, _answerable_facts())
    metrics = recent_window_metrics(rows, recent_window=2)
    # Recent two batches grew ready docs but produced zero NEW answerable facts.
    assert metrics["recent_ready_docs"] > 0
    assert metrics["recent_new_answerable_facts"] == 0
    assert metrics["negative_yield_detected"] is True


def test_cumulative_count_is_not_treated_as_new_yield() -> None:
    history = _history_with_zero_recent_yield()
    rows, meta = build_batch_trace(history, _answerable_facts())
    # Cumulative answerable facts (3 relations/definitions) is large...
    assert meta["cumulative_answerable_facts"] == 3
    # ...but the recent window's NEW facts are zero, not the cumulative count.
    metrics = recent_window_metrics(rows, recent_window=2)
    assert metrics["recent_new_answerable_facts"] == 0
    # The two early batches hold all the credit (first-fetch attribution).
    early = sum(r["new_answerable_facts"] for r in rows[:2])
    assert early == 3


# ---------------------------------------------------------------------------
# Test 3 + 4: zero/negative yield flips the gate
# ---------------------------------------------------------------------------

def test_zero_yield_blind_fetch_not_recommended() -> None:
    metrics = {
        "recent_new_answerable_facts": 0,
        "recent_ready_docs": 500,
        "recent_batches_analyzed": 4,
        "zero_yield_batch_count": 4,
        "negative_yield_detected": True,
    }
    gate = evaluate_yield_gate(metrics, yield_ranked_available=True)
    assert gate["blind_fetch_recommended"] is False


def test_zero_yield_recommends_yield_ranked_fetch() -> None:
    metrics = {
        "recent_new_answerable_facts": 0,
        "recent_ready_docs": 500,
        "recent_batches_analyzed": 4,
        "zero_yield_batch_count": 4,
        "negative_yield_detected": True,
    }
    gate = evaluate_yield_gate(metrics, yield_ranked_available=True)
    assert gate["yield_ranked_fetch_recommended"] is True
    assert gate["force_required_for_blind_fetch"] is True


def test_productive_yield_recommends_blind_fetch() -> None:
    metrics = {
        "recent_new_answerable_facts": 80,
        "recent_ready_docs": 100,
        "recent_batches_analyzed": 4,
        "zero_yield_batch_count": 0,
        "negative_yield_detected": False,
    }
    gate = evaluate_yield_gate(metrics, yield_ranked_available=True)
    assert gate["blind_fetch_recommended"] is True
    assert gate["force_required_for_blind_fetch"] is False


def test_force_allows_blind_fetch_despite_low_yield() -> None:
    metrics = {
        "recent_new_answerable_facts": 0,
        "recent_ready_docs": 500,
        "recent_batches_analyzed": 4,
        "zero_yield_batch_count": 4,
        "negative_yield_detected": True,
    }
    gate = evaluate_yield_gate(metrics, yield_ranked_available=True, force=True)
    # Still not "recommended", but explicitly allowed under force.
    assert gate["blind_fetch_recommended"] is False
    assert gate["blind_fetch_allowed"] is True


# ---------------------------------------------------------------------------
# Test 5 + 7: low-yield blocklisted, high-yield up-ranked
# ---------------------------------------------------------------------------

def _signals_for_scoring() -> dict:
    answerable = [
        {"overlay_type": "overlay_relation", "subject": "Northrop Grumman",
         "predicate": "develops", "object": "Triton", "source_page": "Northrop Grumman"},
        {"overlay_type": "overlay_definition", "subject": "Northrop Grumman",
         "definition": "an aerospace company", "source_page": "Northrop Grumman"},
        {"overlay_type": "overlay_relation", "subject": "Saturn IB",
         "predicate": "is_a", "object": "Rocket", "source_page": "Saturn IB"},
    ]
    # "Aerospace" produced many ingestion candidates but zero facts -> low yield.
    ingestion = [{"source_doc_title": "Aerospace"} for _ in range(8)]
    history = [
        {"batch_index": 0, "fetch_success": ["Northrop Grumman", "Saturn IB", "Aerospace"],
         "titles_fetched": ["Northrop Grumman", "Saturn IB", "Aerospace"], "fetch_failed": []},
    ]
    return build_yield_signals(answerable, ingestion_candidates=ingestion, history=history)


def test_low_yield_source_page_is_blocklisted_and_downranked() -> None:
    signals = _signals_for_scoring()
    # A frontier title that is itself the low-yield "Aerospace" page.
    scored = score_title("Aerospace", "snapshot_phrase", "capitalized phrase in Boeing.md", 50, signals)
    assert "low_yield_source_page" in scored["negative_reasons"]
    assert scored["expected_yield_class"] == "blocked"
    assert scored["score"] < 0


def test_high_yield_fact_bearing_title_is_upranked() -> None:
    signals = _signals_for_scoring()
    # "Northrop Grumman" is a high-yield source page (>=2 facts) and a fact subject.
    high = score_title("Northrop Grumman", "overlay_relation",
                       "relation candidate subject", 30, signals)
    low = score_title("Random Unrelated Phrase", "snapshot_phrase",
                      "capitalized phrase in Aerospace.md", 1, signals)
    assert high["score"] > low["score"]
    assert "source_page_produced_accepted_fact" in high["positive_reasons"]
    assert "in_high_yield_source_pages" in high["positive_reasons"]


# ---------------------------------------------------------------------------
# Test 6: already-fetched titles are not selected
# ---------------------------------------------------------------------------

def test_already_fetched_titles_not_selected_into_batch() -> None:
    answerable = [
        {"overlay_type": "overlay_relation", "subject": "Tesla", "predicate": "develops",
         "object": "Model S", "source_page": "Tesla"},
    ]
    history = [
        {"batch_index": 0, "fetch_success": ["Tesla", "SpaceX"],
         "titles_fetched": ["Tesla", "SpaceX"], "fetch_failed": []},
    ]
    signals = build_yield_signals(answerable, history=history)
    frontier = [
        {"title": "Tesla", "source": "overlay_relation", "reason": "relation subject", "weight": 100},
        {"title": "SpaceX", "source": "snapshot_phrase", "reason": "phrase", "weight": 90},
        {"title": "Rivian", "source": "snapshot_phrase", "reason": "phrase", "weight": 50},
    ]
    scored = score_frontier(frontier, signals)
    from worldpgt.knowledge_pump.yield_ranked_frontier import build_batch_plan
    plan = build_batch_plan(scored)
    plan_titles = {entry["title"] for entry in plan}
    assert "Tesla" not in plan_titles
    assert "SpaceX" not in plan_titles
    assert "Rivian" in plan_titles
    for entry in plan:
        assert entry["already_fetched"] is False


def test_stale_snapshot_signal_allows_recheck_of_already_fetched_title() -> None:
    signals = build_yield_signals(
        [],
        history=[{"fetch_success": ["Elon Musk"], "titles_fetched": ["Elon Musk"], "fetch_failed": []}],
        stale_fact_signals={
            "elon musk": {
                "subject": "Elon Musk",
                "predicate": "estimated_net_worth",
                "object": "US$1.1 trillion",
                "as_of": "2026-06",
                "staleness_ratio": 0.6,
            }
        },
    )
    scored = score_title("Elon Musk", "stale_snapshot_recheck", "refresh net worth", 0, signals)
    assert scored["already_fetched"] is False
    assert scored["recheck_required"] is True
    assert "stale_snapshot_recheck" in scored["positive_reasons"]
    assert "staleness_boost:6.00" in scored["positive_reasons"]
    assert "already_fetched" not in scored["negative_reasons"]


def test_run_yield_ranked_frontier_injects_stale_title_when_missing_from_frontier(tmp_path) -> None:
    pump_dir = tmp_path / "pump"
    out_dir = tmp_path / "out"
    pump_dir.mkdir()
    (pump_dir / "pump_summary.json").write_text("{}", encoding="utf-8")
    (pump_dir / "pump_batch_history.json").write_text(
        json.dumps([
            {"fetch_success": ["Elon Musk"], "titles_fetched": ["Elon Musk"], "fetch_failed": []}
        ]),
        encoding="utf-8",
    )
    (pump_dir / "frontier_titles.json").write_text("[]", encoding="utf-8")
    (pump_dir / "pump_precision_answerable_delta.json").write_text("[]", encoding="utf-8")

    result = run_yield_ranked_frontier(
        pump_dir,
        out_dir,
        stale_fact_signals={
            "elon musk": {
                "subject": "Elon Musk",
                "predicate": "estimated_net_worth",
                "object": "US$1.1 trillion",
                "as_of": "2026-06",
                "staleness_ratio": 0.6,
            }
        },
    )
    assert result["plan"]
    first = result["plan"][0]
    assert first["title"] == "Elon Musk"
    assert first["recheck_required"] is True
    assert "stale_snapshot_recheck" in first["positive_reasons"]


# ---------------------------------------------------------------------------
# Integration tests against real pump artifacts (read-only)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real_result(tmp_path_factory):
    if not _PUMP_DIR.exists():
        pytest.skip("knowledge_pump_v1 artifacts not found")
    out = tmp_path_factory.mktemp("yield_ranked")
    result = run_yield_ranked_frontier(_PUMP_DIR, out, snapshots_dir=_SNAPSHOTS)
    return out, result


# ---------------------------------------------------------------------------
# Test 1 against real data: recent yield metrics match current artifacts
# ---------------------------------------------------------------------------

def test_real_recent_yield_metrics_are_internally_consistent(real_result) -> None:
    _out, result = real_result
    summ = result["incremental_summary"]
    assert summ["recent_ready_docs"] > 0
    assert summ["recent_new_answerable_facts"] >= 0
    assert summ["recent_new_answerable_facts"] == (
        summ.get("recent_new_relations", 0)
        + summ.get("recent_new_definitions", 0)
    )
    if summ["recent_new_answerable_facts"] > 0:
        assert summ["recent_answerable_yield_per_ready_doc"] > 0
        assert summ["negative_yield_detected"] is False
    else:
        assert summ["recent_answerable_yield_per_ready_doc"] == 0


def test_real_gate_recommendation_matches_recent_yield(real_result) -> None:
    _out, result = real_result
    summ = result["incremental_summary"]
    gate = result["gate"]
    if summ["recent_new_answerable_facts"] == 0:
        assert gate["blind_fetch_recommended"] is False
        assert gate["yield_ranked_fetch_recommended"] is True
        assert gate["force_required_for_blind_fetch"] is True
    else:
        assert gate["blind_fetch_recommended"] is True
        assert gate["yield_ranked_fetch_recommended"] is False
        assert gate["force_required_for_blind_fetch"] is False


# ---------------------------------------------------------------------------
# Test 8 + 9: batch plan non-empty, each entry has score + reasons
# ---------------------------------------------------------------------------

def test_batch_plan_non_empty_when_candidates_exist(real_result) -> None:
    _out, result = real_result
    assert len(result["plan"]) > 0
    assert len(result["plan"]) <= 250


def test_each_planned_title_has_score_and_reasons(real_result) -> None:
    _out, result = real_result
    for entry in result["plan"]:
        assert "title" in entry and entry["title"]
        assert "score" in entry and isinstance(entry["score"], (int, float))
        assert "positive_reasons" in entry and isinstance(entry["positive_reasons"], list)
        assert "negative_reasons" in entry and isinstance(entry["negative_reasons"], list)
        assert "already_fetched" in entry
        assert "source_hint" in entry
        assert "expected_yield_class" in entry


def test_all_expected_artifacts_written(real_result) -> None:
    out, _result = real_result
    expected = [
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
    ]
    for name in expected:
        assert (out / name).exists(), f"Missing artifact: {name}"


def test_real_blocklist_non_empty(real_result) -> None:
    _out, result = real_result
    assert len(result["blocklist"]) > 0
    for entry in result["blocklist"]:
        assert entry["block_reasons"]


# ---------------------------------------------------------------------------
# Test 6 against real data
# ---------------------------------------------------------------------------

def test_real_plan_excludes_already_fetched(real_result) -> None:
    _out, result = real_result
    signals = build_yield_signals(
        json.loads((_PUMP_DIR / "pump_precision_answerable_delta.json").read_text()),
        history=json.loads((_PUMP_DIR / "pump_batch_history.json").read_text()),
        manifest=json.loads((_SNAPSHOTS / "snapshot_manifest.json").read_text()),
    )
    fetched = signals["already_fetched"]
    from worldpgt.knowledge_pump.incremental_yield_trace import normalize_page
    for entry in result["plan"]:
        assert normalize_page(entry["title"]) not in fetched


# ---------------------------------------------------------------------------
# Test 10: yield-ranked policy changes selection vs default — no network
# ---------------------------------------------------------------------------

def test_yield_ranked_selection_differs_from_default() -> None:
    if not _PUMP_DIR.exists():
        pytest.skip("artifacts not found")
    from worldpgt.knowledge_pump.expanded_allowlist_builder import build_expanded_allowlist
    from worldpgt.knowledge_pump.types import FrontierTitle

    frontier_raw = json.loads((_PUMP_DIR / "frontier_titles.json").read_text())
    answerable = json.loads((_PUMP_DIR / "pump_precision_answerable_delta.json").read_text())
    history = json.loads((_PUMP_DIR / "pump_batch_history.json").read_text())
    manifest = json.loads((_SNAPSHOTS / "snapshot_manifest.json").read_text())

    signals = build_yield_signals(answerable, history=history, manifest=manifest)
    already = signals["already_fetched"]

    # Default planning order (priority/weight based).
    frontier_objs = [
        FrontierTitle(f["title"], f.get("source", ""), f.get("reason", ""), f.get("weight", 1))
        for f in frontier_raw
    ]
    allowlist = build_expanded_allowlist(frontier_objs, 5000, 250, already)
    default_titles = [e.title for e in allowlist if not e.already_fetched][:250]

    yield_titles = select_yield_ranked_titles(_PUMP_DIR, snapshots_dir=_SNAPSHOTS)

    assert default_titles, "expected a non-empty default selection"
    assert yield_titles, "expected a non-empty yield-ranked selection"
    assert yield_titles != default_titles, "yield-ranked selection should differ from default"


# ---------------------------------------------------------------------------
# Test 11: no weak context promoted to answerable fact
# ---------------------------------------------------------------------------

def test_no_weak_context_promoted_to_answerable_fact() -> None:
    facts = _answerable_facts()  # includes one weak_context_only item
    rows, meta = build_batch_trace(_history_with_zero_recent_yield(), facts)
    # Only relations + definitions are counted; weak context + entity excluded.
    total_facts = sum(r["new_answerable_facts"] for r in rows)
    assert total_facts == 3  # 2 relations + 1 definition, NOT the weak/entity items
    assert meta["cumulative_answerable_facts"] == 3


def test_real_signals_exclude_weak_and_entity_from_fact_pages(real_result) -> None:
    _out, result = real_result
    answerable = json.loads((_PUMP_DIR / "pump_precision_answerable_delta.json").read_text())
    fact_pages_expected = {
        f.get("source_page") for f in answerable
        if f.get("overlay_type") in ("overlay_relation", "overlay_definition")
    }
    entity_only_pages = {
        f.get("source_page") for f in answerable if f.get("overlay_type") == "overlay_entity"
    }
    # Entity-only pages that never produce a relation/definition must not be
    # treated as fact-bearing source pages.
    from worldpgt.knowledge_pump.incremental_yield_trace import normalize_page
    confirmations = json.loads((_out / "yield_gate_report.json").read_text())["confirmations"]
    assert confirmations["weak_context_promoted_to_answerable_fact"] is False
    assert confirmations["entity_cards_counted_as_answerable_facts"] is False
    # sanity: at least one page produces real facts
    assert any(p for p in fact_pages_expected)


# ---------------------------------------------------------------------------
# Test 12: protected files unchanged
# ---------------------------------------------------------------------------

def test_protected_files_unchanged(real_result) -> None:
    before = {p: _sha(p) for p in _PROTECTED if p.exists()}
    # Re-run the frontier computation; it must not touch protected files.
    out, _ = real_result
    run_yield_ranked_frontier(_PUMP_DIR, out, snapshots_dir=_SNAPSHOTS)
    for path, sha in before.items():
        assert _sha(path) == sha, f"Protected file modified: {path}"


# ---------------------------------------------------------------------------
# Test 13: no network calls during the computation
# ---------------------------------------------------------------------------

def test_no_network_calls(monkeypatch, tmp_path) -> None:
    if not _PUMP_DIR.exists():
        pytest.skip("artifacts not found")

    def _blocked(*args, **kwargs):  # pragma: no cover - only hit on failure
        raise AssertionError("network access attempted during yield frontier")

    monkeypatch.setattr(socket.socket, "connect", _blocked)
    monkeypatch.setattr(socket, "create_connection", _blocked)
    result = run_yield_ranked_frontier(_PUMP_DIR, tmp_path / "out", snapshots_dir=_SNAPSHOTS)
    assert result["network_calls"] is False
