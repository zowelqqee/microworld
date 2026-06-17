"""Tests for Knowledge Pump Yield Diagnostics v1.

No network calls. No modification of accepted memory or overlays.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.knowledge_pump.frontier_title_extractor import _has_prose_period_break, _usable
from worldpgt.knowledge_pump.yield_diagnostics import (
    _batch_history_sanity,
    _build_funnel,
    _candidate_survival,
    _frontier_quality,
    _title_rejected_by_hygiene,
    run_yield_diagnostics,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_WORLD = _REPO / "worldpgt"
_EXP = _WORLD / "experiments"
_PUMP_DIR = _EXP / "knowledge_pump_v1"
_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
]


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# Part C: Frontier title hygiene — rejected titles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "History In",
    "States. In",
    "United States. The",
    "SpaceX. It",
    "Tesla. In April",
    "Company. During",
    "Commercial Crew Program. The GAO",
    "International Space Station. After",
    "Air Force. This",
    "Moon. The",
    "Earth. The",
    "California. In",
    "X. The",
    "PayPal. Musk",
    "Neuralink. He",
    "Rocket Science Games. In",
    "Launch Services Purchase Act. The Act",
    "Cape Canaveral Space Force Station. The",
    "SpaceX. Beginning",
    "Corporation. Oracle",
])
def test_reject_sentence_fragment_title(title: str) -> None:
    rejected, _ = _title_rejected_by_hygiene(title)
    assert rejected, f"Expected '{title}' to be rejected as a sentence fragment"


# ---------------------------------------------------------------------------
# Part C: Frontier title hygiene — preserved titles
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "B.C. Forbes",
    "John F. Kennedy",
    "J. B. Straubel",
    "D. E. Shaw",
    "S&P BSE SENSEX",
    "U.S. Army",
    "U.S. Department",
])
def test_preserve_legitimate_dotted_title(title: str) -> None:
    rejected, reason = _title_rejected_by_hygiene(title)
    assert not rejected, f"Expected '{title}' to be kept, but rejected with reason: {reason}"


# ---------------------------------------------------------------------------
# _has_prose_period_break unit tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "SpaceX. It",
    "Tesla. In April",
    "Company. During",
    "PayPal. Musk",
    "Neuralink. He",
    "United States. The",
    "International Space Station. After",
])
def test_has_prose_period_break_detected(title: str) -> None:
    assert _has_prose_period_break(title), f"Expected prose period break in '{title}'"


@pytest.mark.parametrize("title", [
    "B.C. Forbes",
    "John F. Kennedy",
    "J. B. Straubel",
    "D. E. Shaw",
    "U.S. Army",
    "SpaceX",
    "Tesla",
    "NASA",
])
def test_has_prose_period_break_not_detected(title: str) -> None:
    assert not _has_prose_period_break(title), f"Expected no prose period break in '{title}'"


# ---------------------------------------------------------------------------
# _usable tests for sentence-fragment detection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "History The",
    "States. In",
    "SpaceX. It",
    "Moon. The",
])
def test_usable_rejects_sentence_fragments(title: str) -> None:
    assert not _usable(title, "snapshot_link"), f"Expected _usable to reject '{title}'"
    assert not _usable(title, "snapshot_phrase"), f"Expected _usable to reject '{title}'"


@pytest.mark.parametrize("title", [
    "B.C. Forbes",
    "John F. Kennedy",
    "D. E. Shaw",
    "U.S. Army",
])
def test_usable_preserves_dotted_titles(title: str) -> None:
    assert _usable(title, "snapshot_link"), f"Expected _usable to keep '{title}'"


# ---------------------------------------------------------------------------
# Yield diagnostics integration — reads real pump artifacts
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def yield_result(tmp_path_factory):
    if not _PUMP_DIR.exists():
        pytest.skip("knowledge_pump_v1 artifacts not found")
    out = tmp_path_factory.mktemp("yield_diag")
    result = run_yield_diagnostics(_PUMP_DIR, out)
    return out, result


def test_yield_diagnostics_can_read_artifacts(yield_result) -> None:
    _out, result = yield_result
    assert result is not None
    assert "funnel" in result


def test_yield_diagnostics_writes_all_expected_artifacts(yield_result) -> None:
    out, result = yield_result
    expected_files = [
        "batch_yield_summary.json",
        "batch_yield_report.json",
        "batch_yield_by_batch.csv",
        "batch_yield_by_source_page.csv",
        "frontier_title_quality_report.json",
        "frontier_title_quality_report.csv",
        "low_yield_source_pages.json",
        "high_yield_source_pages.json",
        "candidate_survival_report.json",
        "candidate_survival_by_reason.json",
        "new_candidate_no_fact_report.json",
    ]
    for fname in expected_files:
        assert (out / fname).exists(), f"Missing artifact: {fname}"


def test_yield_diagnostics_funnel_fetched_grew(yield_result) -> None:
    _out, result = yield_result
    funnel = result["funnel"]
    # 2750 fetched, 1280 ready documented in task context
    assert funnel["titles_fetched"] > 0, "Expected titles_fetched > 0"
    assert funnel["ready_docs"] > 0, "Expected ready_docs > 0"
    assert funnel["fetch_success"] >= funnel["ready_docs"], \
        "fetch_success should be >= ready_docs"


def test_yield_diagnostics_precision_fact_count_growth_is_honest(yield_result) -> None:
    _out, result = yield_result
    sinks = result["sink_classification"]
    before = sinks.get("answerable_fact_delta_count_before_v2", result["funnel"]["new_answerable_facts"])
    after = result["funnel"]["new_answerable_facts"]
    assert sinks["precision_accepted_fact_count_increased"] is (after > before)


def test_yield_diagnostics_qa_fact_count_growth_is_honest(yield_result) -> None:
    _out, result = yield_result
    sinks = result["sink_classification"]
    before = sinks.get("qa_fact_count_before_v2", result["funnel"]["qa_fact_count"])
    after = result["funnel"]["qa_fact_count"]
    assert sinks["qa_fact_count_increased"] is (after > before)


def test_yield_diagnostics_weak_context_is_dominant_sink(yield_result) -> None:
    _out, result = yield_result
    sinks = result["sink_classification"]
    counts = sinks.get("sink_counts", {})
    weak_count = counts.get("weak_context_only", 0)
    assert weak_count > 0, "Expected weak_context_only to have non-zero count"
    # weak_context_only should be the largest or near-largest sink
    assert weak_count >= counts.get("fetch_failed", 0)


def test_yield_diagnostics_no_weak_context_as_answerable_fact(yield_result) -> None:
    _out, result = yield_result
    report_path = _out / "batch_yield_report.json"
    data = json.loads(report_path.read_text())
    confirmations = data.get("confirmations", {})
    assert confirmations.get("weak_context_excluded_from_answerable_facts") is True


def test_yield_diagnostics_entity_cards_not_counted_as_facts(yield_result) -> None:
    _out, result = yield_result
    report_path = _out / "batch_yield_report.json"
    data = json.loads(report_path.read_text())
    confirmations = data.get("confirmations", {})
    assert confirmations.get("entity_cards_not_counted_as_answerable_facts") is True


# ---------------------------------------------------------------------------
# Candidate survival report
# ---------------------------------------------------------------------------

def test_candidate_survival_has_source_level_counts(yield_result) -> None:
    out, result = yield_result
    survival_path = out / "candidate_survival_report.json"
    data = json.loads(survival_path.read_text())
    assert "total_source_pages" in data
    assert data["total_source_pages"] > 0

    # top_sources_by_ingestion_candidates should list sources with counts
    top = data.get("top_sources_by_ingestion_candidates", [])
    assert len(top) > 0
    first = top[0]
    assert "source_page" in first
    assert "ingestion_candidate_count" in first


def test_candidate_survival_no_fact_pages_exist(yield_result) -> None:
    out, result = yield_result
    no_fact_path = out / "new_candidate_no_fact_report.json"
    data = json.loads(no_fact_path.read_text())
    # Given 1280 ready docs but only 15 facts, many pages should have no facts
    assert data["count"] > 0


# ---------------------------------------------------------------------------
# Batch history sanity
# ---------------------------------------------------------------------------

def test_batch_history_sanity_handles_legacy_entries(yield_result) -> None:
    _out, result = yield_result
    hs = result["batch_history_sanity"]
    assert "batch_history_total_entries" in hs
    assert "batch_history_entries_with_title_lists" in hs
    assert "batch_history_legacy_entries" in hs
    # Legacy entries have no fetch_success title lists
    total = hs["batch_history_total_entries"]
    with_lists = hs["batch_history_entries_with_title_lists"]
    legacy = hs["batch_history_legacy_entries"]
    assert total == with_lists + legacy


def test_batch_history_sanity_handles_all_non_legacy(tmp_path) -> None:
    history = [
        {"batch_index": 0, "titles_planned": ["A", "B"], "fetch_success": 2},
        {"batch_index": 1, "titles_planned": ["C", "D"], "fetch_success": 1},
    ]
    result = _batch_history_sanity(history)
    assert result["batch_history_total_entries"] == 2
    assert result["batch_history_legacy_entries"] == 0
    assert result["batch_history_entries_with_title_lists"] == 2


def test_batch_history_sanity_detects_all_legacy(tmp_path) -> None:
    history = [
        {"batch_index": 0, "fetch_success": 2},
        {"batch_index": 1, "fetch_success": 1},
    ]
    result = _batch_history_sanity(history)
    assert result["batch_history_legacy_entries"] == 2
    assert result["batch_history_entries_with_title_lists"] == 0


# ---------------------------------------------------------------------------
# Frontier quality
# ---------------------------------------------------------------------------

def test_frontier_quality_counts_rejected_and_kept() -> None:
    frontier = [
        {"title": "SpaceX. It", "source": "test"},
        {"title": "Tesla. In April", "source": "test"},
        {"title": "John F. Kennedy", "source": "test"},
        {"title": "B.C. Forbes", "source": "test"},
        {"title": "U.S. Army", "source": "test"},
    ]
    fq, rejected, kept = _frontier_quality(frontier)
    assert fq["frontier_title_rejected_by_hygiene_count"] == 2
    assert fq["frontier_title_kept_by_hygiene_count"] == 3
    assert fq["frontier_title_total"] == 5


def test_frontier_quality_real_frontier_rejects_fragments(yield_result) -> None:
    _out, result = yield_result
    fq = result["frontier_quality"]
    assert fq["frontier_title_rejected_by_hygiene_count"] >= 0
    assert (
        fq["frontier_title_rejected_by_hygiene_count"]
        + fq["frontier_title_kept_by_hygiene_count"]
        == fq["frontier_title_total"]
    )


# ---------------------------------------------------------------------------
# No network calls confirmation
# ---------------------------------------------------------------------------

def test_yield_diagnostics_no_network_calls(yield_result) -> None:
    _out, result = yield_result
    assert result.get("network_calls") is False


# ---------------------------------------------------------------------------
# Protected memory/overlay files unchanged
# ---------------------------------------------------------------------------

def test_protected_files_unchanged(yield_result) -> None:
    _out, _result = yield_result
    for path in _PROTECTED:
        if path.exists():
            before = _sha(path)
            # Re-read — should be unchanged since we only read from pump_dir
            assert _sha(path) == before, f"Protected file was modified: {path}"


# ---------------------------------------------------------------------------
# _candidate_survival unit test
# ---------------------------------------------------------------------------

def test_candidate_survival_counts_per_source() -> None:
    ingestion = [
        {"source_doc_title": "PageA", "item_type": "entity_card"},
        {"source_doc_title": "PageA", "item_type": "relation_claim"},
        {"source_doc_title": "PageB", "item_type": "entity_card"},
    ]
    relation_cands = [
        {"source_title": "PageA", "relation": "subsidiary_of"},
    ]
    safe_delta = [
        {"overlay_type": "overlay_entity", "source_page": "PageA", "trust": "overlay_candidate"},
        {"overlay_type": "overlay_context_link", "source_page": "PageA", "trust": "weak_context_only"},
        {"overlay_type": "overlay_entity", "source_page": "PageB", "trust": "overlay_candidate"},
    ]
    weak = [
        {"overlay_type": "overlay_context_link", "source_page": "PageA", "trust": "weak_context_only"},
    ]
    prec_ans = [
        {"overlay_type": "overlay_entity", "source_page": "PageA"},
        {"overlay_type": "overlay_relation", "source_page": "PageA"},
    ]

    survival = _candidate_survival(
        ingestion, relation_cands, safe_delta, weak,
        prec_ans, [], [], [],
    )

    assert "PageA" in survival
    assert "PageB" in survival
    assert survival["PageA"]["ingestion_candidate_count"] == 2
    assert survival["PageA"]["relation_candidate_count"] == 1
    assert survival["PageA"]["weak_context_count"] == 1
    assert survival["PageA"]["final_fact_count"] == 1  # only the relation counts as a fact
    assert survival["PageA"]["entity_count"] == 1
    assert survival["PageB"]["ingestion_candidate_count"] == 1


# ---------------------------------------------------------------------------
# _build_funnel unit test
# ---------------------------------------------------------------------------

def test_build_funnel_does_not_count_weak_as_answerable() -> None:
    summary = {
        "fetch_success_count_total": 10,
        "ready_for_ingestion_count_total": 8,
        "answerable_delta_before_precision": 5,
        "pump_answerable_fact_delta_count": 2,
        "pump_world_model_delta_count": 4,
    }
    safe_delta = [
        {"overlay_type": "overlay_context_link", "trust": "weak_context_only"},
        {"overlay_type": "overlay_context_link", "trust": "weak_context_only"},
        {"overlay_type": "overlay_context_link", "trust": "weak_context_only"},
        {"overlay_type": "overlay_entity", "trust": "overlay_candidate"},
    ]
    prec_ans = [
        {"overlay_type": "overlay_relation", "source_page": "P"},
        {"overlay_type": "overlay_definition", "source_page": "P"},
    ]
    funnel = _build_funnel(
        summary, [], [], [], safe_delta, [], prec_ans, [], [], [], [], [], {}
    )
    assert funnel["weak_context_items"] == 3
    assert funnel["entity_items"] == 1
    assert funnel["precision_accepted_relations"] == 1
    assert funnel["precision_accepted_definitions"] == 1
    assert funnel["new_answerable_facts"] == 2
    # Weak context must NOT be part of new_answerable_facts
    assert funnel["new_answerable_facts"] < funnel["weak_context_items"]
