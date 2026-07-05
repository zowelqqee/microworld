"""Tests for benchmark_coverage_v1.

Verifies question bank invariants, JSON output shape, and that all 8 question
types are represented correctly.  Does not rely on specific answer/audit ratios
since those may shift as the overlay evolves.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent

_FAST_IDS = {
    "def-01", "def-02", "def-03", "def-04", "def-05",
    "rel-01", "mhop-01", "isa-01", "cmp-01", "cnt-01", "par-01",
    "adv-01", "adv-02", "adv-03", "adv-04", "adv-05", "adv-08", "adv-09", "adv-10",
}


def _fast_question_bank() -> list[dict]:
    from worldpgt.experiments.benchmark_coverage_v1 import QUESTION_BANK
    return [item for item in QUESTION_BANK if item["id"] in _FAST_IDS]


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------

def test_benchmark_coverage_imports_cleanly():
    import worldpgt.experiments.benchmark_coverage_v1 as mod
    assert callable(mod.run)
    assert callable(mod.main)


# ---------------------------------------------------------------------------
# Question bank invariants (no network, no overlay needed)
# ---------------------------------------------------------------------------

def test_question_bank_has_100_questions():
    from worldpgt.experiments.benchmark_coverage_v1 import QUESTION_BANK
    assert len(QUESTION_BANK) == 100


def test_question_bank_type_counts():
    from worldpgt.experiments.benchmark_coverage_v1 import QUESTION_BANK, TYPES_EXPECTED
    from collections import Counter
    counts = Counter(item["type"] for item in QUESTION_BANK)
    for qtype, expected in TYPES_EXPECTED.items():
        assert counts[qtype] == expected, (
            f"type '{qtype}': expected {expected}, got {counts[qtype]}"
        )


def test_question_bank_ids_unique():
    from worldpgt.experiments.benchmark_coverage_v1 import QUESTION_BANK
    ids = [item["id"] for item in QUESTION_BANK]
    assert len(ids) == len(set(ids)), "Duplicate IDs in QUESTION_BANK"


def test_question_bank_questions_non_empty():
    from worldpgt.experiments.benchmark_coverage_v1 import QUESTION_BANK
    for item in QUESTION_BANK:
        assert item["q"].strip(), f"Empty question for id={item['id']}"


# ---------------------------------------------------------------------------
# Tiny run with overlay
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def coverage_result():
    from worldpgt.experiments.benchmark_coverage_v1 import run
    return run(overlay_mode="pump-dry-run", question_bank=_fast_question_bank())


def test_coverage_result_has_required_keys(coverage_result):
    required = {
        "timestamp", "overlay_mode", "total_questions", "overlay_facts",
        "total_time_sec", "summary", "by_type", "rows",
    }
    assert required <= set(coverage_result.keys())


def test_coverage_total_questions(coverage_result):
    assert coverage_result["total_questions"] == len(_FAST_IDS)


def test_coverage_overlay_facts(coverage_result):
    assert coverage_result["overlay_facts"] >= 1299


def test_coverage_summary_sums_to_100(coverage_result):
    s = coverage_result["summary"]
    assert s["answer"] + s["no"] + s["audit"] == len(_FAST_IDS)


def test_coverage_summary_decisions_non_negative(coverage_result):
    s = coverage_result["summary"]
    for key in ("answer", "no", "audit"):
        assert s[key] >= 0


def test_coverage_by_type_all_types_present(coverage_result):
    from worldpgt.experiments.benchmark_coverage_v1 import TYPES_EXPECTED
    for qtype in TYPES_EXPECTED:
        assert qtype in coverage_result["by_type"], f"Missing type: {qtype}"


def test_coverage_by_type_counts_correct(coverage_result):
    fast_counts = {
        qtype: sum(1 for item in _fast_question_bank() if item["type"] == qtype)
        for qtype in coverage_result["by_type"]
    }
    for qtype, expected in fast_counts.items():
        bt = coverage_result["by_type"][qtype]
        assert bt["total"] == expected
        assert bt["answer"] + bt["no"] + bt["audit"] == expected


def test_coverage_rows_length(coverage_result):
    assert len(coverage_result["rows"]) == len(_FAST_IDS)


def test_coverage_rows_have_required_fields(coverage_result):
    for row in coverage_result["rows"]:
        for field in ("id", "type", "question", "decision", "latency_ms"):
            assert field in row, f"Missing field '{field}' in row {row.get('id')}"


def test_coverage_rows_decision_values(coverage_result):
    valid = {"answer", "no", "audit"}
    for row in coverage_result["rows"]:
        assert row["decision"] in valid, (
            f"Unexpected decision '{row['decision']}' for {row['id']}"
        )


def test_coverage_latency_ms_positive(coverage_result):
    for row in coverage_result["rows"]:
        assert row["latency_ms"] >= 0, f"Negative latency for {row['id']}"


# ---------------------------------------------------------------------------
# Adversarial questions must not answer (safety contract)
# ---------------------------------------------------------------------------

def test_adversarial_inversions_are_not_answered(coverage_result):
    """Inverted-relation adversarial questions must never produce 'answer'."""
    inversion_ids = {"adv-01", "adv-02", "adv-03", "adv-10"}
    adv_rows = {r["id"]: r for r in coverage_result["rows"] if r["id"] in inversion_ids}
    for qid, row in adv_rows.items():
        assert row["decision"] != "answer", (
            f"Safety violation: {qid} ({row['question']!r}) got decision='answer'"
        )


def test_current_live_adversarial_not_answered(coverage_result):
    """Current/live data requests must never produce 'answer'."""
    live_ids = {"adv-04", "adv-05"}
    live_rows = {r["id"]: r for r in coverage_result["rows"] if r["id"] in live_ids}
    for qid, row in live_rows.items():
        assert row["decision"] != "answer", (
            f"Safety violation: {qid} ({row['question']!r}) got decision='answer'"
        )


def test_private_adversarial_not_answered(coverage_result):
    """Private information requests must never produce 'answer'."""
    private_ids = {"adv-08", "adv-09"}
    priv_rows = {r["id"]: r for r in coverage_result["rows"] if r["id"] in private_ids}
    for qid, row in priv_rows.items():
        assert row["decision"] != "answer", (
            f"Safety violation: {qid} ({row['question']!r}) got decision='answer'"
        )


# ---------------------------------------------------------------------------
# Core definition questions must answer
# ---------------------------------------------------------------------------

def test_core_definitions_answered(coverage_result):
    """The most basic entity definitions should always be answered."""
    must_answer = {"def-01", "def-02", "def-03", "def-04", "def-05"}
    rows_by_id = {r["id"]: r for r in coverage_result["rows"]}
    for qid in must_answer:
        row = rows_by_id[qid]
        assert row["decision"] == "answer", (
            f"Expected 'answer' for {qid} ({row['question']!r}), got {row['decision']!r}"
        )


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------

def test_main_no_save_does_not_write_files(tmp_path, monkeypatch, capsys):
    import worldpgt.experiments.benchmark_coverage_v1 as mod
    monkeypatch.setattr(mod, "_BENCHMARKS_DIR", tmp_path)
    monkeypatch.setattr(mod, "QUESTION_BANK", _fast_question_bank())
    mod.main(["--overlay", "pump-dry-run", "--no-save"])
    assert list(tmp_path.iterdir()) == [], "Expected no files written with --no-save"
    out = capsys.readouterr().out
    assert "MICROWORLD COVERAGE BENCHMARK" in out


def test_main_saves_json(tmp_path, monkeypatch):
    import worldpgt.experiments.benchmark_coverage_v1 as mod
    monkeypatch.setattr(mod, "_BENCHMARKS_DIR", tmp_path)
    monkeypatch.setattr(mod, "QUESTION_BANK", _fast_question_bank())
    mod.main(["--overlay", "pump-dry-run"])
    files = list(tmp_path.glob("coverage_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["total_questions"] == len(_FAST_IDS)
    assert "by_type" in data
