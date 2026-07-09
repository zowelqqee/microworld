"""Tests for benchmark_speech_quality_v1."""

from __future__ import annotations

import json

import pytest


def _fast_question_bank() -> list[dict]:
    from worldpgt.experiments.benchmark_speech_quality_v1 import QUESTION_BANK

    keep = {"gap-01", "thin-01", "profile-01", "rel-01", "adv-01", "audit-01"}
    return [item for item in QUESTION_BANK if item["id"] in keep]


def test_benchmark_speech_quality_imports_cleanly():
    import worldpgt.experiments.benchmark_speech_quality_v1 as mod

    assert callable(mod.run)
    assert callable(mod.main)


def test_question_bank_ids_unique():
    from worldpgt.experiments.benchmark_speech_quality_v1 import QUESTION_SUITES

    for suite in QUESTION_SUITES.values():
        ids = [item["id"] for item in suite]
        assert len(ids) == len(set(ids))


def test_question_bank_has_expected_contract_fields():
    from worldpgt.experiments.benchmark_speech_quality_v1 import QUESTION_SUITES

    required = {"id", "type", "q", "expected_gap", "must_contain"}
    for suite in QUESTION_SUITES.values():
        for item in suite:
            assert required <= set(item)
            assert "expected_decision" in item or "expected_decisions" in item
            assert item["q"].strip()


def test_large_question_bank_covers_working_modes():
    from worldpgt.experiments.benchmark_speech_quality_v1 import LARGE_QUESTION_BANK

    row_types = {item["type"] for item in LARGE_QUESTION_BANK}
    assert len(LARGE_QUESTION_BANK) >= 48
    assert {
        "profile", "thin_profile", "mechanism_gap", "direct_relation",
        "adversarial", "missing_or_current", "private_info",
    } <= row_types


def test_stress_question_bank_has_1000_rows_and_all_modes():
    from worldpgt.experiments.benchmark_speech_quality_v1 import STRESS_QUESTION_BANK

    row_types = {item["type"] for item in STRESS_QUESTION_BANK}
    assert len(STRESS_QUESTION_BANK) == 1000
    assert {
        "profile", "thin_profile", "mechanism_gap", "direct_relation",
        "connection", "adversarial", "missing_or_current", "private_info",
        "unsupported_universal", "style_control",
    } <= row_types


@pytest.fixture(scope="module")
def speech_quality_result():
    from worldpgt.experiments.benchmark_speech_quality_v1 import run

    return run(
        overlay_mode="pump-dry-run",
        question_bank=_fast_question_bank(),
        suite_name="test-smoke",
    )


@pytest.fixture(scope="module")
def large_speech_quality_result():
    from worldpgt.experiments.benchmark_speech_quality_v1 import LARGE_QUESTION_BANK, run

    return run(
        overlay_mode="pump-dry-run",
        question_bank=LARGE_QUESTION_BANK,
        suite_name="test-large",
    )


def test_speech_quality_result_shape(speech_quality_result):
    required = {
        "timestamp", "overlay_mode", "warmup_questions",
        "total_time_sec", "summary", "rows", "metric_version", "suite_name",
    }
    assert required <= set(speech_quality_result)
    assert speech_quality_result["overlay_mode"] == "pump-dry-run"
    assert speech_quality_result["metric_version"] == "speech_quality_v1"
    assert speech_quality_result["warmup_questions"] >= 1


def test_speech_quality_summary_shape(speech_quality_result):
    summary = speech_quality_result["summary"]
    required = {
        "total", "passed", "quality_rate", "debug_like", "repetitive",
        "decision_mismatch", "missing_required_text", "honest_gap",
        "expected_gap_total", "honest_gap_rate", "latency_ms",
        "answer_text", "decision_counts", "route_counts",
        "support_kind_counts", "source_system_counts", "flag_counts",
        "by_type",
    }
    assert required <= set(summary)
    assert summary["total"] == len(_fast_question_bank())
    assert summary["passed"] <= summary["total"]
    assert {"p50", "p90", "p95", "p99", "max"} <= set(summary["latency_ms"])


def test_speech_quality_rows_shape(speech_quality_result):
    for row in speech_quality_result["rows"]:
        required = {
            "id", "type", "question", "decision", "expected_decision",
            "answer_text", "answer_chars", "answer_words",
            "answer_sentences", "latency_ms", "route", "support_kind",
            "source_system", "supported_by_context", "safe_for_general_runtime",
            "risk_flags", "debug_like", "repetitive", "honest_gap",
            "decision_mismatch", "pass", "flags",
        }
        assert required <= set(row)
        assert row["latency_ms"] >= 0


def test_speech_quality_fast_set_has_no_debug_like_output(speech_quality_result):
    assert speech_quality_result["summary"]["debug_like"] == 0


def test_speech_quality_fast_set_has_honest_gaps(speech_quality_result):
    summary = speech_quality_result["summary"]
    assert summary["expected_gap_total"] >= 2
    assert summary["honest_gap"] == summary["expected_gap_total"]


def test_speech_quality_fast_set_has_no_decision_drift(speech_quality_result):
    assert speech_quality_result["summary"]["decision_mismatch"] == 0


def test_large_speech_quality_records_full_metric_surface(large_speech_quality_result):
    summary = large_speech_quality_result["summary"]

    assert summary["total"] >= 48
    assert summary["passed"] == summary["total"]
    assert summary["quality_rate"] == 1.0
    assert summary["debug_like"] == 0
    assert summary["repetitive"] == 0
    assert summary["decision_mismatch"] == 0
    assert summary["missing_required_text"] == 0
    assert summary["expected_gap_total"] >= 6
    assert summary["honest_gap"] == summary["expected_gap_total"]
    assert summary["latency_ms"]["p95"] >= 0
    assert summary["answer_text"]["words"]["p95"] > 0
    assert summary["decision_counts"]
    assert summary["route_counts"]
    assert summary["support_kind_counts"]
    assert summary["by_type"]


def test_large_speech_quality_rows_keep_failure_diagnostics(large_speech_quality_result):
    failed_rows = [row for row in large_speech_quality_result["rows"] if not row["pass"]]
    for row in large_speech_quality_result["rows"]:
        assert isinstance(row["flags"], list)
        assert isinstance(row["debug_hits"], list)
        assert isinstance(row["repeat_pairs"], list)
        assert isinstance(row["missing_required"], list)
        assert row["answer_text"].strip()
    for row in failed_rows:
        assert row["flags"]


def test_repetition_detector_flags_near_duplicate_adjacent_sentences():
    from worldpgt.experiments.benchmark_speech_quality_v1 import _repetitive_pairs

    pairs = _repetitive_pairs(
        "SpaceX develops rockets and spacecraft. "
        "It develops rockets and spacecraft for launches."
    )
    assert pairs


def test_main_no_save_does_not_write_files(tmp_path, monkeypatch, capsys):
    import worldpgt.experiments.benchmark_speech_quality_v1 as mod

    monkeypatch.setattr(mod, "_BENCHMARKS_DIR", tmp_path)
    monkeypatch.setattr(mod, "QUESTION_BANK", _fast_question_bank())
    monkeypatch.setitem(mod.QUESTION_SUITES, "smoke", _fast_question_bank())
    mod.main(["--overlay", "pump-dry-run", "--suite", "smoke", "--no-save"])
    assert list(tmp_path.iterdir()) == []
    assert "MICROWORLD SPEECH QUALITY BENCHMARK" in capsys.readouterr().out


def test_main_saves_json(tmp_path, monkeypatch):
    import worldpgt.experiments.benchmark_speech_quality_v1 as mod

    monkeypatch.setattr(mod, "_BENCHMARKS_DIR", tmp_path)
    monkeypatch.setattr(mod, "QUESTION_BANK", _fast_question_bank())
    monkeypatch.setitem(mod.QUESTION_SUITES, "smoke", _fast_question_bank())
    mod.main(["--overlay", "pump-dry-run", "--suite", "smoke"])
    files = list(tmp_path.glob("speech_quality_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["summary"]["total"] == len(_fast_question_bank())
