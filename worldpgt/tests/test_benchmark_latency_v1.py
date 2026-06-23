"""Tests for benchmark_latency_v1.

Uses a tiny question pool and --no-save to avoid writing to disk.
All assertions are structural: verify JSON shape, required fields,
and numeric sanity checks only.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Import smoke test
# ---------------------------------------------------------------------------

def test_benchmark_latency_imports_cleanly():
    import worldpgt.experiments.benchmark_latency_v1 as mod
    assert callable(mod.run)
    assert callable(mod.main)


# ---------------------------------------------------------------------------
# Tiny run (10 questions, no save)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def latency_result():
    from worldpgt.experiments.benchmark_latency_v1 import run
    return run(overlay_mode="pump-dry-run", target=10, warmup=2)


def test_latency_result_has_required_keys(latency_result):
    required = {
        "timestamp", "overlay_mode", "questions", "overlay_facts",
        "warmup", "latency_ms", "throughput", "memory_mb",
        "gpt4_comparison", "hardware",
    }
    assert required <= set(latency_result.keys())


def test_latency_ms_has_all_percentiles(latency_result):
    lat = latency_result["latency_ms"]
    for key in ("p50", "p95", "p99", "min", "max", "mean"):
        assert key in lat, f"missing key: {key}"
        assert isinstance(lat[key], (int, float))
        assert lat[key] >= 0


def test_latency_ordering(latency_result):
    lat = latency_result["latency_ms"]
    assert lat["min"] <= lat["p50"] <= lat["p95"] <= lat["p99"] <= lat["max"]


def test_throughput_positive(latency_result):
    tp = latency_result["throughput"]
    assert tp["requests_per_sec"] > 0
    assert tp["total_time_sec"] > 0


def test_memory_fields_present(latency_result):
    mem = latency_result["memory_mb"]
    assert "overlay_load" in mem
    assert "peak_rss" in mem
    assert mem["peak_rss"] > 0


def test_question_count_matches_target(latency_result):
    assert latency_result["questions"] == 10


def test_overlay_facts_matches_pump_dry_run(latency_result):
    assert latency_result["overlay_facts"] >= 1299


def test_gpt4_speedup_positive(latency_result):
    cmp = latency_result["gpt4_comparison"]
    assert cmp["speedup_vs_gpt4"] > 0
    assert cmp["gpt4_typical_ms"] == 800.0


def test_hardware_string_non_empty(latency_result):
    assert isinstance(latency_result["hardware"], str)
    assert len(latency_result["hardware"]) > 0


# ---------------------------------------------------------------------------
# CLI / save integration
# ---------------------------------------------------------------------------

def test_main_no_save_does_not_write_files(tmp_path, monkeypatch, capsys):
    import worldpgt.experiments.benchmark_latency_v1 as mod
    monkeypatch.setattr(mod, "_BENCHMARKS_DIR", tmp_path)
    mod.main(["--overlay", "pump-dry-run", "--questions", "5", "--warmup", "1", "--no-save"])
    assert list(tmp_path.iterdir()) == [], "Expected no files written with --no-save"
    out = capsys.readouterr().out
    assert "MICROWORLD LATENCY BENCHMARK" in out


def test_main_saves_json(tmp_path, monkeypatch):
    import worldpgt.experiments.benchmark_latency_v1 as mod
    monkeypatch.setattr(mod, "_BENCHMARKS_DIR", tmp_path)
    mod.main(["--overlay", "pump-dry-run", "--questions", "5", "--warmup", "1"])
    files = list(tmp_path.glob("latency_*.json"))
    assert len(files) == 1
    data = json.loads(files[0].read_text())
    assert data["questions"] == 5


def test_question_pool_reaches_target():
    from worldpgt.experiments.benchmark_latency_v1 import _build_question_pool
    from worldpgt.assistant_surface.context_selector import resolve_overlay
    path, _ = resolve_overlay("pump-dry-run")
    pool = _build_question_pool(path, target=50)
    assert len(pool) == 50
    assert len(set(pool)) == 50  # no duplicates up to target


def test_question_pool_1000_unique():
    from worldpgt.experiments.benchmark_latency_v1 import _build_question_pool
    from worldpgt.assistant_surface.context_selector import resolve_overlay
    path, _ = resolve_overlay("pump-dry-run")
    pool = _build_question_pool(path, target=1000)
    assert len(pool) == 1000
