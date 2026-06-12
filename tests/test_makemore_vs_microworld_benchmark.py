"""Tests for examples/makemore_vs_microworld_benchmark.py."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples import makemore_vs_microworld_benchmark as bench

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _write_names(tmp_path) -> str:
    p = tmp_path / "names.txt"
    p.write_text("anna\nanne\nanya\nivan\nirina\nmarina\nmaria\n", encoding="utf-8")
    return str(p)


def _write_audit(tmp_path) -> str:
    p = tmp_path / "audit.csv"
    p.write_text(
        "name,manual_label,notes\n"
        "anna,good,\n"
        "qwe,bad,\n"
        "kha,bad,\n"
        "maybe,unclear,\n",
        encoding="utf-8",
    )
    return str(p)


def test_quality_parser_computes_precision(tmp_path):
    metrics = bench.quality_metrics_from_audit(_write_audit(tmp_path))
    assert metrics["reviewed"] == 4
    assert metrics["good"] == 1
    assert metrics["bad"] == 2
    assert metrics["unclear"] == 1
    assert metrics["generation_precision"] == 1 / 3


def test_generated_makemore_csv_has_audit_columns(tmp_path):
    out = tmp_path / "makemore.csv"
    bench.write_makemore_audit_csv(["anna", "maria"], str(out))
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert reader.fieldnames == ["name", "manual_label", "notes"]
    assert rows[0]["name"] == "anna"
    assert rows[0]["manual_label"] == ""


def test_benchmark_json_has_required_top_level_keys(tmp_path):
    result = bench.run_benchmark(
        input_path=_write_names(tmp_path),
        count=5,
        order=2,
        seed=1,
        audit_path=_write_audit(tmp_path),
        trust_profile_path=None,
        run_neural=False,
        makemore_steps=1,
        makemore_embedding_dim=4,
        makemore_hidden_dim=8,
        makemore_block_size=2,
        makemore_csv=str(tmp_path / "makemore.csv"),
    )
    assert set(result) == {
        "input",
        "microworld",
        "makemore",
        "quality",
        "efficiency_ratios",
        "memory_notes",
    }
    assert result["microworld"]["trainable_parameter_count"] == 0
    assert result["microworld"]["uses_backpropagation"] is False
    assert result["microworld"]["uses_neural_weights"] is False
    assert "memory" in result["microworld"]
    assert "build_peak_rss_bytes" in result["microworld"]["memory"]
    assert "generation_peak_rss_bytes" in result["microworld"]["memory"]
    assert "audit_adaptation_peak_rss_bytes" in result["microworld"]["memory"]
    assert "memory" in result["makemore"]
    assert "training_peak_rss_bytes" in result["makemore"]["memory"]
    assert result["memory_notes"]
    assert result["makemore"]["available"] is False
    assert result["makemore"]["skipped_reason"]
    assert result["efficiency_ratios"]["parameter_count_ratio_makemore_to_microworld"] is None


def test_efficiency_ratios_use_null_safely():
    assert bench._safe_ratio(1, 0) is None
    assert bench._safe_ratio(1, -1) is None
    assert bench._safe_ratio(None, 1) is None
    assert bench._safe_ratio(4, 2) == 2


def test_torch_availability_branch(tmp_path):
    result = bench.run_makemore(
        _write_names(tmp_path),
        count=2,
        seed=1,
        steps=1,
        embedding_dim=4,
        hidden_dim=8,
        block_size=2,
        output_csv=str(tmp_path / "makemore.csv"),
    )
    if result["available"]:
        assert result["trainable_parameter_count"] > 0
        assert result["model_state_size_bytes"] > 0
        assert result["generated_count"] == 2
        assert "memory" in result
        assert "training_peak_rss_bytes" in result["memory"]
    else:
        assert result["skipped_reason"]
    assert os.path.exists(str(tmp_path / "makemore.csv"))


def test_cli_smoke_writes_json_and_makemore_csv(tmp_path):
    output = tmp_path / "benchmark.json"
    makemore_csv = tmp_path / "makemore.csv"
    subprocess.run(
        [
            sys.executable,
            "examples/makemore_vs_microworld_benchmark.py",
            "--input",
            _write_names(tmp_path),
            "--count",
            "5",
            "--order",
            "2",
            "--seed",
            "1",
            "--audit",
            _write_audit(tmp_path),
            "--run-makemore",
            "false",
            "--output",
            str(output),
            "--makemore-output",
            str(makemore_csv),
        ],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["makemore"]["available"] is False
    assert data["microworld"]["generated_count"] > 0
    assert "memory_notes" in data
    assert "memory" in data["microworld"]
    assert "memory" in data["makemore"]
    with open(makemore_csv, newline="", encoding="utf-8") as f:
        assert csv.DictReader(f).fieldnames == ["name", "manual_label", "notes"]


def test_memory_ratios_are_null_safe_in_benchmark(tmp_path):
    result = bench.run_benchmark(
        input_path=_write_names(tmp_path),
        count=2,
        order=2,
        seed=1,
        audit_path=None,
        trust_profile_path=None,
        run_neural=False,
        makemore_steps=1,
        makemore_embedding_dim=4,
        makemore_hidden_dim=8,
        makemore_block_size=2,
        makemore_csv=str(tmp_path / "makemore.csv"),
        track_memory=False,
    )
    ratios = result["efficiency_ratios"]
    assert ratios["peak_training_rss_ratio_makemore_to_microworld_build"] is None
    assert ratios["peak_generation_rss_ratio_makemore_to_microworld_generation"] is None
    assert ratios["rss_delta_training_ratio_makemore_to_microworld_build"] is None
    assert ratios["rss_delta_generation_ratio_makemore_to_microworld_generation"] is None
