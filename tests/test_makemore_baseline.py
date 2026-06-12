"""Tests for examples/makemore_baseline.py."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples import makemore_baseline as makemore

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def _write_names(tmp_path) -> str:
    path = tmp_path / "names.txt"
    path.write_text("anna\nanne\nanya\nivan\nirina\nmarina\nmaria\n", encoding="utf-8")
    return str(path)


def test_write_makemore_audit_csv_has_expected_columns(tmp_path):
    out = tmp_path / "makemore.csv"
    result = makemore.write_makemore_audit_csv(["anna", "maria"], str(out))
    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    assert result["written"] is True
    assert reader.fieldnames == ["name", "manual_label", "notes"]
    assert [row["name"] for row in rows] == ["anna", "maria"]
    assert rows[0]["manual_label"] == ""


def test_existing_manual_labels_are_not_overwritten_without_force(tmp_path):
    out = tmp_path / "makemore.csv"
    out.write_text("name,manual_label,notes\nanna,good,keep me\n", encoding="utf-8")
    result = makemore.write_makemore_audit_csv(["maria"], str(out))
    assert result["written"] is False
    assert "anna,good,keep me" in out.read_text(encoding="utf-8")


def test_force_allows_overwriting_existing_manual_labels(tmp_path):
    out = tmp_path / "makemore.csv"
    out.write_text("name,manual_label,notes\nanna,good,replace me\n", encoding="utf-8")
    result = makemore.write_makemore_audit_csv(["maria"], str(out), force=True)
    assert result["written"] is True
    assert "maria,," in out.read_text(encoding="utf-8")
    assert "replace me" not in out.read_text(encoding="utf-8")


def test_run_baseline_optional_torch_branch(tmp_path):
    out = tmp_path / "makemore.csv"
    result = makemore.run_baseline(
        _write_names(tmp_path),
        count=2,
        seed=1,
        steps=2,
        embedding_dim=4,
        hidden_dim=8,
        block_size=2,
        batch_size=2,
        output_csv=str(out),
        force=True,
    )
    assert os.path.exists(out)
    if result["available"]:
        assert result["initial_loss"] is not None
        assert result["final_loss"] is not None
        assert result["trainable_parameter_count"] > 0
        assert result["parameter_count"] == result["trainable_parameter_count"]
        assert result["model_state_size_bytes"] > 0
        assert result["generated_count"] == 2
        assert result["uses_backpropagation"] is True
        assert result["uses_neural_weights"] is True
        assert "memory" in result
        assert "training_peak_rss_bytes" in result["memory"]
        assert "generation_peak_rss_bytes" in result["memory"]
        if result["memory"]["available"]:
            assert isinstance(result["memory"]["training_peak_rss_bytes"], int)
            assert result["memory"]["training_peak_rss_bytes"] >= 0
    else:
        assert result["skipped_reason"].startswith("PyTorch unavailable")


def test_cli_smoke_writes_metrics_and_csv_without_requiring_torch(tmp_path):
    out = tmp_path / "makemore.csv"
    metrics = tmp_path / "metrics.json"
    subprocess.run(
        [
            sys.executable,
            "examples/makemore_baseline.py",
            "--input",
            _write_names(tmp_path),
            "--count",
            "2",
            "--seed",
            "1",
            "--steps",
            "1",
            "--hidden-dim",
            "8",
            "--embedding-dim",
            "4",
            "--block-size",
            "2",
            "--batch-size",
            "2",
            "--output",
            str(out),
            "--metrics-output",
            str(metrics),
            "--force",
            "true",
        ],
        cwd=_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    data = json.loads(metrics.read_text(encoding="utf-8"))
    assert "available" in data
    assert "memory" in data
    assert os.path.exists(out)
    with open(out, newline="", encoding="utf-8") as f:
        assert csv.DictReader(f).fieldnames == ["name", "manual_label", "notes"]
