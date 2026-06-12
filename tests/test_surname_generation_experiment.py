"""Tests for the surname audit summary, trust learning, and full experiment."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.surname_audit_summary import compute_summary, normalize_label
from examples.surname_generation_experiment import run_experiment
from examples.surname_trust_learn import (
    BAD_FACTOR,
    GOOD_FACTOR,
    TRUST_MAX,
    TRUST_MIN,
    learn_trust,
)

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


# ── fixtures ──────────────────────────────────────────────────────────────────

def _write_surnames(path) -> str:
    names = [
        "ivanov", "petrov", "sidorov", "abramov", "smirnov", "kuznetsov",
        "abramidze", "gelashvili", "saroyan", "petrosyan", "dmitriev",
        "volkov", "popov", "sokolov", "lebedev", "kozlov",
    ]
    p = path / "surnames.txt"
    p.write_text("\n".join(names) + "\n", encoding="utf-8")
    return str(p)


def _audit_row(name: str, label: str, score: str = "0.9", reasons: str = "") -> dict:
    return {
        "name": name,
        "quality_score": score,
        "quality_reasons": reasons,
        "duplicate": "false",
        "source": "baseline",
        "manual_label": label,
        "notes": "",
    }


# ── audit summary ─────────────────────────────────────────────────────────────

class TestAuditSummary:
    def test_precision_good_over_good_plus_bad(self):
        rows = (
            [_audit_row(f"g{i}", "good") for i in range(7)]
            + [_audit_row(f"b{i}", "bad") for i in range(3)]
        )
        summary = compute_summary(rows)
        assert summary["counts"]["good"] == 7
        assert summary["counts"]["bad"] == 3
        assert summary["generation_precision"] == pytest.approx(0.7)

    def test_unclear_excluded_from_denominator(self):
        rows = (
            [_audit_row("g1", "good")]
            + [_audit_row("b1", "bad")]
            + [_audit_row(f"u{i}", "unclear") for i in range(8)]
        )
        summary = compute_summary(rows)
        assert summary["generation_precision"] == pytest.approx(0.5)
        assert summary["counts"]["unclear"] == 8

    def test_zero_denominator(self):
        rows = [_audit_row("u1", "unclear"), _audit_row("u2", "unclear")]
        summary = compute_summary(rows)
        assert summary["generation_precision"] == 0.0

    def test_avg_quality_per_label(self):
        rows = [
            _audit_row("g1", "good", score="0.8"),
            _audit_row("g2", "good", score="1.0"),
            _audit_row("b1", "bad", score="0.2"),
        ]
        summary = compute_summary(rows)
        assert summary["avg_quality_good"] == pytest.approx(0.9)
        assert summary["avg_quality_bad"] == pytest.approx(0.2)

    def test_common_bad_reasons(self):
        rows = [
            _audit_row("b1", "bad", reasons="no vowels|weird consonant cluster"),
            _audit_row("b2", "bad", reasons="no vowels"),
        ]
        summary = compute_summary(rows)
        top = dict(summary["common_bad_reasons"])
        assert top["no vowels"] == 2
        assert top["weird consonant cluster"] == 1

    def test_label_normalization(self):
        assert normalize_label("GOOD") == "good"
        assert normalize_label(" b ") == "bad"
        assert normalize_label("weird") == "unclear"


# ── trust learning ────────────────────────────────────────────────────────────

class TestTrustLearning:
    def test_good_increases_trust(self):
        profile = learn_trust([_audit_row("ab", "good")], order=2)
        tt = profile["transition_trust"]
        # transition ab-><END> used by name "ab"
        assert tt["ab-><END>"] == pytest.approx(GOOD_FACTOR)
        assert all(v >= 1.0 for v in tt.values())

    def test_bad_decreases_trust(self):
        profile = learn_trust([_audit_row("ab", "bad")], order=2)
        tt = profile["transition_trust"]
        assert tt["ab-><END>"] == pytest.approx(BAD_FACTOR)
        assert all(v <= 1.0 for v in tt.values())

    def test_unclear_is_noop(self):
        profile = learn_trust([_audit_row("ab", "unclear")], order=2)
        assert profile["transition_trust"] == {}
        assert profile["stats"]["unclear"] == 1

    def test_trust_is_bounded(self):
        # 200 good labels on the same name would blow past 2.0 unbounded.
        rows = [_audit_row("ab", "good") for _ in range(200)]
        profile = learn_trust(rows, order=2)
        for v in profile["transition_trust"].values():
            assert TRUST_MIN <= v <= TRUST_MAX
        assert max(profile["transition_trust"].values()) == pytest.approx(TRUST_MAX)

    def test_trust_lower_bound(self):
        rows = [_audit_row("ab", "bad") for _ in range(200)]
        profile = learn_trust(rows, order=2)
        for v in profile["transition_trust"].values():
            assert v >= TRUST_MIN
        assert min(profile["transition_trust"].values()) == pytest.approx(TRUST_MIN)

    def test_json_schema_stable(self):
        profile = learn_trust(
            [_audit_row("ab", "good"), _audit_row("cd", "bad")], order=2
        )
        assert set(profile.keys()) == {"order", "transition_trust", "stats"}
        assert profile["order"] == 2
        assert set(profile["stats"].keys()) == {
            "reviewed", "good", "bad", "unclear"
        }
        # round-trips through JSON unchanged
        assert json.loads(json.dumps(profile)) == profile

    def test_stats_counts(self):
        rows = [
            _audit_row("ab", "good"),
            _audit_row("cd", "bad"),
            _audit_row("ef", "unclear"),
        ]
        profile = learn_trust(rows, order=2)
        assert profile["stats"] == {
            "reviewed": 3, "good": 1, "bad": 1, "unclear": 1
        }


# ── full experiment ───────────────────────────────────────────────────────────

class TestExperiment:
    def test_baseline_only_without_trust(self, tmp_path):
        input_path = _write_surnames(tmp_path)
        result = run_experiment(
            input_path,
            order=2,
            count=20,
            seed=42,
            baseline_output=str(tmp_path / "baseline.csv"),
            learned_output=str(tmp_path / "learned.csv"),
            trust_profile_path=None,
        )
        assert result["learned"] is None
        assert result["baseline"]["count"] > 0
        assert os.path.exists(str(tmp_path / "baseline.csv"))

    def test_runs_learned_when_trust_present(self, tmp_path):
        input_path = _write_surnames(tmp_path)
        trust_path = tmp_path / "trust.json"
        trust_path.write_text(
            json.dumps({"order": 2, "transition_trust": {"<START>a->b": 1.5}}),
            encoding="utf-8",
        )
        result = run_experiment(
            input_path,
            order=2,
            count=20,
            seed=42,
            baseline_output=str(tmp_path / "baseline.csv"),
            learned_output=str(tmp_path / "learned.csv"),
            trust_profile_path=str(trust_path),
        )
        assert result["learned"] is not None
        assert result["learned"]["count"] > 0
        assert os.path.exists(str(tmp_path / "learned.csv"))


# ── subprocess smoke tests ────────────────────────────────────────────────────

class TestSubprocessSmoke:
    def test_surname_generate_cli(self, tmp_path):
        input_path = _write_surnames(tmp_path)
        output_path = tmp_path / "generated.csv"
        result = subprocess.run(
            [
                sys.executable, "examples/surname_generate.py",
                "--input", input_path,
                "--output", str(output_path),
                "--count", "10",
                "--order", "2",
                "--seed", "42",
            ],
            cwd=_ROOT, check=True, text=True, capture_output=True,
        )
        assert "Generated" in result.stderr
        assert output_path.exists()
        with open(output_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows
        assert set(rows[0].keys()) >= {
            "name", "quality_score", "quality_reasons", "duplicate",
            "source", "manual_label", "notes",
        }

    def test_surname_audit_summary_cli(self, tmp_path):
        audit = tmp_path / "audit.csv"
        with open(audit, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name", "quality_score", "quality_reasons", "duplicate",
                    "source", "manual_label", "notes",
                ],
            )
            writer.writeheader()
            writer.writerow(_audit_row("ivanov", "good", score="0.9"))
            writer.writerow(_audit_row("petrov", "good", score="0.8"))
            writer.writerow(_audit_row("qzxqz", "bad", score="0.1",
                                       reasons="no vowels"))
        result = subprocess.run(
            [
                sys.executable, "examples/surname_audit_summary.py",
                "--input", str(audit),
            ],
            cwd=_ROOT, check=True, text=True, capture_output=True,
        )
        assert "Generation precision" in result.stdout
        assert "0.6667" in result.stdout  # 2 good / 3

    def test_surname_trust_learn_cli(self, tmp_path):
        audit = tmp_path / "audit.csv"
        with open(audit, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name", "quality_score", "quality_reasons", "duplicate",
                    "source", "manual_label", "notes",
                ],
            )
            writer.writeheader()
            writer.writerow(_audit_row("ivanov", "good"))
            writer.writerow(_audit_row("qzxqz", "bad"))
        out = tmp_path / "trust.json"
        result = subprocess.run(
            [
                sys.executable, "examples/surname_trust_learn.py",
                "--input", str(audit),
                "--order", "2",
                "--output", str(out),
            ],
            cwd=_ROOT, check=True, text=True, capture_output=True,
        )
        assert "Reviewed" in result.stderr
        assert out.exists()
        profile = json.loads(out.read_text(encoding="utf-8"))
        assert profile["order"] == 2
        assert profile["stats"]["good"] == 1
        assert profile["stats"]["bad"] == 1
        assert profile["transition_trust"]  # non-empty
