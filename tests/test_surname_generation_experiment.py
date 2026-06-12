"""Tests for the surname audit summary, trust learning, and full experiment."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.surname_audit_summary import (
    compute_summary,
    format_summary,
    normalize_label,
    read_labeled_rows,
)
from examples.surname_generation_experiment import run_experiment
from examples.surname_generate import adjusted_quality_score, build_rows, generate_names
from examples.surname_trust_learn import (
    BAD_FACTOR,
    GOOD_FACTOR,
    TRUST_MAX,
    TRUST_MIN,
    UNCLEAR_FACTOR,
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


def _write_compact_audit(path, rows, delimiter: str = ",") -> str:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["name", "manual_label", "notes"],
            delimiter=delimiter,
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(path)


def _good_unclear_rows(good: int = 61, unclear: int = 39) -> list[dict]:
    rows = [
        {"name": f"goodname{i}", "manual_label": "good", "notes": ""}
        for i in range(good)
    ]
    rows.extend(
        {"name": f"unclearname{i}", "manual_label": "unclear", "notes": ""}
        for i in range(unclear)
    )
    return rows


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

    def test_reports_label_rates(self):
        rows = (
            [_audit_row(f"g{i}", "good") for i in range(6)]
            + [_audit_row(f"b{i}", "bad") for i in range(2)]
            + [_audit_row(f"u{i}", "unclear") for i in range(2)]
        )
        summary = compute_summary(rows)
        assert summary["generation_precision"] == pytest.approx(0.75)
        assert summary["good_rate"] == pytest.approx(0.6)
        assert summary["bad_rate"] == pytest.approx(0.2)
        assert summary["unclear_rate"] == pytest.approx(0.2)
        rendered = format_summary(summary)
        assert "Good rate" in rendered
        assert "Bad rate" in rendered
        assert "Unclear rate" in rendered


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

    def test_unclear_is_weak_negative_by_default(self):
        profile = learn_trust([_audit_row("ab", "unclear")], order=2)
        assert profile["transition_trust"]["ab-><END>"] == pytest.approx(
            UNCLEAR_FACTOR
        )
        assert profile["stats"]["unclear"] == 1

    def test_default_multipliers(self):
        assert GOOD_FACTOR == pytest.approx(1.04)
        assert UNCLEAR_FACTOR == pytest.approx(0.97)
        assert BAD_FACTOR == pytest.approx(0.80)

    def test_unclear_multiplier_can_restore_noop(self):
        profile = learn_trust(
            [_audit_row("ab", "unclear")],
            order=2,
            unclear_multiplier=1.0,
        )
        assert profile["transition_trust"]["ab-><END>"] == pytest.approx(1.0)

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
        assert set(profile.keys()) == {
            "order", "transition_trust", "shape_trust", "pattern_trust",
            "pattern_stats", "stats"
        }
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

    def test_bad_decreases_transition_trust_more_than_unclear(self):
        unclear = learn_trust([_audit_row("ab", "unclear")], order=2)
        bad = learn_trust([_audit_row("ab", "bad")], order=2)
        assert bad["transition_trust"]["ab-><END>"] < unclear["transition_trust"]["ab-><END>"]

    def test_reads_compact_manual_audit_file(self, tmp_path):
        path = tmp_path / "compact.csv"
        _write_compact_audit(
            path,
            [
                {"name": "majanis", "manual_label": "Good", "notes": "ok"},
                {"name": "jahd", "manual_label": "Unclear", "notes": "hard"},
                {"name": "chelis", "manual_label": " good ", "notes": "ok"},
            ],
        )
        rows = read_labeled_rows(str(path))
        profile = learn_trust(rows, order=3)
        assert profile["stats"] == {
            "reviewed": 3, "good": 2, "bad": 0, "unclear": 1
        }
        assert profile["transition_trust"]

    def test_reads_compact_tsv(self, tmp_path):
        path = tmp_path / "compact.tsv"
        _write_compact_audit(path, _good_unclear_rows(2, 1), delimiter="\t")
        profile = learn_trust(read_labeled_rows(str(path)), order=2)
        assert profile["stats"]["good"] == 2
        assert profile["stats"]["unclear"] == 1
        assert profile["transition_trust"]

    def test_reads_compact_csv(self, tmp_path):
        path = tmp_path / "compact.csv"
        _write_compact_audit(path, _good_unclear_rows(2, 1), delimiter=",")
        profile = learn_trust(read_labeled_rows(str(path)), order=2)
        assert profile["stats"]["good"] == 2
        assert profile["stats"]["unclear"] == 1
        assert profile["transition_trust"]

    def test_compact_fixture_counts_good_and_unclear(self, tmp_path):
        path = tmp_path / "compact.csv"
        _write_compact_audit(path, _good_unclear_rows())
        profile = learn_trust(read_labeled_rows(str(path)), order=3)
        assert profile["stats"] == {
            "reviewed": 100, "good": 61, "bad": 0, "unclear": 39
        }

    def test_good_rows_increase_trust_when_bad_is_zero(self, tmp_path):
        path = tmp_path / "compact.csv"
        _write_compact_audit(
            path,
            [
                {"name": "ab", "manual_label": "good", "notes": ""},
                {"name": "xy", "manual_label": "unclear", "notes": ""},
            ],
        )
        profile = learn_trust(read_labeled_rows(str(path)), order=2)
        assert profile["stats"]["bad"] == 0
        assert profile["transition_trust"]
        assert any(value > 1.0 for value in profile["transition_trust"].values())

    def test_unclear_rows_weakly_decrease_transition_trust(self, tmp_path):
        path = tmp_path / "compact.csv"
        _write_compact_audit(
            path,
            [
                {"name": "ab", "manual_label": "good", "notes": ""},
                {"name": "xy", "manual_label": "unclear", "notes": ""},
            ],
        )
        profile = learn_trust(read_labeled_rows(str(path)), order=2)
        assert profile["transition_trust"]["xy-><END>"] == pytest.approx(
            UNCLEAR_FACTOR
        )
        assert profile["transition_trust"]["<START>x->y"] == pytest.approx(
            UNCLEAR_FACTOR
        )

    def test_shape_trust_is_written_to_profile(self):
        profile = learn_trust(
            [_audit_row("ab", "bad", reasons="long_name")],
            order=2,
        )
        assert "shape_trust" in profile
        assert profile["shape_trust"]["long_name"] == pytest.approx(BAD_FACTOR)

    def test_diagnostic_quality_reasons_are_learned(self):
        profile = learn_trust(
            [
                _audit_row(
                    "ab",
                    "unclear",
                    reasons=(
                        "no vowels|weird consonant cluster|too_long|"
                        "common_word_like|brand_like|too_fragmentary|"
                        "awkward_short_form|medium_glued_name|poor_readability|"
                        "weird_q_usage|nickname_like|distorted_known_name"
                    ),
                )
            ],
            order=2,
        )
        assert profile["shape_trust"]["no_vowels"] == pytest.approx(UNCLEAR_FACTOR)
        assert profile["shape_trust"]["weird_consonant_cluster"] == pytest.approx(
            UNCLEAR_FACTOR
        )
        assert profile["shape_trust"]["too_long"] == pytest.approx(UNCLEAR_FACTOR)
        for reason in (
            "common_word_like",
            "brand_like",
            "too_fragmentary",
            "awkward_short_form",
            "medium_glued_name",
            "poor_readability",
            "weird_q_usage",
            "nickname_like",
            "distorted_known_name",
        ):
            assert profile["shape_trust"][reason] == pytest.approx(UNCLEAR_FACTOR)

    def test_compact_audit_recomputes_missing_quality_reasons_for_shape_trust(self):
        profile = learn_trust(
            [{"name": "ab", "manual_label": "unclear", "notes": ""}],
            order=2,
        )
        assert profile["shape_trust"]["very_short"] == pytest.approx(UNCLEAR_FACTOR)

    def test_generic_positive_reasons_ignored_for_shape_trust(self):
        profile = learn_trust(
            [
                _audit_row(
                    "ab",
                    "bad",
                    reasons=(
                        "looks like a plausible name|reasonable_length|"
                        "balanced_vowels|common_name_ending"
                    ),
                )
            ],
            order=2,
        )
        assert profile["shape_trust"] == {}

    def test_good_does_not_boost_negative_diagnostic_shape_reasons(self):
        profile = learn_trust(
            [_audit_row("ab", "good", reasons="long_name")],
            order=2,
        )
        assert profile["shape_trust"] == {}

    def test_summary_and_trust_learner_agree_on_counts(self, tmp_path):
        path = tmp_path / "compact.tsv"
        _write_compact_audit(path, _good_unclear_rows(), delimiter="\t")
        rows = read_labeled_rows(str(path))
        summary = compute_summary(rows)
        profile = learn_trust(rows, order=3)
        assert summary["total"] == profile["stats"]["reviewed"]
        assert summary["counts"] == {
            "good": profile["stats"]["good"],
            "bad": profile["stats"]["bad"],
            "unclear": profile["stats"]["unclear"],
        }


# ── adjusted quality generation ──────────────────────────────────────────────

class TestAdjustedQualityGeneration:
    def test_adjusted_quality_score_penalizes_shape_reasons(self):
        score = adjusted_quality_score(
            0.8,
            ["very short", "reasonable_length"],
            {"shape_trust": {"very_short": 0.5}},
        )
        assert score == pytest.approx(0.4)

    def test_generation_rows_include_adjusted_quality_score(self):
        rows = build_rows(
            ["ab"],
            set(),
            "learned",
            {"shape_trust": {"very_short": 0.5}},
        )
        assert "adjusted_quality_score" in rows[0]
        assert float(rows[0]["adjusted_quality_score"]) < float(
            rows[0]["quality_score"]
        )

    def test_adjusted_quality_score_penalizes_bad_patterns(self):
        score = adjusted_quality_score(
            1.0,
            ["looks like a plausible name"],
            {"pattern_trust": {"qwe": 0.7}},
            name="qweslienna",
        )
        assert score == pytest.approx(0.7)

    def test_generation_rows_include_pattern_reasons(self):
        rows = build_rows(
            ["qweslienna"],
            set(),
            "learned",
            {"pattern_trust": {"qwe": 0.7}},
        )
        assert "pattern_reasons" in rows[0]
        assert "bad_pattern:qwe" in rows[0]["pattern_reasons"]
        assert float(rows[0]["adjusted_quality_score"]) < float(
            rows[0]["quality_score"]
        )

    def test_min_adjusted_quality_resampling_filters_low_quality_candidates(self):
        class StubGraph:
            def __init__(self):
                self.names = iter(["ab", "ivan"])

            def generate(self, **kwargs):
                return next(self.names)

        names = generate_names(
            StubGraph(),
            1,
            rng=None,
            source_set=set(),
            trust_profile={"shape_trust": {"very_short": 0.1}},
            min_adjusted_quality=0.7,
            max_attempts_per_name=5,
        )
        assert names == ["ivan"]


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
            "name", "quality_score", "adjusted_quality_score",
            "quality_reasons", "pattern_reasons", "duplicate",
            "source", "manual_label", "notes",
        }
        for row in rows:
            float(row["adjusted_quality_score"])

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
        assert "pattern_trust" in profile
        assert "pattern_stats" in profile

    def test_surname_trust_learn_cli_mines_bad_patterns(self, tmp_path):
        audit = tmp_path / "audit.csv"
        with open(audit, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["name", "manual_label", "quality_reasons"],
            )
            writer.writeheader()
            writer.writerow({"name": "qweslienna", "manual_label": "bad"})
            writer.writerow({"name": "qwemara", "manual_label": "bad"})
            writer.writerow({"name": "majanis", "manual_label": "good"})
        out = tmp_path / "trust.json"
        subprocess.run(
            [
                sys.executable, "examples/surname_trust_learn.py",
                "--input", str(audit),
                "--order", "2",
                "--output", str(out),
            ],
            cwd=_ROOT, check=True, text=True, capture_output=True,
        )
        profile = json.loads(out.read_text(encoding="utf-8"))
        assert profile["pattern_trust"]["qwe"] == pytest.approx(0.70)
        assert profile["pattern_stats"]["qwe"] == {
            "good": 0,
            "bad": 2,
            "unclear": 0,
        }

    def test_surname_trust_learn_cli_can_restore_unclear_noop(self, tmp_path):
        audit = tmp_path / "audit.csv"
        with open(audit, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["name", "manual_label", "quality_reasons"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "name": "ab",
                    "manual_label": "unclear",
                    "quality_reasons": "very short",
                }
            )
        out = tmp_path / "trust.json"
        subprocess.run(
            [
                sys.executable, "examples/surname_trust_learn.py",
                "--input", str(audit),
                "--order", "2",
                "--output", str(out),
                "--unclear-multiplier", "1.0",
            ],
            cwd=_ROOT, check=True, text=True, capture_output=True,
        )
        profile = json.loads(out.read_text(encoding="utf-8"))
        assert profile["transition_trust"]["ab-><END>"] == pytest.approx(1.0)
        assert profile["shape_trust"]["very_short"] == pytest.approx(1.0)
