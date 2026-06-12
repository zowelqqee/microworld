"""Tests for examples/suppression_audit_export.py and examples/suppression_audit_summary.py."""
import csv
import json
import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pattern_prediction import PatternPrediction
from core.suppression_policy import apply_suppression_policy
from examples.suppression_audit_export import (
    COLUMNS,
    find_suppressed_rows,
    apply_calibrated_filter,
    write_audit_csv,
    build_suppressed_rows,
)
from examples.suppression_audit_summary import compute_summary, read_labeled_rows
from examples.suppression_audit_compare import (
    read_all_rows,
    compute_stats,
    is_labeled,
    format_file_report,
)

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_DATA_CSV = os.path.join(_ROOT, "data", "conceptnet_sample.csv")
_TRUST_JSON = os.path.join(_ROOT, "data", "trust_profile.json")


# ── fixtures ──────────────────────────────────────────────────────────────────

def _pred(source: str, relation: str, target: str, confidence: float) -> PatternPrediction:
    return PatternPrediction(
        source=source,
        relation_type=relation,
        target=target,
        confidence=confidence,
        reason=f"{source} via middle {target}",
        evidence=["middle"],
    )


# Two baseline predictions above threshold; learned drops them both below.
_BASELINE = [
    _pred("apple", "is_a", "plant", 0.50),
    _pred("banana", "is_a", "plant", 0.45),
    _pred("cherry", "is_a", "organism", 0.38),   # below threshold — not suppressed
]
_LEARNED = [
    _pred("apple", "is_a", "plant", 0.20),       # suppressed
    _pred("banana", "is_a", "plant", 0.18),      # suppressed
    _pred("cherry", "is_a", "organism", 0.15),   # was already below baseline threshold
]

_THRESHOLD = 0.40

# Fixtures with one noisy token (oxegen) and one clean row, for policy tests.
_BASELINE_NOISY = [
    _pred("ice", "made_of", "oxegen", 0.50),   # noisy target
    _pred("apple", "is_a", "fruit", 0.50),     # clean
]
_LEARNED_NOISY = [
    _pred("ice", "made_of", "oxegen", 0.20),
    _pred("apple", "is_a", "fruit", 0.20),
]


@pytest.fixture()
def suppressed_rows():
    return find_suppressed_rows(_BASELINE, _LEARNED, _THRESHOLD)


@pytest.fixture()
def trust_profile_json(tmp_path):
    """Trust profile with very low is_a trust to guarantee suppression on tiny CSV."""
    profile = {
        "counts": {},
        "drift_trust": {},
        "evidence_trust": {},
        "relation_trust": {"is_a": 0.10, "made_of": 0.10, "part_of": 0.10},
        "rule_trust": {},
    }
    p = tmp_path / "trust_profile.json"
    p.write_text(json.dumps(profile, indent=2))
    return str(p)


_TINY_CSV = """\
source,relation_type,target
apple,is_a,fruit
banana,is_a,fruit
cherry,is_a,fruit
mango,is_a,fruit
peach,is_a,fruit
fruit,is_a,plant
plant,is_a,organism
knife,made_of,steel
fork,made_of,steel
spoon,made_of,steel
blade,made_of,steel
steel,made_of,iron
wheel,part_of,car
engine,part_of,car
door,part_of,car
window,part_of,car
car,part_of,vehicle
"""


@pytest.fixture()
def tiny_csv(tmp_path):
    p = tmp_path / "tiny.csv"
    p.write_text(_TINY_CSV)
    return str(p)


# ── find_suppressed_rows: row invariants ──────────────────────────────────────


class TestFindSuppressedRows:
    def test_returns_only_suppressed(self, suppressed_rows):
        assert len(suppressed_rows) == 2

    def test_baseline_accepted(self, suppressed_rows):
        for row in suppressed_rows:
            assert float(row["baseline_confidence"]) >= _THRESHOLD

    def test_learned_suppressed(self, suppressed_rows):
        for row in suppressed_rows:
            assert float(row["learned_confidence"]) < _THRESHOLD

    def test_delta_is_negative(self, suppressed_rows):
        for row in suppressed_rows:
            assert float(row["delta"]) < 0.0

    def test_delta_equals_learned_minus_baseline(self, suppressed_rows):
        for row in suppressed_rows:
            expected = float(row["learned_confidence"]) - float(row["baseline_confidence"])
            assert float(row["delta"]) == pytest.approx(expected, abs=1e-9)

    def test_all_columns_present(self, suppressed_rows):
        for row in suppressed_rows:
            assert set(row.keys()) == set(COLUMNS)

    def test_manual_label_empty(self, suppressed_rows):
        for row in suppressed_rows:
            assert row["manual_label"] == ""

    def test_notes_empty(self, suppressed_rows):
        for row in suppressed_rows:
            assert row["notes"] == ""

    def test_sorted_by_delta_ascending(self, suppressed_rows):
        deltas = [float(r["delta"]) for r in suppressed_rows]
        assert deltas == sorted(deltas)

    def test_evidence_is_pipe_string(self, suppressed_rows):
        for row in suppressed_rows:
            assert isinstance(row["evidence"], str)
            assert "[" not in row["evidence"]

    def test_empty_predictions_returns_empty(self):
        assert find_suppressed_rows([], [], _THRESHOLD) == []

    def test_no_suppression_returns_empty(self):
        # learned confidence equals baseline — nothing suppressed
        baseline = [_pred("a", "is_a", "b", 0.50)]
        learned = [_pred("a", "is_a", "b", 0.50)]
        assert find_suppressed_rows(baseline, learned, _THRESHOLD) == []

    def test_prediction_below_threshold_not_included(self):
        # baseline below threshold → should not appear even if learned is also below
        baseline = [_pred("a", "is_a", "b", 0.30)]
        learned = [_pred("a", "is_a", "b", 0.10)]
        assert find_suppressed_rows(baseline, learned, _THRESHOLD) == []

    def test_triple_missing_from_learned_skipped(self):
        baseline = [_pred("a", "is_a", "b", 0.50)]
        assert find_suppressed_rows(baseline, [], _THRESHOLD) == []


# ── write_audit_csv ───────────────────────────────────────────────────────────


class TestWriteAuditCSV:
    def test_creates_file(self, suppressed_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(suppressed_rows, out)
        assert os.path.exists(out)

    def test_returns_row_count(self, suppressed_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        n = write_audit_csv(suppressed_rows, out)
        assert n == len(suppressed_rows)

    def test_header_matches_columns(self, suppressed_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(suppressed_rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            assert list(csv.DictReader(f).fieldnames) == COLUMNS

    def test_empty_rows_writes_header_only(self, tmp_path):
        out = str(tmp_path / "empty.csv")
        n = write_audit_csv([], out)
        assert n == 0
        with open(out, newline="", encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        assert len(lines) == 1
        assert lines[0] == ",".join(COLUMNS)

    def test_creates_parent_dir(self, suppressed_rows, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "audit.csv")
        write_audit_csv(suppressed_rows, out)
        assert os.path.exists(out)


# ── build_suppressed_rows: missing trust profile ──────────────────────────────


class TestMissingTrustProfile:
    def test_raises_file_not_found(self, tiny_csv):
        with pytest.raises(FileNotFoundError) as exc_info:
            build_suppressed_rows(tiny_csv, "/nonexistent/trust_profile.json")
        assert "Trust profile not found" in str(exc_info.value)
        assert "/nonexistent/trust_profile.json" in str(exc_info.value)


# ── build_suppressed_rows: end-to-end with real data ─────────────────────────


@pytest.mark.skipif(
    not os.path.exists(_DATA_CSV) or not os.path.exists(_TRUST_JSON),
    reason="real data files not present",
)
class TestBuildSuppressedRowsRealData:
    def test_produces_suppressed_rows(self):
        rows = build_suppressed_rows(_DATA_CSV, _TRUST_JSON, threshold=0.40)
        assert len(rows) > 0

    def test_all_baseline_accepted(self):
        rows = build_suppressed_rows(_DATA_CSV, _TRUST_JSON, threshold=0.40)
        for row in rows:
            assert float(row["baseline_confidence"]) >= 0.40

    def test_all_learned_suppressed(self):
        rows = build_suppressed_rows(_DATA_CSV, _TRUST_JSON, threshold=0.40)
        for row in rows:
            assert float(row["learned_confidence"]) < 0.40

    def test_all_deltas_negative(self):
        rows = build_suppressed_rows(_DATA_CSV, _TRUST_JSON, threshold=0.40)
        for row in rows:
            assert float(row["delta"]) < 0.0

    def test_limit_respected(self):
        rows = build_suppressed_rows(_DATA_CSV, _TRUST_JSON, threshold=0.40, limit=10)
        assert len(rows) <= 10

    def test_columns_correct(self):
        rows = build_suppressed_rows(_DATA_CSV, _TRUST_JSON, threshold=0.40, limit=5)
        for row in rows:
            assert set(row.keys()) == set(COLUMNS)


# ── suppression_audit_summary: compute_summary ────────────────────────────────


class TestComputeSummary:
    def _rows(self, labels: list[str]) -> list[dict]:
        return [{"manual_label": lbl} for lbl in labels]

    def test_precision_all_suppress(self):
        summary = compute_summary(self._rows(["should_suppress", "should_suppress"]))
        assert summary["suppression_precision"] == pytest.approx(1.0)

    def test_precision_all_keep(self):
        summary = compute_summary(self._rows(["should_keep", "should_keep"]))
        assert summary["suppression_precision"] == pytest.approx(0.0)

    def test_precision_mixed(self):
        rows = self._rows(["should_suppress", "should_suppress", "should_keep"])
        summary = compute_summary(rows)
        assert summary["suppression_precision"] == pytest.approx(2 / 3)

    def test_precision_zero_when_no_decisives(self):
        summary = compute_summary(self._rows(["unclear", "unclear"]))
        assert summary["suppression_precision"] == pytest.approx(0.0)

    def test_total_count(self):
        rows = self._rows(["should_suppress", "should_keep", "unclear"])
        assert compute_summary(rows)["total"] == 3

    def test_unknown_label_counted_as_unclear(self):
        summary = compute_summary(self._rows(["garbage_label"]))
        assert summary["counts"]["unclear"] == 1

    def test_empty_rows(self):
        summary = compute_summary([])
        assert summary["total"] == 0
        assert summary["suppression_precision"] == pytest.approx(0.0)

    def test_counts_keys_present(self):
        summary = compute_summary(self._rows(["should_suppress"]))
        assert "should_suppress" in summary["counts"]
        assert "should_keep" in summary["counts"]
        assert "unclear" in summary["counts"]


# ── suppression_audit_summary: read_labeled_rows ─────────────────────────────


class TestReadLabeledRows:
    def test_skips_empty_label(self, tmp_path):
        p = tmp_path / "audit.csv"
        p.write_text(
            "source,relation_type,target,baseline_confidence,learned_confidence,"
            "delta,evidence,reason,manual_label,notes\n"
            "a,is_a,b,0.50,0.20,-0.30,,reason,should_suppress,\n"
            "c,is_a,d,0.45,0.15,-0.30,,reason,,\n"
        )
        rows = read_labeled_rows(str(p))
        assert len(rows) == 1
        assert rows[0]["manual_label"] == "should_suppress"


# ── integration: script runs end-to-end ──────────────────────────────────────


@pytest.mark.skipif(
    not os.path.exists(_DATA_CSV) or not os.path.exists(_TRUST_JSON),
    reason="real data files not present",
)
class TestScriptIntegration:
    def test_export_script_runs(self, tmp_path):
        out = str(tmp_path / "suppression_audit.csv")
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_export.py",
             "--limit", "20", "--output", out],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert os.path.exists(out)
        assert "Wrote" in result.stderr

    def test_export_script_missing_trust_profile(self, tmp_path):
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_export.py",
             "--trust-profile", "/nonexistent/trust.json"],
            cwd=_ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0
        assert "Trust profile not found" in result.stderr

    def test_summary_script_runs_on_labeled_csv(self, tmp_path):
        labeled = tmp_path / "labeled.csv"
        labeled.write_text(
            "source,relation_type,target,baseline_confidence,learned_confidence,"
            "delta,evidence,reason,manual_label,notes\n"
            "a,is_a,b,0.50,0.20,-0.30,,r,should_suppress,\n"
            "c,is_a,d,0.45,0.15,-0.30,,r,should_keep,\n"
        )
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_summary.py",
             "--input", str(labeled)],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "suppression_precision" in result.stdout
        assert "Total reviewed" in result.stdout

    def test_calibrated_export_script_runs(self, tmp_path):
        out = str(tmp_path / "calibrated.csv")
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_export.py",
             "--limit", "20",
             "--min-negative-delta", "0.01",
             "--output", out],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert os.path.exists(out)
        assert "Wrote" in result.stderr


# ── apply_calibrated_filter ───────────────────────────────────────────────────

# apple: baseline=0.50, learned=0.20, delta=-0.30
# banana: baseline=0.45, learned=0.18, delta=-0.27


class TestApplyCalibratedFilter:
    def test_default_args_returns_all_rows(self, suppressed_rows):
        filtered = apply_calibrated_filter(suppressed_rows)
        assert len(filtered) == len(suppressed_rows)

    def test_zero_min_negative_delta_returns_all_rows(self, suppressed_rows):
        filtered = apply_calibrated_filter(suppressed_rows, min_negative_delta=0.0)
        assert len(filtered) == len(suppressed_rows)

    def test_min_negative_delta_excludes_shallow_drops(self, suppressed_rows):
        # delta must be <= -0.28; banana delta=-0.27 is excluded
        filtered = apply_calibrated_filter(suppressed_rows, min_negative_delta=0.28)
        assert len(filtered) == 1
        assert filtered[0]["source"] == "apple"

    def test_min_negative_delta_keeps_exact_match(self, suppressed_rows):
        # apple delta=-0.30; threshold=-0.30 → kept (<=)
        filtered = apply_calibrated_filter(suppressed_rows, min_negative_delta=0.30)
        assert any(r["source"] == "apple" for r in filtered)

    def test_min_negative_delta_excludes_all(self, suppressed_rows):
        filtered = apply_calibrated_filter(suppressed_rows, min_negative_delta=0.99)
        assert filtered == []

    def test_max_learned_confidence_excludes_high_conf(self, suppressed_rows):
        # apple learned=0.20 > 0.19 → excluded; banana learned=0.18 → kept
        filtered = apply_calibrated_filter(suppressed_rows, max_learned_confidence=0.19)
        assert len(filtered) == 1
        assert filtered[0]["source"] == "banana"

    def test_max_learned_confidence_keeps_all_when_generous(self, suppressed_rows):
        filtered = apply_calibrated_filter(suppressed_rows, max_learned_confidence=1.0)
        assert len(filtered) == len(suppressed_rows)

    def test_combined_both_filters(self, suppressed_rows):
        # min_negative_delta=0.25: both pass (-0.30 <= -0.25, -0.27 <= -0.25)
        # max_learned_confidence=0.19: apple (0.20) excluded; banana (0.18) kept
        filtered = apply_calibrated_filter(
            suppressed_rows, min_negative_delta=0.25, max_learned_confidence=0.19
        )
        assert len(filtered) == 1
        assert filtered[0]["source"] == "banana"

    def test_output_columns_unchanged(self, suppressed_rows):
        filtered = apply_calibrated_filter(suppressed_rows, min_negative_delta=0.25)
        for row in filtered:
            assert set(row.keys()) == set(COLUMNS)

    def test_all_deltas_still_negative(self, suppressed_rows):
        filtered = apply_calibrated_filter(suppressed_rows, min_negative_delta=0.10)
        for row in filtered:
            assert float(row["delta"]) < 0.0

    def test_empty_input_returns_empty(self):
        assert apply_calibrated_filter([]) == []


# ── suppression_audit_compare ─────────────────────────────────────────────────

_LABELED_CSV = """\
source,relation_type,target,baseline_confidence,learned_confidence,delta,evidence,reason,manual_label,notes
a,is_a,b,0.50,0.20,-0.30,,r,should_suppress,
c,is_a,d,0.45,0.15,-0.30,,r,should_keep,
e,is_a,f,0.42,0.10,-0.32,,r,should_suppress,
"""

_UNLABELED_CSV = """\
source,relation_type,target,baseline_confidence,learned_confidence,delta,evidence,reason,manual_label,notes
a,is_a,b,0.50,0.20,-0.30,,r,,
c,part_of,d,0.45,0.15,-0.30,,r,,
"""


@pytest.fixture()
def labeled_csv(tmp_path):
    p = tmp_path / "labeled.csv"
    p.write_text(_LABELED_CSV)
    return str(p)


@pytest.fixture()
def unlabeled_csv(tmp_path):
    p = tmp_path / "unlabeled.csv"
    p.write_text(_UNLABELED_CSV)
    return str(p)


class TestCompareHelper:
    # ── read_all_rows ──────────────────────────────────────────────────────────

    def test_read_all_rows_returns_all(self, labeled_csv):
        rows = read_all_rows(labeled_csv)
        assert len(rows) == 3

    def test_read_all_rows_includes_unlabeled(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        assert len(rows) == 2

    # ── is_labeled ────────────────────────────────────────────────────────────

    def test_is_labeled_true_when_any_label(self, labeled_csv):
        rows = read_all_rows(labeled_csv)
        assert is_labeled(rows) is True

    def test_is_labeled_false_when_all_empty(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        assert is_labeled(rows) is False

    # ── compute_stats ─────────────────────────────────────────────────────────

    def test_compute_stats_total(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        stats = compute_stats(rows)
        assert stats["total"] == 2

    def test_compute_stats_avg_delta(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        stats = compute_stats(rows)
        assert stats["avg_delta"] == pytest.approx(-0.30, abs=1e-9)

    def test_compute_stats_relation_dist(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        stats = compute_stats(rows)
        assert stats["relation_type_distribution"] == {"is_a": 1, "part_of": 1}

    def test_compute_stats_empty(self):
        stats = compute_stats([])
        assert stats == {"total": 0}

    # ── format_file_report: labeled ───────────────────────────────────────────

    def test_labeled_report_contains_precision(self, labeled_csv):
        rows = read_all_rows(labeled_csv)
        report = format_file_report("test", rows)
        assert "suppression_precision" in report

    def test_labeled_report_precision_value(self, labeled_csv):
        rows = read_all_rows(labeled_csv)
        report = format_file_report("test", rows)
        # 2 suppress, 1 keep → precision = 2/3 ≈ 0.667
        assert "0.667" in report

    def test_labeled_report_counts(self, labeled_csv):
        rows = read_all_rows(labeled_csv)
        report = format_file_report("test", rows)
        assert "should_suppress" in report
        assert "should_keep" in report

    # ── format_file_report: unlabeled ─────────────────────────────────────────

    def test_unlabeled_report_shows_total_rows(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        report = format_file_report("test", rows)
        assert "Total rows" in report

    def test_unlabeled_report_shows_avg_delta(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        report = format_file_report("test", rows)
        assert "avg delta" in report

    def test_unlabeled_report_shows_relation_dist(self, unlabeled_csv):
        rows = read_all_rows(unlabeled_csv)
        report = format_file_report("test", rows)
        assert "relation_type dist" in report

    def test_unlabeled_report_no_crash_on_empty(self):
        report = format_file_report("empty", [])
        assert "0" in report

    # ── compare script integration ────────────────────────────────────────────

    def test_compare_script_labeled_vs_labeled(self, labeled_csv, tmp_path):
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_compare.py",
             "--old", labeled_csv, "--new", labeled_csv],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert "suppression_precision" in result.stdout

    def test_compare_script_unlabeled_no_crash(self, unlabeled_csv):
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_compare.py",
             "--old", unlabeled_csv, "--new", unlabeled_csv],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0

    def test_compare_script_missing_file_exits_nonzero(self, labeled_csv, tmp_path):
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_compare.py",
             "--old", labeled_csv, "--new", "/nonexistent/file.csv"],
            cwd=_ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode != 0


# ── suppression policy integration ───────────────────────────────────────────


class TestPolicyInExport:
    def test_default_policy_is_naive_unchanged(self):
        rows = find_suppressed_rows(_BASELINE, _LEARNED, _THRESHOLD)
        assert apply_suppression_policy(rows, "naive") == rows

    def test_quality_aware_reduces_rows_with_noisy_token(self):
        # naive gives 2 rows; quality_aware keeps only ice/oxegen (noisy target)
        rows = find_suppressed_rows(_BASELINE_NOISY, _LEARNED_NOISY, _THRESHOLD)
        naive_rows = apply_suppression_policy(rows, "naive")
        qa_rows = apply_suppression_policy(rows, "quality_aware")
        assert len(qa_rows) < len(naive_rows)
        assert len(qa_rows) == 1
        assert qa_rows[0]["target"] == "oxegen"

    def test_quality_aware_output_columns_unchanged(self):
        rows = find_suppressed_rows(_BASELINE_NOISY, _LEARNED_NOISY, _THRESHOLD)
        qa_rows = apply_suppression_policy(rows, "quality_aware")
        for row in qa_rows:
            assert set(row.keys()) == set(COLUMNS)

    def test_naive_policy_preserves_all_suppressed_rows(self):
        rows = find_suppressed_rows(_BASELINE_NOISY, _LEARNED_NOISY, _THRESHOLD)
        assert len(apply_suppression_policy(rows, "naive")) == 2


@pytest.mark.skipif(
    not os.path.exists(_DATA_CSV) or not os.path.exists(_TRUST_JSON),
    reason="real data files not present",
)
class TestPolicyScriptIntegration:
    def test_export_script_policy_quality_aware(self, tmp_path):
        out = str(tmp_path / "quality_aware.csv")
        result = subprocess.run(
            [sys.executable, "examples/suppression_audit_export.py",
             "--policy", "quality_aware",
             "--output", out],
            cwd=_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        assert os.path.exists(out)
        assert "Wrote" in result.stderr

    def test_export_script_policy_naive_is_default(self, tmp_path):
        out_default = str(tmp_path / "default.csv")
        out_naive = str(tmp_path / "naive.csv")
        for out, extra_args in [
            (out_default, []),
            (out_naive, ["--policy", "naive"]),
        ]:
            subprocess.run(
                [sys.executable, "examples/suppression_audit_export.py",
                 "--limit", "10", "--output", out] + extra_args,
                cwd=_ROOT, check=True, capture_output=True,
            )
        with open(out_default) as f1, open(out_naive) as f2:
            assert f1.read() == f2.read()
