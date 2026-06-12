import pytest

from worldpgt.continuation.audit import read_audit_rows, write_audit_rows
from worldpgt.continuation.metrics import summarize_continuation_audit
from worldpgt.continuation.types import ContinuationAuditRow


def _row(label, notes=""):
    return ContinuationAuditRow(
        prompt="p",
        continuation="c",
        ambiguous_term="bank",
        selected_sense="financial_institution",
        expected_sense="financial_institution",
        label=label,
        notes=notes,
    )


def test_rates_and_counts():
    rows = [_row("good"), _row("good"), _row("bad"), _row("unclear")]
    metrics = summarize_continuation_audit(rows)
    assert metrics["total"] == 4
    assert metrics["good"] == 2
    assert metrics["bad"] == 1
    assert metrics["unclear"] == 1
    assert metrics["good_rate"] == pytest.approx(0.5)
    assert metrics["bad_rate"] == pytest.approx(0.25)
    assert metrics["unclear_rate"] == pytest.approx(0.25)


def test_precision_good_over_good_plus_bad():
    rows = [_row("good"), _row("good"), _row("good"), _row("bad")]
    metrics = summarize_continuation_audit(rows)
    assert metrics["precision"] == pytest.approx(3 / 4)


def test_precision_zero_when_no_good_or_bad():
    metrics = summarize_continuation_audit([_row("unclear"), _row("unclear")])
    assert metrics["precision"] == 0.0


def test_empty_rows():
    metrics = summarize_continuation_audit([])
    assert metrics["total"] == 0
    assert metrics["good_rate"] == 0.0
    assert metrics["precision"] == 0.0


def test_audit_csv_roundtrip(tmp_path):
    rows = [_row("good", notes="clean match"), _row("bad")]
    path = tmp_path / "audit.csv"
    write_audit_rows(path, rows)
    loaded = read_audit_rows(path)
    assert loaded == rows
