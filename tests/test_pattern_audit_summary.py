"""Tests for examples/pattern_audit_summary.py."""
import csv
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.pattern_audit_summary import (
    VALID_LABELS,
    USEFUL_LABELS,
    _detect_delimiter,
    normalize_label,
    read_labeled_rows,
    compute_summary,
    format_summary,
)


# ── helpers ───────────────────────────────────────────────────────────────────

_COLS = ["source", "relation_type", "target", "confidence", "reason",
         "evidence", "manual_label", "notes"]


def _write_csv(tmp_path, rows: list[dict]) -> str:
    p = tmp_path / "audit.csv"
    with open(str(p), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_COLS)
        writer.writeheader()
        writer.writerows(rows)
    return str(p)


def _row(source="a", relation_type="is_a", target="b",
         manual_label="correct", notes="") -> dict:
    return {
        "source": source, "relation_type": relation_type, "target": target,
        "confidence": "0.700000", "reason": "test", "evidence": "mid",
        "manual_label": manual_label, "notes": notes,
    }


# ── read_labeled_rows ─────────────────────────────────────────────────────────

class TestReadLabeledRows:
    def test_returns_list_of_dicts(self, tmp_path):
        path = _write_csv(tmp_path, [_row(manual_label="correct")])
        rows = read_labeled_rows(path)
        assert isinstance(rows, list)
        assert all(isinstance(r, dict) for r in rows)

    def test_skips_empty_label(self, tmp_path):
        path = _write_csv(tmp_path, [
            _row(manual_label="correct"),
            _row(manual_label=""),
            _row(manual_label="wrong"),
        ])
        rows = read_labeled_rows(path)
        assert len(rows) == 2

    def test_skips_whitespace_only_label(self, tmp_path):
        path = _write_csv(tmp_path, [_row(manual_label="   ")])
        rows = read_labeled_rows(path)
        assert rows == []

    def test_includes_all_valid_labels(self, tmp_path):
        data = [_row(manual_label=lbl) for lbl in VALID_LABELS]
        path = _write_csv(tmp_path, data)
        rows = read_labeled_rows(path)
        assert len(rows) == len(VALID_LABELS)

    def test_empty_file_returns_empty(self, tmp_path):
        path = _write_csv(tmp_path, [])
        rows = read_labeled_rows(path)
        assert rows == []

    def test_all_empty_labels_returns_empty(self, tmp_path):
        path = _write_csv(tmp_path, [_row(manual_label=""), _row(manual_label="")])
        rows = read_labeled_rows(path)
        assert rows == []


# ── compute_summary ───────────────────────────────────────────────────────────

class TestComputeSummary:
    def _rows(self, specs):
        """specs: list of (relation_type, label)"""
        return [_row(relation_type=r, manual_label=l) for r, l in specs]

    def test_total_count(self):
        rows = self._rows([("is_a", "correct")] * 5)
        s = compute_summary(rows)
        assert s["total"] == 5

    def test_empty_rows(self):
        s = compute_summary([])
        assert s["total"] == 0
        for lbl in VALID_LABELS:
            assert s["counts"].get(lbl, 0) == 0

    def test_correct_count(self):
        rows = self._rows([("is_a", "correct")] * 3 + [("is_a", "wrong")] * 2)
        s = compute_summary(rows)
        assert s["counts"]["correct"] == 3
        assert s["counts"]["wrong"] == 2

    def test_plausible_count(self):
        rows = self._rows([("r", "plausible")] * 4)
        s = compute_summary(rows)
        assert s["counts"]["plausible"] == 4

    def test_unclear_count(self):
        rows = self._rows([("r", "unclear")] * 2)
        s = compute_summary(rows)
        assert s["counts"]["unclear"] == 2

    def test_by_relation_keys(self):
        rows = self._rows([("is_a", "correct"), ("part_of", "wrong")])
        s = compute_summary(rows)
        assert "is_a" in s["by_relation"]
        assert "part_of" in s["by_relation"]

    def test_by_relation_totals(self):
        rows = self._rows([("is_a", "correct")] * 3 + [("is_a", "wrong")] * 2)
        s = compute_summary(rows)
        assert s["by_relation"]["is_a"]["total"] == 5

    def test_by_relation_correct(self):
        rows = self._rows([("is_a", "correct")] * 2 + [("is_a", "plausible")] * 1)
        s = compute_summary(rows)
        assert s["by_relation"]["is_a"]["correct"] == 2
        assert s["by_relation"]["is_a"]["plausible"] == 1

    def test_by_relation_multiple_types(self):
        rows = self._rows([
            ("is_a",   "correct"),
            ("is_a",   "correct"),
            ("part_of","wrong"),
            ("part_of","plausible"),
        ])
        s = compute_summary(rows)
        assert s["by_relation"]["is_a"]["total"] == 2
        assert s["by_relation"]["part_of"]["total"] == 2

    def test_useful_is_correct_plus_plausible(self):
        rows = self._rows([
            ("r", "correct"),
            ("r", "plausible"),
            ("r", "wrong"),
            ("r", "unclear"),
        ])
        s = compute_summary(rows)
        useful = s["counts"]["correct"] + s["counts"]["plausible"]
        assert useful == 2

    def test_useful_labels_constant(self):
        assert "correct"  in USEFUL_LABELS
        assert "plausible" in USEFUL_LABELS
        assert "wrong"    not in USEFUL_LABELS
        assert "unclear"  not in USEFUL_LABELS


# ── format_summary ────────────────────────────────────────────────────────────

class TestFormatSummary:
    def _summary(self, specs):
        rows = [_row(relation_type=r, manual_label=l) for r, l in specs]
        return compute_summary(rows)

    def test_total_in_output(self):
        out = format_summary(self._summary([("is_a", "correct")] * 7))
        assert "7" in out

    def test_correct_percentage(self):
        # 2 correct out of 4 total → 50.0%
        out = format_summary(self._summary([
            ("r", "correct"), ("r", "correct"),
            ("r", "wrong"),   ("r", "unclear"),
        ]))
        assert "50.0%" in out

    def test_useful_percentage_correct_plus_plausible(self):
        # 1 correct + 1 plausible out of 4 → 50.0%
        out = format_summary(self._summary([
            ("r", "correct"), ("r", "plausible"),
            ("r", "wrong"),   ("r", "unclear"),
        ]))
        assert "50.0%" in out

    def test_all_labels_mentioned(self):
        out = format_summary(self._summary([
            ("r", "correct"), ("r", "plausible"),
            ("r", "wrong"),   ("r", "unclear"),
        ]))
        assert "correct"  in out.lower()
        assert "plausible" in out.lower()
        assert "wrong"    in out.lower()
        assert "unclear"  in out.lower()

    def test_useful_mentioned(self):
        out = format_summary(self._summary([("r", "correct")]))
        assert "useful" in out.lower()

    def test_per_relation_table_present(self):
        out = format_summary(self._summary([
            ("is_a",   "correct"),
            ("part_of","wrong"),
        ]))
        assert "is_a"   in out
        assert "part_of" in out

    def test_per_relation_table_header(self):
        out = format_summary(self._summary([("is_a", "correct")]))
        assert "relation_type" in out
        assert "reviewed" in out

    def test_empty_summary_message(self):
        out = format_summary(compute_summary([]))
        assert "0" in out

    def test_100_percent_useful(self):
        out = format_summary(self._summary([("r", "correct")] * 5))
        assert "100.0%" in out

    def test_zero_percent_useful(self):
        out = format_summary(self._summary([("r", "wrong")] * 3))
        assert "0.0%" in out

    def test_relation_useful_percent_per_row(self):
        # is_a: 2 correct 0 wrong → 100%; part_of: 0 correct 2 wrong → 0%
        out = format_summary(self._summary([
            ("is_a",   "correct"), ("is_a",   "correct"),
            ("part_of","wrong"),   ("part_of","wrong"),
        ]))
        assert "100.0%" in out
        assert "0.0%"   in out


# ── round-trip via file ───────────────────────────────────────────────────────

class TestRoundTrip:
    def test_read_then_summarise(self, tmp_path):
        path = _write_csv(tmp_path, [
            _row(relation_type="is_a",   manual_label="correct"),
            _row(relation_type="is_a",   manual_label="plausible"),
            _row(relation_type="part_of",manual_label="wrong"),
            _row(relation_type="part_of",manual_label=""),      # skipped
        ])
        rows    = read_labeled_rows(path)
        summary = compute_summary(rows)
        assert summary["total"] == 3
        assert summary["counts"]["correct"]  == 1
        assert summary["counts"]["plausible"] == 1
        assert summary["counts"]["wrong"]    == 1
        assert summary["by_relation"]["is_a"]["total"]   == 2
        assert summary["by_relation"]["part_of"]["total"]== 1

    def test_format_round_trip_output(self, tmp_path):
        path = _write_csv(tmp_path, [
            _row(relation_type="is_a",   manual_label="correct"),
            _row(relation_type="made_of",manual_label="wrong"),
        ])
        rows    = read_labeled_rows(path)
        summary = compute_summary(rows)
        out     = format_summary(summary)
        assert "is_a"   in out
        assert "made_of" in out
        assert "Total reviewed" in out


# ── delimiter detection ───────────────────────────────────────────────────────

def _write_raw(tmp_path, content: str, filename: str = "audit.csv") -> str:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return str(p)


_SEMICOLON_CSV = (
    "source;relation_type;target;confidence;reason;evidence;manual_label;notes\n"
    "apple;is_a;fruit;0.700000;test;mid;correct;\n"
    "knife;made_of;steel;0.650000;test;mid;wrong;\n"
    "dog;capable_of;barking;0.600000;test;mid;;\n"          # empty label → skipped
)

_COMMA_CSV = (
    "source,relation_type,target,confidence,reason,evidence,manual_label,notes\n"
    "apple,is_a,fruit,0.700000,test,mid,correct,\n"
    "knife,made_of,steel,0.650000,test,mid,wrong,\n"
    "dog,capable_of,barking,0.600000,test,mid,,\n"          # empty label → skipped
)


class TestDetectDelimiter:
    def test_detects_comma(self):
        assert _detect_delimiter(_COMMA_CSV) == ","

    def test_detects_semicolon(self):
        assert _detect_delimiter(_SEMICOLON_CSV) == ";"

    def test_fallback_on_empty_string(self):
        # Sniffer will fail on an empty sample → fallback to comma
        assert _detect_delimiter("") == ","

    def test_fallback_on_single_word(self):
        # No delimiter present → Sniffer fails → comma
        assert _detect_delimiter("justoneword") == ","


class TestReadLabeledRowsDelimiter:
    def test_comma_reads_correct_rows(self, tmp_path):
        path = _write_raw(tmp_path, _COMMA_CSV)
        rows = read_labeled_rows(path)
        assert len(rows) == 2

    def test_comma_reads_label_values(self, tmp_path):
        path = _write_raw(tmp_path, _COMMA_CSV)
        rows = read_labeled_rows(path)
        labels = {r["manual_label"] for r in rows}
        assert labels == {"correct", "wrong"}

    def test_comma_reads_relation_type(self, tmp_path):
        path = _write_raw(tmp_path, _COMMA_CSV)
        rows = read_labeled_rows(path)
        rels = {r["relation_type"] for r in rows}
        assert "is_a" in rels

    def test_semicolon_reads_correct_rows(self, tmp_path):
        path = _write_raw(tmp_path, _SEMICOLON_CSV)
        rows = read_labeled_rows(path)
        assert len(rows) == 2

    def test_semicolon_reads_label_values(self, tmp_path):
        path = _write_raw(tmp_path, _SEMICOLON_CSV)
        rows = read_labeled_rows(path)
        labels = {r["manual_label"] for r in rows}
        assert labels == {"correct", "wrong"}

    def test_semicolon_reads_relation_type(self, tmp_path):
        path = _write_raw(tmp_path, _SEMICOLON_CSV)
        rows = read_labeled_rows(path)
        rels = {r["relation_type"] for r in rows}
        assert "is_a" in rels

    def test_semicolon_skips_empty_label(self, tmp_path):
        path = _write_raw(tmp_path, _SEMICOLON_CSV)
        rows = read_labeled_rows(path)
        # third row has empty label and must be excluded
        assert all(r["manual_label"].strip() for r in rows)

    def test_comma_and_semicolon_same_counts(self, tmp_path):
        p_comma = _write_raw(tmp_path, _COMMA_CSV,     "comma.csv")
        p_semi  = _write_raw(tmp_path, _SEMICOLON_CSV, "semi.csv")
        rows_comma = read_labeled_rows(p_comma)
        rows_semi  = read_labeled_rows(p_semi)
        assert len(rows_comma) == len(rows_semi)

    def test_empty_file_no_crash(self, tmp_path):
        path = _write_raw(tmp_path, "")
        rows = read_labeled_rows(path)
        assert rows == []

    def test_header_only_no_rows(self, tmp_path):
        content = "source;relation_type;target;confidence;reason;evidence;manual_label;notes\n"
        path = _write_raw(tmp_path, content)
        rows = read_labeled_rows(path)
        assert rows == []

    def test_semicolon_summary_totals_match_comma(self, tmp_path):
        p_comma = _write_raw(tmp_path, _COMMA_CSV,     "comma.csv")
        p_semi  = _write_raw(tmp_path, _SEMICOLON_CSV, "semi.csv")
        s_comma = compute_summary(read_labeled_rows(p_comma))
        s_semi  = compute_summary(read_labeled_rows(p_semi))
        assert s_comma["total"]  == s_semi["total"]
        assert s_comma["counts"] == s_semi["counts"]


# ── label normalisation ───────────────────────────────────────────────────────

class TestNormalizeLabel:
    # canonical labels pass through unchanged
    def test_correct_unchanged(self):
        assert normalize_label("correct") == "correct"

    def test_plausible_unchanged(self):
        assert normalize_label("plausible") == "plausible"

    def test_wrong_unchanged(self):
        assert normalize_label("wrong") == "wrong"

    def test_unclear_unchanged(self):
        assert normalize_label("unclear") == "unclear"

    # case-folding
    def test_uppercase_correct(self):
        assert normalize_label("Correct") == "correct"

    def test_uppercase_wrong(self):
        assert normalize_label("WRONG") == "wrong"

    def test_mixed_case_plausible(self):
        assert normalize_label("Plausible") == "plausible"

    # whitespace stripping
    def test_leading_trailing_spaces(self):
        assert normalize_label("  correct  ") == "correct"

    def test_tab_padded(self):
        assert normalize_label("\tcorrect\t") == "correct"

    # aliases → plausible
    def test_plusable_alias(self):
        assert normalize_label("plusable") == "plausible"

    def test_plausable_alias(self):
        assert normalize_label("plausable") == "plausible"

    def test_posible_alias(self):
        assert normalize_label("posible") == "plausible"

    def test_alias_case_insensitive(self):
        assert normalize_label("Plusable") == "plausible"

    # aliases → correct
    def test_true_alias(self):
        assert normalize_label("true") == "correct"

    def test_yes_alias(self):
        assert normalize_label("yes") == "correct"

    # aliases → wrong
    def test_false_alias(self):
        assert normalize_label("false") == "wrong"

    def test_no_alias(self):
        assert normalize_label("no") == "wrong"

    # unknown → unclear
    def test_unknown_label_becomes_unclear(self):
        assert normalize_label("garbage") == "unclear"

    def test_random_string_becomes_unclear(self):
        assert normalize_label("??") == "unclear"

    def test_numeric_string_becomes_unclear(self):
        assert normalize_label("1") == "unclear"


class TestComputeSummaryNormalization:
    def _rows_with_labels(self, *labels):
        return [_row(manual_label=lbl) for lbl in labels]

    def test_uppercase_label_counted(self):
        s = compute_summary(self._rows_with_labels("Correct"))
        assert s["counts"]["correct"] == 1

    def test_plusable_counted_as_plausible(self):
        s = compute_summary(self._rows_with_labels("plusable"))
        assert s["counts"]["plausible"] == 1

    def test_plausable_counted_as_plausible(self):
        s = compute_summary(self._rows_with_labels("plausable"))
        assert s["counts"]["plausible"] == 1

    def test_yes_counted_as_correct(self):
        s = compute_summary(self._rows_with_labels("yes"))
        assert s["counts"]["correct"] == 1

    def test_no_counted_as_wrong(self):
        s = compute_summary(self._rows_with_labels("no"))
        assert s["counts"]["wrong"] == 1

    def test_unknown_label_counted_as_unclear(self):
        s = compute_summary(self._rows_with_labels("junk"))
        assert s["counts"]["unclear"] == 1

    def test_unknown_label_not_dropped(self):
        s = compute_summary(self._rows_with_labels("junk"))
        assert s["total"] == 1

    def test_alias_plausible_counts_as_useful(self):
        s = compute_summary(self._rows_with_labels("plusable", "posible"))
        useful = s["counts"]["correct"] + s["counts"]["plausible"]
        assert useful == 2

    def test_mixed_variants_total(self):
        s = compute_summary(self._rows_with_labels(
            "Correct", "WRONG", "plusable", "garbage", "yes"
        ))
        assert s["total"] == 5
        assert s["counts"]["correct"]  == 2   # Correct + yes
        assert s["counts"]["wrong"]    == 1   # WRONG
        assert s["counts"]["plausible"]== 1   # plusable
        assert s["counts"]["unclear"]  == 1   # garbage
