"""Tests for examples/pattern_audit_export.py."""
import csv
import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.pattern_audit_export import COLUMNS, build_audit_rows, write_audit_csv

# ── tiny fixture shared across all tests ──────────────────────────────────────

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


@pytest.fixture()
def audit_rows(tiny_csv):
    return build_audit_rows(tiny_csv, limit=200)


# ── build_audit_rows ───────────────────────────────────────────────────────────


class TestBuildAuditRows:
    def test_returns_list_of_dicts(self, audit_rows):
        assert isinstance(audit_rows, list)
        assert all(isinstance(r, dict) for r in audit_rows)

    def test_all_columns_present(self, audit_rows):
        for row in audit_rows:
            for col in COLUMNS:
                assert col in row, f"Missing column '{col}' in row {row}"

    def test_no_extra_columns(self, audit_rows):
        for row in audit_rows:
            assert set(row.keys()) == set(COLUMNS)

    def test_manual_label_always_empty(self, audit_rows):
        for row in audit_rows:
            assert row["manual_label"] == ""

    def test_notes_always_empty(self, audit_rows):
        for row in audit_rows:
            assert row["notes"] == ""

    def test_confidence_is_numeric_string(self, audit_rows):
        for row in audit_rows:
            val = float(row["confidence"])
            assert 0.0 <= val <= 1.0

    def test_relation_type_is_string(self, audit_rows):
        for row in audit_rows:
            assert isinstance(row["relation_type"], str)
            assert len(row["relation_type"]) > 0

    def test_source_and_target_non_empty(self, audit_rows):
        for row in audit_rows:
            assert row["source"]
            assert row["target"]

    def test_produces_at_least_one_row(self, audit_rows):
        assert len(audit_rows) >= 1

    def test_sorted_by_confidence_desc(self, audit_rows):
        confs = [float(r["confidence"]) for r in audit_rows]
        assert confs == sorted(confs, reverse=True)

    def test_secondary_sort_by_relation_type(self, audit_rows):
        # within the same confidence tier the rows must be sorted by relation_type
        prev_conf = None
        group: list[str] = []
        for row in audit_rows:
            conf = float(row["confidence"])
            if conf != prev_conf:
                # check previous group
                assert group == sorted(group), (
                    f"relation_types not sorted within confidence tier {prev_conf}: {group}"
                )
                group = []
                prev_conf = conf
            group.append(row["relation_type"])
        # check last group
        assert group == sorted(group)

    def test_tertiary_sort_by_source(self, audit_rows):
        # within the same (confidence, relation_type) tier, sorted by source
        prev_key = None
        group: list[str] = []
        for row in audit_rows:
            key = (float(row["confidence"]), row["relation_type"])
            if key != prev_key:
                assert group == sorted(group), (
                    f"sources not sorted within key {prev_key}: {group}"
                )
                group = []
                prev_key = key
            group.append(row["source"])
        assert group == sorted(group)

    def test_limit_respected(self, tiny_csv):
        rows = build_audit_rows(tiny_csv, limit=3)
        assert len(rows) == 3

    def test_limit_zero_returns_empty(self, tiny_csv):
        rows = build_audit_rows(tiny_csv, limit=0)
        assert rows == []

    def test_limit_larger_than_predictions_returns_all(self, tiny_csv):
        rows_10 = build_audit_rows(tiny_csv, limit=10)
        rows_huge = build_audit_rows(tiny_csv, limit=10_000)
        assert len(rows_10) <= len(rows_huge)

    def test_relation_filter_single(self, tiny_csv):
        rows = build_audit_rows(tiny_csv, relation_filter={"is_a"}, limit=200)
        assert len(rows) >= 1
        for r in rows:
            assert r["relation_type"] == "is_a"

    def test_relation_filter_multiple(self, tiny_csv):
        rows = build_audit_rows(tiny_csv, relation_filter={"is_a", "made_of"}, limit=200)
        assert len(rows) >= 1
        for r in rows:
            assert r["relation_type"] in {"is_a", "made_of"}

    def test_relation_filter_unknown_returns_empty(self, tiny_csv):
        rows = build_audit_rows(tiny_csv, relation_filter={"no_such_relation"}, limit=200)
        assert rows == []

    def test_relation_filter_none_returns_all_types(self, tiny_csv):
        rows_no_filter = build_audit_rows(tiny_csv, limit=200)
        types_no_filter = {r["relation_type"] for r in rows_no_filter}
        rows_filtered   = build_audit_rows(tiny_csv, relation_filter={"is_a"}, limit=200)
        types_filtered  = {r["relation_type"] for r in rows_filtered}
        # unfiltered should include at least every type that the filtered run has
        assert types_filtered <= types_no_filter

    def test_evidence_is_pipe_separated_string(self, audit_rows):
        for row in audit_rows:
            # must be a plain string (may be empty, or "a|b|c" etc.)
            assert isinstance(row["evidence"], str)

    def test_evidence_no_list_objects(self, audit_rows):
        for row in audit_rows:
            assert "[" not in row["evidence"], (
                f"evidence looks like a repr'd list: {row['evidence']}"
            )

    def test_no_self_loops(self, audit_rows):
        for row in audit_rows:
            assert row["source"] != row["target"]


# ── write_audit_csv ───────────────────────────────────────────────────────────


class TestWriteAuditCSV:
    def test_creates_file(self, audit_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(audit_rows, out)
        assert os.path.exists(out)

    def test_returns_row_count(self, audit_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        n = write_audit_csv(audit_rows, out)
        assert n == len(audit_rows)

    def test_header_matches_columns(self, audit_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(audit_rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert list(reader.fieldnames) == COLUMNS

    def test_data_rows_match(self, audit_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(audit_rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            written = list(reader)
        assert len(written) == len(audit_rows)
        for orig, read in zip(audit_rows, written):
            for col in COLUMNS:
                assert orig[col] == read[col]

    def test_manual_label_empty_in_file(self, audit_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(audit_rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["manual_label"] == ""

    def test_notes_empty_in_file(self, audit_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(audit_rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["notes"] == ""

    def test_creates_parent_dir_if_missing(self, audit_rows, tmp_path):
        out = str(tmp_path / "nested" / "deep" / "audit.csv")
        write_audit_csv(audit_rows, out)
        assert os.path.exists(out)

    def test_empty_rows_writes_only_header(self, tmp_path):
        out = str(tmp_path / "empty.csv")
        n = write_audit_csv([], out)
        assert n == 0
        with open(out, newline="", encoding="utf-8") as f:
            content = f.read()
        lines = content.strip().splitlines()
        assert len(lines) == 1  # header only
        assert lines[0] == ",".join(COLUMNS)

    def test_overwrite_existing_file(self, audit_rows, tmp_path):
        out = str(tmp_path / "out.csv")
        write_audit_csv(audit_rows, out)
        # write empty to overwrite
        write_audit_csv([], out)
        with open(out, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert rows == []


# ── integration: build → write → read back ────────────────────────────────────


class TestRoundTrip:
    def test_roundtrip_limit(self, tiny_csv, tmp_path):
        out = str(tmp_path / "audit.csv")
        rows = build_audit_rows(tiny_csv, limit=5)
        write_audit_csv(rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            read_back = list(csv.DictReader(f))
        assert len(read_back) == len(rows)

    def test_roundtrip_relation_filter(self, tiny_csv, tmp_path):
        out = str(tmp_path / "audit_is_a.csv")
        rows = build_audit_rows(tiny_csv, relation_filter={"is_a"}, limit=50)
        write_audit_csv(rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                assert row["relation_type"] == "is_a"

    def test_roundtrip_all_columns_readable(self, tiny_csv, tmp_path):
        out = str(tmp_path / "audit_all.csv")
        rows = build_audit_rows(tiny_csv, limit=20)
        write_audit_csv(rows, out)
        with open(out, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert set(row.keys()) == set(COLUMNS)
