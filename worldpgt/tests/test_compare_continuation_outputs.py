"""Tests for compare_continuation_outputs comparison tooling."""

from __future__ import annotations

import csv
import json
import pytest

from worldpgt.experiments.compare_continuation_outputs import compare, _classify, _norm


# ---------------------------------------------------------------------------
# Unit tests for helpers
# ---------------------------------------------------------------------------


def test_norm_empty_string_is_none():
    assert _norm("") is None
    assert _norm("  ") is None


def test_norm_none_is_none():
    assert _norm(None) is None


def test_norm_non_empty_string_preserved():
    assert _norm("financial_institution") == "financial_institution"
    assert _norm("  river_edge  ") == "river_edge"


def test_classify_both_none_is_correct():
    assert _classify(None, None) == "correct"


def test_classify_matching_senses_is_correct():
    assert _classify("animal", "animal") == "correct"


def test_classify_expected_exists_selected_none_is_no_sense():
    assert _classify("animal", None) == "no_sense"


def test_classify_differing_senses_is_wrong():
    assert _classify("animal", "sports_equipment") == "wrong"


def test_classify_expected_none_selected_non_none_is_wrong():
    # system picked a sense when none was expected
    assert _classify(None, "financial_institution") == "wrong"


# ---------------------------------------------------------------------------
# Fixtures – write temp CSV / JSONL files
# ---------------------------------------------------------------------------


def _write_prompts(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["id", "prompt", "ambiguous_term", "expected_sense"])
        writer.writeheader()
        writer.writerows(rows)


def _write_microworld(path, rows):
    fields = ["id", "prompt", "ambiguous_term", "expected_sense", "continuation",
              "selected_sense", "confidence", "decision", "reasons", "memory_hits"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _write_fable5(path, rows):
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _mw_row(id, expected, selected, confidence="0.9000", decision="continue"):
    return {
        "id": id, "prompt": f"prompt {id}", "ambiguous_term": "bank",
        "expected_sense": expected, "continuation": "cont",
        "selected_sense": selected, "confidence": confidence,
        "decision": decision, "reasons": "", "memory_hits": "",
    }


def _f5_row(id, selected, confidence=0.9):
    return {"id": str(id), "prompt": f"prompt {id}", "ambiguous_term": "bank",
            "selected_sense": selected, "continuation": "cont",
            "reason": "reason", "confidence": confidence}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_perfect_match_both_systems(tmp_path):
    prompts = [{"id": "1", "prompt": "p1", "ambiguous_term": "bank", "expected_sense": "financial_institution"}]
    mw = [_mw_row("1", "financial_institution", "financial_institution")]
    f5 = [_f5_row("1", "financial_institution")]

    p = tmp_path / "prompts.csv"
    m = tmp_path / "mw.csv"
    f = tmp_path / "f5.jsonl"
    _write_prompts(p, prompts)
    _write_microworld(m, mw)
    _write_fable5(f, f5)

    summary = compare(str(p), str(m), str(f))

    assert summary["microworld"]["total"] == 1
    assert summary["microworld"]["correct_sense_count"] == 1
    assert summary["microworld"]["wrong_sense_count"] == 0
    assert summary["microworld"]["correct_sense_rate"] == pytest.approx(1.0)

    assert summary["fable5"]["total"] == 1
    assert summary["fable5"]["correct_sense_count"] == 1
    assert summary["fable5"]["correct_sense_rate"] == pytest.approx(1.0)

    assert summary["comparison"]["both_correct"] == 1
    assert summary["comparison"]["microworld_only_correct"] == 0
    assert summary["comparison"]["fable5_only_correct"] == 0
    assert summary["comparison"]["both_wrong"] == 0
    assert summary["comparison"]["disagreement_count"] == 0
    assert summary["comparison"]["disagreements"] == []


def test_microworld_audits_fable5_correct(tmp_path):
    """Microworld returns empty sense (audit); Fable5 picks the right sense."""
    prompts = [{"id": "1", "prompt": "p1", "ambiguous_term": "bank", "expected_sense": "financial_institution"}]
    mw = [_mw_row("1", "financial_institution", "", confidence="0.0000", decision="audit")]
    f5 = [_f5_row("1", "financial_institution", confidence=0.95)]

    p = tmp_path / "prompts.csv"
    m = tmp_path / "mw.csv"
    f = tmp_path / "f5.jsonl"
    _write_prompts(p, prompts)
    _write_microworld(m, mw)
    _write_fable5(f, f5)

    summary = compare(str(p), str(m), str(f))

    assert summary["microworld"]["audit_count"] == 1
    assert summary["microworld"]["no_sense_count"] == 1
    assert summary["microworld"]["correct_sense_count"] == 0

    assert summary["fable5"]["correct_sense_count"] == 1
    assert summary["fable5"]["wrong_sense_count"] == 0

    assert summary["comparison"]["fable5_only_correct"] == 1
    assert summary["comparison"]["microworld_only_correct"] == 0
    assert summary["comparison"]["disagreement_count"] == 1
    assert summary["comparison"]["disagreements"][0]["id"] == "1"
    assert summary["comparison"]["disagreements"][0]["microworld_decision"] == "audit"


def test_fable5_wrong_microworld_correct(tmp_path):
    """Fable5 picks the wrong sense; Microworld is correct."""
    prompts = [{"id": "1", "prompt": "p1", "ambiguous_term": "bat", "expected_sense": "animal"}]
    mw = [_mw_row("1", "animal", "animal")]
    f5 = [_f5_row("1", "sports_equipment", confidence=0.7)]

    p = tmp_path / "prompts.csv"
    m = tmp_path / "mw.csv"
    f = tmp_path / "f5.jsonl"
    _write_prompts(p, prompts)
    _write_microworld(m, mw)
    _write_fable5(f, f5)

    summary = compare(str(p), str(m), str(f))

    assert summary["microworld"]["correct_sense_count"] == 1
    assert summary["microworld"]["wrong_sense_count"] == 0

    assert summary["fable5"]["wrong_sense_count"] == 1
    assert summary["fable5"]["correct_sense_count"] == 0
    assert summary["fable5"]["correct_sense_rate"] == pytest.approx(0.0)

    assert summary["comparison"]["microworld_only_correct"] == 1
    assert summary["comparison"]["fable5_only_correct"] == 0
    assert summary["comparison"]["disagreement_count"] == 1


def test_no_known_term_row_handled_correctly(tmp_path):
    """Row with no expected sense and both systems returning no sense counts as correct."""
    prompts = [{"id": "1", "prompt": "The dog ran", "ambiguous_term": "", "expected_sense": ""}]
    mw = [_mw_row("1", "", "", confidence="0.0000", decision="audit")]
    f5 = [{"id": "1", "prompt": "The dog ran", "ambiguous_term": None,
           "selected_sense": None, "continuation": "", "reason": "no ambiguous term detected",
           "confidence": None}]

    p = tmp_path / "prompts.csv"
    m = tmp_path / "mw.csv"
    f = tmp_path / "f5.jsonl"
    _write_prompts(p, prompts)
    _write_microworld(m, mw)
    _write_fable5(f, f5)

    summary = compare(str(p), str(m), str(f))

    assert summary["microworld"]["correct_sense_count"] == 1
    assert summary["microworld"]["no_sense_count"] == 0
    assert summary["microworld"]["wrong_sense_count"] == 0

    assert summary["fable5"]["correct_sense_count"] == 1
    assert summary["fable5"]["no_sense_count"] == 0
    assert summary["fable5"]["wrong_sense_count"] == 0

    assert summary["comparison"]["both_correct"] == 1
    assert summary["comparison"]["disagreement_count"] == 0


def test_disagreement_list_populated(tmp_path):
    """Multiple rows – disagreement list captures all rows where senses differ."""
    prompts = [
        {"id": "1", "prompt": "p1", "ambiguous_term": "bank", "expected_sense": "financial_institution"},
        {"id": "2", "prompt": "p2", "ambiguous_term": "bat", "expected_sense": "animal"},
    ]
    mw = [
        _mw_row("1", "financial_institution", "financial_institution"),
        _mw_row("2", "animal", "animal"),
    ]
    f5 = [
        _f5_row("1", "financial_institution"),
        _f5_row("2", "sports_equipment", confidence=0.6),
    ]

    p = tmp_path / "prompts.csv"
    m = tmp_path / "mw.csv"
    f = tmp_path / "f5.jsonl"
    _write_prompts(p, prompts)
    _write_microworld(m, mw)
    _write_fable5(f, f5)

    summary = compare(str(p), str(m), str(f))

    assert summary["comparison"]["disagreement_count"] == 1
    d = summary["comparison"]["disagreements"][0]
    assert d["id"] == "2"
    assert d["expected_sense"] == "animal"
    assert d["microworld_selected_sense"] == "animal"
    assert d["fable5_selected_sense"] == "sports_equipment"
    assert d["fable5_confidence"] == pytest.approx(0.6)


def test_avg_confidence_computed(tmp_path):
    """avg_confidence averages only non-null confidence values."""
    prompts = [
        {"id": "1", "prompt": "p1", "ambiguous_term": "bank", "expected_sense": "financial_institution"},
        {"id": "2", "prompt": "p2", "ambiguous_term": "bat", "expected_sense": "animal"},
        {"id": "3", "prompt": "p3", "ambiguous_term": "", "expected_sense": ""},
    ]
    mw = [
        _mw_row("1", "financial_institution", "financial_institution", confidence="0.8000"),
        _mw_row("2", "animal", "animal", confidence="0.6000"),
        _mw_row("3", "", "", confidence="0.0000", decision="audit"),
    ]
    f5 = [
        _f5_row("1", "financial_institution", confidence=0.9),
        _f5_row("2", "animal", confidence=0.7),
        {"id": "3", "prompt": "p3", "ambiguous_term": None,
         "selected_sense": None, "continuation": "", "reason": "none", "confidence": None},
    ]

    p = tmp_path / "prompts.csv"
    m = tmp_path / "mw.csv"
    f = tmp_path / "f5.jsonl"
    _write_prompts(p, prompts)
    _write_microworld(m, mw)
    _write_fable5(f, f5)

    summary = compare(str(p), str(m), str(f))

    # Microworld: (0.8 + 0.6 + 0.0) / 3 = 0.4667
    assert summary["microworld"]["avg_confidence"] == pytest.approx((0.8 + 0.6 + 0.0) / 3, abs=1e-3)

    # Fable5: only rows 1 and 2 have confidence → (0.9 + 0.7) / 2 = 0.8
    assert summary["fable5"]["avg_confidence"] == pytest.approx(0.8, abs=1e-3)


def test_microworld_decision_counts(tmp_path):
    """continue/audit/suppress are counted independently."""
    prompts = [
        {"id": "1", "prompt": "p1", "ambiguous_term": "bank", "expected_sense": "financial_institution"},
        {"id": "2", "prompt": "p2", "ambiguous_term": "bank", "expected_sense": ""},
        {"id": "3", "prompt": "p3", "ambiguous_term": "bank", "expected_sense": "river_edge"},
    ]
    mw = [
        _mw_row("1", "financial_institution", "financial_institution", decision="continue"),
        _mw_row("2", "", "", decision="audit"),
        _mw_row("3", "river_edge", "river_edge", decision="suppress"),
    ]
    f5 = [
        _f5_row("1", "financial_institution"),
        {"id": "2", "prompt": "p2", "ambiguous_term": None,
         "selected_sense": None, "continuation": "", "reason": "none", "confidence": None},
        _f5_row("3", "river_edge"),
    ]

    p = tmp_path / "prompts.csv"
    m = tmp_path / "mw.csv"
    f = tmp_path / "f5.jsonl"
    _write_prompts(p, prompts)
    _write_microworld(m, mw)
    _write_fable5(f, f5)

    summary = compare(str(p), str(m), str(f))

    assert summary["microworld"]["continue_count"] == 1
    assert summary["microworld"]["audit_count"] == 1
    assert summary["microworld"]["suppress_count"] == 1
    assert summary["microworld"]["audit_rate"] == pytest.approx(1 / 3, abs=1e-3)
    assert summary["microworld"]["suppress_rate"] == pytest.approx(1 / 3, abs=1e-3)
