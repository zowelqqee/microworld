"""Tests for the deterministic audit-reason mining layer.

Covers the classifier taxonomy, summary/row consistency, determinism, the CLI
(JSON + CSV), and a current-benchmark sanity check. These are diagnostic tests:
they assert no generation output changes (the miner is read-only).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from worldpgt.experiments import audit_reason_miner as miner
from worldpgt.experiments.audit_reason_types import (
    LOW_MARGIN,
    MISSING_PHRASE_CANDIDATE,
    MISSING_SEMANTIC_FRAME,
    MISSING_SENSE_MEMORY,
    NO_SAFE_REPAIRED_CANDIDATE,
    SENSE_TIE,
    SUBJECT_ACTION_INVALID,
    SURFACE_VALIDATION_FAILED,
    TRUE_UNSAFE,
    UNKNOWN_REASON,
    UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT,
    classify,
)

_BENCHMARK = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "microworld_continuation_v1_2_outputs.csv"
)


def _row(reasons="", memory_hits="", **overrides) -> dict:
    base = {
        "id": "t-001",
        "prompt": "a prompt",
        "ambiguous_term": "bank",
        "expected_sense": "",
        "selected_sense": "",
        "confidence": "0.0000",
        "decision": "audit",
        "reasons": reasons,
        "memory_hits": memory_hits,
    }
    base.update(overrides)
    return base


# --- 1. explicit no_safe_repaired_candidate --------------------------------


def test_explicit_no_safe_repaired_candidate():
    row = _row(
        reasons="score_and_margin_above_thresholds | "
        "surface_repair_reason=attachment_drift | "
        "audit_reason=no_safe_repaired_candidate",
    )
    assert classify(row).primary_reason == NO_SAFE_REPAIRED_CANDIDATE


# --- 2. low-margin / sense-tie ---------------------------------------------


def test_low_margin_conflict_with_positive_margin_is_low_margin():
    row = _row(
        reasons="top_sense=animal | top_score=2 | second_score=1 | margin=1 | "
        "conflict_detected | audit_reason=low_margin_conflict",
    )
    assert classify(row).primary_reason == LOW_MARGIN


def test_low_margin_conflict_with_zero_margin_is_sense_tie():
    row = _row(
        reasons="top_sense=animal | top_score=1 | second_score=1 | margin=0 | "
        "conflict_detected | audit_reason=low_margin_conflict",
    )
    assert classify(row).primary_reason == SENSE_TIE


# --- 3. missing candidate / frame ------------------------------------------


def test_no_safe_semantic_candidate_with_frame_is_missing_phrase_candidate():
    row = _row(
        reasons="score_and_margin_above_thresholds | audit_reason=no_safe_semantic_candidate",
        memory_hits="semantic_frame_intent=animal_behavior | candidate_count=0",
    )
    assert classify(row).primary_reason == MISSING_PHRASE_CANDIDATE


def test_no_safe_semantic_candidate_without_frame_is_missing_frame():
    row = _row(
        reasons="score_and_margin_above_thresholds | audit_reason=no_safe_semantic_candidate",
        memory_hits="renderer=semantic_renderer_v2",
    )
    assert classify(row).primary_reason == MISSING_SEMANTIC_FRAME


def test_surface_pattern_classifies_as_surface_validation_failed():
    row = _row(
        reasons="audit_reason=surface_realization_risk | "
        "surface_pattern=malformed_until_subject | "
        "audit_reason=no_safe_semantic_candidate",
    )
    result = classify(row)
    assert result.primary_reason == SURFACE_VALIDATION_FAILED
    assert MISSING_PHRASE_CANDIDATE in result.secondary_reasons


# --- 4. unknown traces -> unknown_reason, never a fake reason ----------------


def test_insufficient_trace_is_unknown_reason():
    row = _row(reasons="some_unrecognized_marker", memory_hits="term=bank")
    assert classify(row).primary_reason == UNKNOWN_REASON


def test_empty_trace_is_unknown_reason():
    row = _row(reasons="", memory_hits="")
    assert classify(row).primary_reason == UNKNOWN_REASON


# --- other taxonomy branches (grounding) ------------------------------------


def test_guard_failure_is_weak_cue_support():
    row = _row(reasons="all_sense_scores_zero | audit_reason=guard_failure")
    assert classify(row).primary_reason == "missing_or_weak_cue_support"


def test_anti_cue_and_negation_are_true_unsafe():
    assert classify(_row(reasons="all_sense_scores_zero | audit_reason=anti_cue")).primary_reason == TRUE_UNSAFE
    assert (
        classify(_row(reasons="all_sense_scores_zero | audit_reason=only_negated_evidence")).primary_reason
        == TRUE_UNSAFE
    )


def test_zero_score_with_expected_is_weak_cue_else_unsupported():
    answerable = _row(reasons="all_sense_scores_zero", expected_sense="financial_institution")
    no_answer = _row(reasons="all_sense_scores_zero", expected_sense="")
    assert classify(answerable).primary_reason == "missing_or_weak_cue_support"
    assert classify(no_answer).primary_reason == UNSUPPORTED_OR_UNDERCONSTRAINED_CONTEXT


def test_no_ambiguous_term_is_missing_sense_memory():
    row = _row(reasons="no_ambiguous_term", ambiguous_term="")
    assert classify(row).primary_reason == MISSING_SENSE_MEMORY


def test_subject_action_invalid_marker():
    row = _row(reasons="subject_action_invalid")
    assert classify(row).primary_reason == SUBJECT_ACTION_INVALID


# --- 5. summary counts match row diagnostics --------------------------------


def test_summary_counts_match_row_diagnostics():
    report = miner.mine_file(str(_BENCHMARK))
    summary = report["summary"]
    rows = report["rows"]

    assert summary["audit_count"] == len(rows)
    assert sum(summary["reason_counts"].values()) == summary["audit_count"]

    recomputed: dict[str, int] = {}
    for diag in rows:
        recomputed[diag["primary_audit_reason"]] = (
            recomputed.get(diag["primary_audit_reason"], 0) + 1
        )
    assert recomputed == summary["reason_counts"]

    for reason, ids in summary["rows_by_reason"].items():
        assert len(ids) == summary["reason_counts"][reason]


# --- 6. determinism ---------------------------------------------------------


def test_mining_is_deterministic():
    first = miner.mine_file(str(_BENCHMARK))
    second = miner.mine_file(str(_BENCHMARK))
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


# --- 7. CLI writes JSON and CSV ---------------------------------------------


def test_cli_writes_json_and_csv(tmp_path, monkeypatch):
    out_json = tmp_path / "reasons.json"
    out_csv = tmp_path / "reasons.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "audit_reason_miner",
            "--input",
            str(_BENCHMARK),
            "--output-json",
            str(out_json),
            "--output-csv",
            str(out_csv),
        ],
    )
    miner.main()

    assert out_json.exists() and out_csv.exists()
    report = json.loads(out_json.read_text())
    assert report["summary"]["audit_count"] == 62

    with out_csv.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert len(csv_rows) == 62
    assert set(miner.CSV_FIELDS) == set(csv_rows[0].keys())


# --- 8. current-benchmark sanity + no generation change ---------------------


def test_benchmark_sanity_counts_and_v1_051():
    report = miner.mine_file(str(_BENCHMARK))
    summary = report["summary"]
    assert summary["continue_count"] == 58
    assert summary["audit_count"] == 62
    assert summary["total_rows"] == 120
    assert "v1-051" in summary["rows_by_reason"][NO_SAFE_REPAIRED_CANDIDATE]


def test_miner_does_not_modify_benchmark_output():
    before = hashlib.sha256(_BENCHMARK.read_bytes()).hexdigest()
    miner.mine_file(str(_BENCHMARK))
    after = hashlib.sha256(_BENCHMARK.read_bytes()).hexdigest()
    assert before == after
