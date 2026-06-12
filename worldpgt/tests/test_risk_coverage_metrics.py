from __future__ import annotations

import csv

import pytest

from worldpgt.experiments.policy_sweep import OUTPUT_FIELDS, run_sweep
from worldpgt.experiments.risk_coverage_metrics import summarize_rows


PROMPT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
]


def test_risk_coverage_metrics_on_synthetic_rows():
    summary = summarize_rows(
        [
            _row("financial_institution", "financial_institution", "continue", "cue_rich"),
            _row("river_edge", "financial_institution", "continue", "conflicting_cue"),
            _row("animal", "", "audit", "weak_cue"),
            _row("", "", "audit", "no_clear_answer"),
            _row("machine", "machine", "suppress", "negation"),
        ]
    )

    assert summary["total"] == 5
    assert summary["expected_answerable_count"] == 4
    assert summary["expected_no_answer_count"] == 1
    assert summary["continue_count"] == 2
    assert summary["audit_count"] == 2
    assert summary["suppress_count"] == 1
    assert summary["correct_continue_count"] == 1
    assert summary["wrong_continue_count"] == 1
    assert summary["audited_answerable_count"] == 1
    assert summary["correct_no_answer_audit_count"] == 1
    assert summary["coverage_rate"] == pytest.approx(0.4)
    assert summary["wrong_continue_rate"] == pytest.approx(0.2)
    assert summary["precision_on_continued"] == pytest.approx(0.5)


def test_precision_on_continued_zero_when_denominator_zero():
    summary = summarize_rows(
        [
            _row("animal", "", "audit", "weak_cue"),
            _row("", "", "audit", "no_clear_answer"),
        ]
    )

    assert summary["continue_count"] == 0
    assert summary["precision_on_continued"] == 0.0


def test_answerable_recall_and_abstention_rate_on_answerable():
    summary = summarize_rows(
        [
            _row("animal", "animal", "continue", "cue_rich"),
            _row("sports_equipment", "", "audit", "weak_cue"),
            _row("machine", "", "audit", "weak_cue"),
            _row("", "", "audit", "no_known_term"),
        ]
    )

    assert summary["expected_answerable_count"] == 3
    assert summary["correct_continue_count"] == 1
    assert summary["audited_answerable_count"] == 2
    assert summary["answerable_recall"] == pytest.approx(0.3333)
    assert summary["abstention_rate_on_answerable"] == pytest.approx(0.6667)


def test_correct_no_answer_audit_count():
    summary = summarize_rows(
        [
            _row("", "", "audit", "no_clear_answer"),
            _row("", "", "audit", "no_known_term"),
            _row("animal", "", "audit", "weak_cue"),
        ]
    )

    assert summary["correct_no_answer_audit_count"] == 2


def test_grouped_metrics_by_difficulty_type():
    summary = summarize_rows(
        [
            _row("animal", "animal", "continue", "cue_rich"),
            _row("sports_equipment", "", "audit", "cue_rich"),
            _row("", "", "audit", "no_clear_answer"),
        ]
    )

    cue_rich = summary["by_difficulty_type"]["cue_rich"]
    assert cue_rich["total"] == 2
    assert cue_rich["expected_answerable_count"] == 2
    assert cue_rich["correct_continue_count"] == 1
    assert cue_rich["audited_answerable_count"] == 1
    assert cue_rich["answerable_recall"] == pytest.approx(0.5)

    no_clear = summary["by_difficulty_type"]["no_clear_answer"]
    assert no_clear["total"] == 1
    assert no_clear["expected_no_answer_count"] == 1
    assert no_clear["correct_no_answer_audit_count"] == 1


def test_policy_sweep_writes_rows_and_expected_columns(tmp_path):
    input_path = tmp_path / "prompts.csv"
    output_path = tmp_path / "sweep.csv"
    with open(input_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROMPT_FIELDS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "1",
                    "prompt": "The customer reached the bank teller with cash to",
                    "ambiguous_term": "bank",
                    "expected_sense": "financial_institution",
                    "difficulty_type": "cue_rich",
                    "notes": "direct cues",
                },
                {
                    "id": "2",
                    "prompt": "The teacher opened the notebook and",
                    "ambiguous_term": "",
                    "expected_sense": "",
                    "difficulty_type": "no_known_term",
                    "notes": "no known term",
                },
            ]
        )

    rows = run_sweep(str(input_path), str(output_path))

    assert len(rows) == 8
    with open(output_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == OUTPUT_FIELDS
        written = list(reader)
    assert len(written) == 8
    assert set(written[0]) == set(OUTPUT_FIELDS)


def _row(expected_sense: str, selected_sense: str, decision: str, difficulty_type: str) -> dict:
    return {
        "expected_sense": expected_sense,
        "selected_sense": selected_sense,
        "decision": decision,
        "difficulty_type": difficulty_type,
    }
