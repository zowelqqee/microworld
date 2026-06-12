from __future__ import annotations

import csv
import json
from collections import Counter

import pytest

from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.experiments.run_v1_microworld_continuation import OUTPUT_FIELDS, run
from worldpgt.experiments.summarize_v1_microworld_results import summarize


PROMPTS_PATH = "worldpgt/experiments/continuation_prompts_v1.csv"
REQUIRED_COLUMNS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
]
EXPECTED_DISTRIBUTION = {
    "cue_rich": 20,
    "weak_cue": 20,
    "delayed_cue": 20,
    "conflicting_cue": 20,
    "negation": 15,
    "misleading_surface_cue": 15,
    "no_clear_answer": 5,
    "no_known_term": 5,
}
BLANK_EXPECTED_TYPES = {"no_clear_answer", "no_known_term"}


def _read_prompt_rows() -> list[dict]:
    with open(PROMPTS_PATH, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_v1_csv_exists_has_120_rows_and_required_columns():
    with open(PROMPTS_PATH, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == REQUIRED_COLUMNS
        rows = list(reader)

    assert len(rows) == 120


def test_v1_distribution_by_difficulty_type_matches_required_counts():
    rows = _read_prompt_rows()
    assert Counter(row["difficulty_type"] for row in rows) == EXPECTED_DISTRIBUTION


def test_v1_expected_sense_blank_only_for_allowed_difficulty_types():
    rows = _read_prompt_rows()
    for row in rows:
        is_blank = row["expected_sense"] == ""
        should_be_blank = row["difficulty_type"] in BLANK_EXPECTED_TYPES
        assert is_blank == should_be_blank


def test_v1_nonblank_expected_senses_are_valid_for_ambiguous_term():
    rows = _read_prompt_rows()
    memory = ExplicitSenseMemory()
    valid_senses = {
        term: {entry.sense_id for entry in memory.get_senses(term)}
        for term in memory.known_terms()
    }

    for row in rows:
        expected = row["expected_sense"]
        if not expected:
            continue
        term = row["ambiguous_term"]
        assert term in valid_senses
        assert expected in valid_senses[term]


def test_v1_runner_writes_output_rows(tmp_path):
    input_path = tmp_path / "prompts.csv"
    output_path = tmp_path / "outputs.csv"
    with open(input_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_COLUMNS)
        writer.writeheader()
        writer.writerows(
            [
                {
                    "id": "1",
                    "prompt": "The bank teller counted cash to",
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

    rows = run(str(input_path), str(output_path))

    assert len(rows) == 2
    with open(output_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == OUTPUT_FIELDS
        written = list(reader)
    assert len(written) == 2
    assert written[0]["difficulty_type"] == "cue_rich"
    assert written[0]["selected_sense"] == "financial_institution"
    assert written[1]["decision"] == "audit"


def test_v1_summarizer_computes_global_metrics(tmp_path):
    input_path = tmp_path / "outputs.csv"
    _write_output_rows(
        input_path,
        [
            _output_row("1", "financial_institution", "financial_institution", "continue", "0.8000", "cue_rich"),
            _output_row("2", "animal", "sports_equipment", "continue", "0.6000", "conflicting_cue"),
            _output_row("3", "animal", "", "audit", "0.0000", "weak_cue"),
            _output_row("4", "", "", "audit", "0.0000", "no_clear_answer"),
            _output_row("5", "machine", "machine", "suppress", "0.7000", "negation"),
        ],
    )

    summary = summarize(str(input_path))

    assert summary["total"] == 5
    assert summary["correct_sense_count"] == 3
    assert summary["wrong_sense_count"] == 1
    assert summary["no_sense_count"] == 1
    assert summary["continue_count"] == 2
    assert summary["audit_count"] == 2
    assert summary["suppress_count"] == 1
    assert summary["correct_sense_rate"] == pytest.approx(0.6)
    assert summary["wrong_sense_rate"] == pytest.approx(0.2)
    assert summary["audit_rate"] == pytest.approx(0.4)
    assert summary["suppress_rate"] == pytest.approx(0.2)
    assert summary["avg_confidence"] == pytest.approx(0.42)


def test_v1_summarizer_computes_grouped_difficulty_metrics(tmp_path):
    input_path = tmp_path / "outputs.csv"
    _write_output_rows(
        input_path,
        [
            _output_row("1", "financial_institution", "financial_institution", "continue", "0.8000", "cue_rich"),
            _output_row("2", "river_edge", "", "audit", "0.0000", "cue_rich"),
            _output_row("3", "", "", "audit", "0.0000", "no_clear_answer"),
            _output_row("4", "", "animal", "continue", "0.5000", "no_clear_answer"),
        ],
    )

    output_path = tmp_path / "summary.json"
    summary = summarize(str(input_path))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle)

    cue_rich = summary["by_difficulty_type"]["cue_rich"]
    assert cue_rich["total"] == 2
    assert cue_rich["correct_sense_count"] == 1
    assert cue_rich["no_sense_count"] == 1
    assert cue_rich["audit_count"] == 1
    assert cue_rich["correct_sense_rate"] == pytest.approx(0.5)
    assert cue_rich["audit_rate"] == pytest.approx(0.5)
    assert cue_rich["avg_confidence"] == pytest.approx(0.4)

    no_clear = summary["by_difficulty_type"]["no_clear_answer"]
    assert no_clear["total"] == 2
    assert no_clear["correct_sense_count"] == 1
    assert no_clear["wrong_sense_count"] == 1
    assert no_clear["no_sense_count"] == 0
    assert no_clear["continue_count"] == 1
    assert no_clear["audit_count"] == 1
    assert no_clear["wrong_sense_rate"] == pytest.approx(0.5)


def _write_output_rows(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _output_row(
    row_id: str,
    expected_sense: str,
    selected_sense: str,
    decision: str,
    confidence: str,
    difficulty_type: str,
) -> dict:
    return {
        "id": row_id,
        "prompt": f"prompt {row_id}",
        "ambiguous_term": "bank",
        "expected_sense": expected_sense,
        "difficulty_type": difficulty_type,
        "notes": "test row",
        "continuation": "continuation",
        "selected_sense": selected_sense,
        "confidence": confidence,
        "decision": decision,
        "reasons": "",
        "memory_hits": "",
    }
