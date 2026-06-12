from __future__ import annotations

import csv
import json

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.experiments.compare_v1_to_v1_1 import compare
from worldpgt.experiments.run_v1_1_microworld_continuation import run
from worldpgt.experiments.run_v1_microworld_continuation import OUTPUT_FIELDS


PROMPT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
]


def test_negated_cue_does_not_count_as_positive_evidence():
    memory = ExplicitSenseMemory()
    evidence = memory.score_senses_with_evidence("The bank was not near the river", "bank")

    assert evidence.positive_scores["river_edge"] == 0.0
    assert "river" in evidence.negated_cues["river_edge"]
    assert "river" not in evidence.positive_cues["river_edge"]


def test_positive_cue_still_counts_normally():
    memory = ExplicitSenseMemory()
    evidence = memory.score_senses_with_evidence("The bank was near the river", "bank")

    assert evidence.positive_scores["river_edge"] == 1.0
    assert "river" in evidence.positive_cues["river_edge"]
    assert evidence.negated_cues["river_edge"] == []


def test_prompt_with_only_negated_cue_audits():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bank was not near the river")

    assert result.decision == "audit"
    assert result.selected_sense is None
    assert "negated_cue=river -> river_edge" in result.memory_hits
    assert "audit_reason=only_negated_evidence" in result.reasons


def test_negated_wrong_cue_plus_positive_correct_cue_selects_correct_or_audits():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bank was not near the river; the customer had cash")

    assert result.selected_sense in ("financial_institution", None)
    assert result.selected_sense != "river_edge"
    assert "negated_cue=river -> river_edge" in result.memory_hits


def test_bat_negation_case_does_not_select_animal():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bat was not flying; the player")

    assert result.selected_sense != "animal"
    assert result.decision in ("continue", "audit")


def test_bank_negation_case_does_not_select_river_edge():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bank was not near the river; the customer")

    assert result.selected_sense != "river_edge"
    assert result.decision in ("continue", "audit")


def test_crane_negation_case_does_not_select_bird():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The crane was not a bird; the operator")

    assert result.selected_sense != "bird"
    assert result.decision in ("continue", "audit")


def test_old_cue_rich_cases_still_pass():
    engine = ControlledContinuationEngine()

    bank = engine.continue_prompt("He put money into his account at the bank")
    bat = engine.continue_prompt("The baseball player picked up the bat and took a swing")
    crane = engine.continue_prompt("The crane at the construction site lifted a steel beam")

    assert bank.decision == "continue"
    assert bank.selected_sense == "financial_institution"
    assert bat.decision == "continue"
    assert bat.selected_sense == "sports_equipment"
    assert crane.decision == "continue"
    assert crane.selected_sense == "machine"


def test_v1_1_runner_writes_output(tmp_path):
    input_path = tmp_path / "prompts.csv"
    output_path = tmp_path / "outputs.csv"
    with open(input_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROMPT_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "prompt": "The bank was not near the river; the customer",
                "ambiguous_term": "bank",
                "expected_sense": "financial_institution",
                "difficulty_type": "negation",
                "notes": "negated wrong cue",
            }
        )

    rows = run(str(input_path), str(output_path))

    assert len(rows) == 1
    with open(output_path, "r", newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["selected_sense"] != "river_edge"
    assert "negated_cue=river -> river_edge" in written[0]["memory_hits"]


def test_v1_vs_v1_1_comparison_computes_changed_row_categories(tmp_path):
    before_path = tmp_path / "before.csv"
    after_path = tmp_path / "after.csv"
    _write_output_rows(
        before_path,
        [
            _row("1", "animal", "sports_equipment", "continue"),
            _row("2", "animal", "sports_equipment", "continue"),
            _row("3", "animal", "", "audit"),
            _row("4", "animal", "animal", "continue"),
            _row("5", "animal", "animal", "continue"),
        ],
    )
    _write_output_rows(
        after_path,
        [
            _row("1", "animal", "", "audit"),
            _row("2", "animal", "animal", "continue"),
            _row("3", "animal", "animal", "continue"),
            _row("4", "animal", "sports_equipment", "continue"),
            _row("5", "animal", "", "audit"),
        ],
    )

    output_path = tmp_path / "comparison.json"
    summary = compare(str(before_path), str(after_path))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle)

    assert summary["wrong_sense_count_before"] == 2
    assert summary["wrong_sense_count_after"] == 1
    assert summary["correct_sense_count_before"] == 2
    assert summary["correct_sense_count_after"] == 2
    assert summary["audit_count_before"] == 1
    assert summary["audit_count_after"] == 2
    assert summary["rows_changed"] == 5
    assert summary["wrong_to_audit"] == 1
    assert summary["wrong_to_correct"] == 1
    assert summary["audit_to_correct"] == 1
    assert summary["correct_to_wrong"] == 1
    assert summary["correct_to_audit"] == 1
    assert summary["changed_rows"][0]["id"] == "1"


def _write_output_rows(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(row_id: str, expected: str, selected: str, decision: str) -> dict:
    return {
        "id": row_id,
        "prompt": f"prompt {row_id}",
        "ambiguous_term": "bat",
        "expected_sense": expected,
        "difficulty_type": "negation",
        "notes": "test row",
        "continuation": "",
        "selected_sense": selected,
        "confidence": "0.5000",
        "decision": decision,
        "reasons": f"{decision} reasons",
        "memory_hits": "",
    }
