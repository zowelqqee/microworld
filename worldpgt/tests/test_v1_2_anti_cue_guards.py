from __future__ import annotations

import csv
import json

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.experiments.compare_v1_1_to_v1_2 import compare
from worldpgt.experiments.run_v1_2_microworld_continuation import run
from worldpgt.experiments.run_v1_microworld_continuation import OUTPUT_FIELDS


PROMPT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
]


def test_phrase_level_anti_cue_blocks_financial_institution():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bank was not a place for money as the hikers slid down to")

    assert result.decision == "audit"
    assert result.selected_sense is None
    assert "anti_cue=not a place for money -> financial_institution" in result.memory_hits
    assert "audit_reason=anti_cue_conflict" in result.reasons


def test_player_alone_does_not_select_sports_equipment():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bat was near the player")

    assert result.decision == "audit"
    assert result.selected_sense is None
    assert "guard_failure=player_alone_insufficient -> sports_equipment" in result.memory_hits
    assert "audit_reason=guard_failure" in result.reasons


def test_sports_equipment_still_selected_with_stronger_sports_cues():
    engine = ControlledContinuationEngine()
    prompts = [
        "The baseball player picked up the bat",
        "The player took a swing with the bat",
        "The player hit the ball with the bat",
        "The bat cracked during the game",
    ]

    for prompt in prompts:
        result = engine.continue_prompt(prompt)
        assert result.decision == "continue"
        assert result.selected_sense == "sports_equipment"


def test_anti_cue_diagnostics_are_in_memory_hits_and_reasons():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bat was not an animal")

    assert "anti_cue=not an animal -> animal" in result.memory_hits
    assert "audit_reason=anti_cue" in result.reasons


def test_guard_failure_diagnostics_are_in_memory_hits_and_reasons():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The injured player pointed at the bat")

    assert "guard_failure=player_alone_insufficient -> sports_equipment" in result.memory_hits
    assert "audit_reason=guard_failure" in result.reasons


def test_old_cue_rich_cases_still_pass_under_v1_2():
    engine = ControlledContinuationEngine()

    bank = engine.continue_prompt("The customer reached the bank teller with cash to")
    bat = engine.continue_prompt("The baseball player lifted the bat before the swing")
    crane = engine.continue_prompt("The operator used the crane at the construction site to")

    assert bank.decision == "continue"
    assert bank.selected_sense == "financial_institution"
    assert bat.decision == "continue"
    assert bat.selected_sense == "sports_equipment"
    assert crane.decision == "continue"
    assert crane.selected_sense == "machine"


def test_v1_2_runner_writes_output(tmp_path):
    input_path = tmp_path / "prompts.csv"
    output_path = tmp_path / "outputs.csv"
    with open(input_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PROMPT_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "id": "1",
                "prompt": "The bank was not a place for money as the hikers slid down to",
                "ambiguous_term": "bank",
                "expected_sense": "river_edge",
                "difficulty_type": "negation",
                "notes": "phrase anti-cue",
            }
        )

    rows = run(str(input_path), str(output_path))

    assert len(rows) == 1
    with open(output_path, "r", newline="", encoding="utf-8") as handle:
        written = list(csv.DictReader(handle))
    assert len(written) == 1
    assert written[0]["decision"] == "audit"
    assert "anti_cue=not a place for money -> financial_institution" in written[0]["memory_hits"]


def test_v1_1_vs_v1_2_comparison_works(tmp_path):
    before_path = tmp_path / "before.csv"
    after_path = tmp_path / "after.csv"
    _write_output_rows(
        before_path,
        [
            _row("1", "river_edge", "financial_institution", "continue"),
            _row("2", "animal", "", "audit"),
            _row("3", "animal", "animal", "continue"),
        ],
    )
    _write_output_rows(
        after_path,
        [
            _row("1", "river_edge", "", "audit"),
            _row("2", "animal", "animal", "continue"),
            _row("3", "animal", "", "audit"),
        ],
    )

    output_path = tmp_path / "comparison.json"
    summary = compare(str(before_path), str(after_path))
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle)

    assert summary["wrong_sense_count_before"] == 1
    assert summary["wrong_sense_count_after"] == 0
    assert summary["audit_count_before"] == 1
    assert summary["audit_count_after"] == 2
    assert summary["rows_changed"] == 3
    assert summary["wrong_to_audit"] == 1
    assert summary["audit_to_correct"] == 1
    assert summary["correct_to_audit"] == 1


def test_anti_cues_recorded_in_evidence():
    memory = ExplicitSenseMemory()
    evidence = memory.score_senses_with_evidence("The bank was not a place for money", "bank")

    assert "not a place for money" in evidence.anti_cues["financial_institution"]
    assert evidence.positive_scores["financial_institution"] == 1.0
    assert evidence.adjusted_scores["financial_institution"] == 0.0


def _write_output_rows(path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _row(row_id: str, expected: str, selected: str, decision: str) -> dict:
    return {
        "id": row_id,
        "prompt": f"prompt {row_id}",
        "ambiguous_term": "bank",
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
