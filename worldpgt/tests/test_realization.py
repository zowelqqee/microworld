from __future__ import annotations

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.experiments.check_realization_quality import check_rows
from worldpgt.continuation.realization import (
    ENDING_AND,
    ENDING_AS,
    ENDING_BEFORE_SUBJECT,
    ENDING_INFINITIVE,
    ENDING_NEUTRAL,
    ENDING_UNTIL_SUBJECT,
    ENDING_WHEN_SUBJECT,
    classify_prompt_ending,
)


def test_classify_prompt_ending():
    assert classify_prompt_ending("She walked to the bank to") == ENDING_INFINITIVE
    assert classify_prompt_ending("She pressed the seal and") == ENDING_AND
    assert classify_prompt_ending("The boat drifted as the current") == ENDING_AS
    assert classify_prompt_ending("She waited inside the bank until the teller") == ENDING_UNTIL_SUBJECT
    assert classify_prompt_ending("The crane waited before the bird") == ENDING_BEFORE_SUBJECT
    assert classify_prompt_ending("The rock changed mood when the guitar") == ENDING_WHEN_SUBJECT
    assert classify_prompt_ending("The bat waited nearby") == ENDING_NEUTRAL


def test_prompt_ending_with_to_uses_infinitive_continuation():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The customer reached the bank teller with cash to")

    assert result.decision == "continue"
    assert result.selected_sense == "financial_institution"
    assert result.continuation.endswith("to open an account")
    assert "realization_type=infinitive_after_to" in result.memory_hits


def test_prompt_ending_with_and_uses_clause_continuation():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The wax seal marked the document and")

    assert result.decision == "continue"
    assert result.selected_sense == "closure_stamp"
    assert result.continuation.endswith("and the wax hardened")
    assert "realization_type=clause_after_and" in result.memory_hits


def test_river_bank_current_example_avoids_bad_phrase():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The boat drifted toward the bank as the current")

    assert result.decision == "continue"
    assert result.selected_sense == "river_edge"
    assert "current cast his line" not in result.continuation
    assert result.continuation.endswith("current carried it downstream")


def test_wax_seal_example_avoids_to_closed():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("She pressed the wax seal onto the envelope to")

    assert result.decision == "continue"
    assert result.selected_sense == "closure_stamp"
    assert "to closed" not in result.continuation
    assert result.continuation.endswith("to close the envelope")


def test_animal_seal_example_avoids_to_swam():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The seal swam through ocean water chasing fish to")

    assert result.decision == "continue"
    assert result.selected_sense == "animal"
    assert "to swam" not in result.continuation
    assert result.continuation.endswith("to catch another fish")


def test_realized_output_is_deterministic():
    engine = ControlledContinuationEngine()
    prompt = "The boat drifted toward the bank as the current"

    first = engine.continue_prompt(prompt)
    for _ in range(5):
        assert engine.continue_prompt(prompt) == first


def test_until_teller_does_not_repeat_subject():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("She waited inside the bank without speaking until the teller")

    assert result.decision == "continue"
    assert result.selected_sense == "financial_institution"
    assert "the teller the teller" not in result.continuation
    assert result.continuation.endswith("until the teller called her forward")


def test_until_boat_does_not_repeat_subject():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The bank looked empty from above until the boat")

    assert result.decision == "continue"
    assert result.selected_sense == "river_edge"
    assert "the boat the boat" not in result.continuation
    assert result.continuation.endswith("until the boat touched the mud")


def test_before_bird_does_not_use_and_connector():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The crane was silent before the bird")

    assert result.decision == "continue"
    assert result.selected_sense == "bird"
    assert "before the bird and" not in result.continuation
    assert result.continuation.endswith("before the bird spread its wings")


def test_when_guitar_does_not_use_and_connector():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt("The rock changed mood when the guitar")

    assert result.decision == "continue"
    assert result.selected_sense == "music"
    assert "when the guitar and" not in result.continuation
    assert result.continuation.endswith("when the guitar started playing")


def test_repeated_action_avoidance_for_wings_hit_and_open_account():
    engine = ControlledContinuationEngine()

    wings = engine.continue_prompt("The crane spread its wings above the marsh")
    hit = engine.continue_prompt("The bat cracked when he hit it during the game")
    account = engine.continue_prompt("At the bank she opened an account and asked about credit to")

    assert wings.selected_sense == "bird"
    assert "spread its wings and spread its wings" not in wings.continuation
    assert wings.continuation.endswith("and crossed the shallows")

    assert hit.selected_sense == "sports_equipment"
    assert "hit it during the game and hit the ball" not in hit.continuation
    assert hit.continuation.endswith("and the player dropped the bat")

    assert account.selected_sense == "financial_institution"
    assert "opened an account and asked about credit to open an account" not in account.continuation
    assert account.continuation.endswith("to speak with the teller")


def test_quality_checker_flags_repeated_subject_and_duplicated_action_rows():
    summary = check_rows(
        [
            {
                "id": "1",
                "prompt": "p",
                "selected_sense": "financial_institution",
                "decision": "continue",
                "continuation": "She waited until the teller the teller called her forward",
            },
            {
                "id": "2",
                "prompt": "p",
                "selected_sense": "bird",
                "decision": "continue",
                "continuation": "The crane spread its wings and spread its wings",
            },
            {
                "id": "3",
                "prompt": "p",
                "selected_sense": "music",
                "decision": "continue",
                "continuation": "The rock changed when the guitar and filled the stadium",
            },
        ]
    )

    assert summary["flagged_count"] == 3
    flags = {flag for row in summary["flagged_rows"] for flag in row["flags"]}
    assert "repeated_subject" in flags
    assert "repeated_spread_wings" in flags
    assert "connector_subject_and" in flags


def test_quality_checker_accepts_v1_3_problem_examples():
    engine = ControlledContinuationEngine()
    prompts = [
        "She waited inside the bank without speaking until the teller",
        "The bank looked empty from above until the boat",
        "The crane was silent before the bird",
        "The rock changed mood when the guitar",
        "The bat cracked when he hit it during the game",
    ]
    rows = []
    for idx, prompt in enumerate(prompts, start=1):
        result = engine.continue_prompt(prompt)
        rows.append(
            {
                "id": str(idx),
                "prompt": prompt,
                "selected_sense": result.selected_sense or "",
                "decision": result.decision,
                "continuation": result.continuation,
            }
        )

    summary = check_rows(rows)

    assert summary["flagged_count"] == 0
