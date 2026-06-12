from __future__ import annotations

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.continuation.surface_validator import validate_surface_text


def test_validator_catches_repeated_subject():
    result = validate_surface_text(
        "She waited until the teller",
        "She waited until the teller the teller called her forward",
    )

    assert not result.ok
    assert "repeated_subject" in result.matched_patterns


def test_validator_catches_until_current_the_boat():
    result = validate_surface_text(
        "The bank looked empty until the current",
        "The bank looked empty until the current the boat touched the mud",
    )

    assert not result.ok
    assert "malformed_until_subject" in result.matched_patterns


def test_validator_catches_until_document_the_wax():
    result = validate_surface_text(
        "She studied the seal until the document",
        "She studied the seal until the document the wax hardened",
    )

    assert not result.ok
    assert "malformed_until_subject" in result.matched_patterns


def test_validator_catches_duplicated_spread_wings():
    result = validate_surface_text(
        "The crane spread its wings",
        "The crane spread its wings and spread its wings",
    )

    assert not result.ok
    assert "repeated_spread_wings" in result.matched_patterns


def test_validator_catches_bad_infinitive():
    result = validate_surface_text(
        "She pressed the seal to",
        "She pressed the seal to closed the envelope",
    )

    assert not result.ok
    assert "to_closed" in result.matched_patterns


def test_validator_catches_current_actor_bug():
    result = validate_surface_text(
        "The boat drifted as the current",
        "The boat drifted as the current cast his line",
    )

    assert not result.ok
    assert "current_cast_his_line" in result.matched_patterns


def test_validator_passes_clean_sentence():
    result = validate_surface_text(
        "The customer reached the bank teller with cash to",
        "The customer reached the bank teller with cash to open an account",
    )

    assert result.ok
    assert result.risk_score == 0.0
    assert result.matched_patterns == []


def test_engine_audits_surface_risky_composed_output():
    memory = ExplicitSenseMemory(include_builtin=False)
    memory.add_sense(
        "bank",
        "river_edge",
        ["river"],
        ["current cast his line"],
        {"neutral_extension": ["current cast his line"]},
    )
    engine = ControlledContinuationEngine(memory=memory)

    result = engine.continue_prompt("The river bank")

    assert result.decision == "audit"
    assert result.continuation == ""
    assert result.selected_sense == "river_edge"
    assert "audit_reason=surface_realization_risk" in result.reasons
    assert "surface_risk=current_cast_his_line" in result.memory_hits


def test_engine_clean_cue_rich_output_still_continues():
    engine = ControlledContinuationEngine()

    result = engine.continue_prompt("The customer reached the bank teller with cash to")

    assert result.decision == "continue"
    assert result.selected_sense == "financial_institution"
    assert result.continuation.endswith("to open an account")


def test_engine_surface_gate_does_not_create_wrong_continue():
    engine = ControlledContinuationEngine()
    prompts = [
        "The customer reached the bank teller with cash to",
        "The boat drifted toward the bank as the current",
        "She pressed the wax seal onto the envelope to",
    ]

    for prompt in prompts:
        result = engine.continue_prompt(prompt)
        assert result.decision == "continue"
        assert result.selected_sense is not None
