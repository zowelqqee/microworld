from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.continuation_policy import ContinuationPolicy


def make_engine():
    return ControlledContinuationEngine()


def test_financial_bank_prompt_selects_financial_institution():
    engine = make_engine()
    result = engine.continue_prompt("He put money into his account at the bank")
    assert result.decision == "continue"
    assert result.selected_sense == "financial_institution"
    assert result.continuation == "He put money into his account at the bank and completed the transaction"


def test_river_bank_prompt_selects_river_edge():
    engine = make_engine()
    result = engine.continue_prompt("The fisherman sat by the river near the bank")
    assert result.decision == "continue"
    assert result.selected_sense == "river_edge"
    assert result.continuation.endswith("and watched the current")


def test_bat_flew_prompt_selects_animal():
    engine = make_engine()
    result = engine.continue_prompt("The bat flew out of the cave at night")
    assert result.decision == "continue"
    assert result.selected_sense == "animal"


def test_baseball_bat_prompt_selects_sports_equipment():
    engine = make_engine()
    result = engine.continue_prompt("The baseball player picked up the bat and took a swing")
    assert result.decision == "continue"
    assert result.selected_sense == "sports_equipment"


def test_low_cue_bank_prompt_goes_audit():
    engine = make_engine()
    result = engine.continue_prompt("The man went to the bank to")
    assert result.decision == "audit"
    assert result.selected_sense is None
    assert result.continuation == ""


def test_no_ambiguous_term_goes_audit():
    engine = make_engine()
    result = engine.continue_prompt("The dog ran across the yard and")
    assert result.decision == "audit"
    assert result.ambiguous_term is None
    assert result.continuation == ""


def test_result_includes_reasons():
    engine = make_engine()
    result = engine.continue_prompt("He put money into his account at the bank")
    assert result.reasons
    assert all(isinstance(reason, str) for reason in result.reasons)


def test_result_includes_memory_hits():
    engine = make_engine()
    result = engine.continue_prompt("He put money into his account at the bank")
    assert "term=bank" in result.memory_hits
    assert "cue=money -> financial_institution" in result.memory_hits
    assert "selected_sense=financial_institution" in result.memory_hits


def test_suppress_returns_empty_continuation():
    engine = ControlledContinuationEngine(policy=ContinuationPolicy(banned_patterns=["account"]))
    result = engine.continue_prompt("He put money into his account at the bank")
    assert result.decision == "suppress"
    assert result.continuation == ""


def test_deterministic_across_repeated_calls():
    engine = make_engine()
    prompt = "The fisherman sat by the river near the bank"
    first = engine.continue_prompt(prompt)
    for _ in range(5):
        again = engine.continue_prompt(prompt)
        assert again == first
