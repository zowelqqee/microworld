from worldpgt.continuation.prompt_parser import parse_continuation_prompt
from worldpgt.continuation.sense_memory import ExplicitSenseMemory


def test_detects_bank():
    memory = ExplicitSenseMemory()
    parsed = parse_continuation_prompt("The man walked into the bank to deposit his cash", memory)
    assert parsed.ambiguous_term == "bank"
    assert set(parsed.candidate_senses) == {"financial_institution", "river_edge"}


def test_detects_bat():
    memory = ExplicitSenseMemory()
    parsed = parse_continuation_prompt("The bat flew out of the cave at night", memory)
    assert parsed.ambiguous_term == "bat"
    assert set(parsed.candidate_senses) == {"animal", "sports_equipment"}


def test_no_known_term_returns_none():
    memory = ExplicitSenseMemory()
    parsed = parse_continuation_prompt("The dog ran across the yard and", memory)
    assert parsed.ambiguous_term is None
    assert parsed.candidate_senses == []


def test_multiple_terms_chooses_strongest():
    memory = ExplicitSenseMemory()
    # "bank" appears first but has no cue support; "bat" has cave/flew/night cues.
    prompt = "Near the bank the bat flew toward the cave at night"
    parsed = parse_continuation_prompt(prompt, memory)
    assert parsed.ambiguous_term == "bat"


def test_multiple_terms_tie_prefers_first_appearance():
    memory = ExplicitSenseMemory()
    # Neither "bank" nor "bat" has any cue support: tie at 0.0, first appearance wins.
    prompt = "The bank and the bat were both mentioned"
    parsed = parse_continuation_prompt(prompt, memory)
    assert parsed.ambiguous_term == "bank"
