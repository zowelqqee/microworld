from __future__ import annotations

from worldpgt.continuation.semantic_frame import build_semantic_frame


def test_financial_bank_prompt_builds_transaction_frame():
    frame = build_semantic_frame(
        "The customer reached the bank teller with cash to", "bank", "financial_institution", []
    )
    assert frame.intent == "transaction"
    assert frame.term == "bank"
    assert frame.sense_id == "financial_institution"
    assert frame.actor == "customer"
    assert frame.connector_type == "infinitive_after_to"


def test_river_bank_prompt_builds_river_edge_activity_frame():
    frame = build_semantic_frame(
        "The fisherman sat on the muddy bank by the river to", "bank", "river_edge", []
    )
    assert frame.intent == "river_edge_activity"
    assert frame.actor == "fisherman"
    assert frame.location == "river"


def test_sports_bat_prompt_builds_sports_action_frame():
    frame = build_semantic_frame(
        "The baseball player lifted the bat before the swing", "bat", "sports_equipment", []
    )
    assert frame.intent == "sports_action"
    assert frame.actor == "player"
    assert frame.connector_type == "clause_after_before_subject"


def test_animal_bat_prompt_builds_animal_behavior_frame():
    frame = build_semantic_frame(
        "The bat flew from the cave at night with its wings", "bat", "animal", []
    )
    assert frame.intent == "animal_behavior"
    assert frame.term == "bat"
    # "bat" is the ambiguous term, never reported as the actor.
    assert frame.actor != "bat"


def test_evidence_is_carried_through():
    frame = build_semantic_frame("The seal swam to", "seal", "animal", ["positive_cue=ocean -> animal"])
    assert frame.evidence == ["positive_cue=ocean -> animal"]
