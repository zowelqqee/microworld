"""Tests for the deterministic surface-repair layer.

Covers connector-grammar repair, role-aware subject/action validation, repeated-
object / prey repair, attachment-drift routing to audit, and determinism. These
exercise both the unit modules and the end-to-end engine integration.
"""

from __future__ import annotations

from worldpgt.continuation.connector_rewriter import repair_connector_grammar
from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.coreference_repair import repair_repetition
from worldpgt.continuation.realization import ENDING_BEFORE_SUBJECT
from worldpgt.continuation.semantic_frame import build_semantic_frame
from worldpgt.continuation.subject_action_validator import (
    safe_body_part_clause,
    validate_subject_action,
)
from worldpgt.continuation.surface_repair import (
    AUDIT_REASON,
    repair_surface_candidate,
)


def _frame(prompt: str, term: str, sense_id: str):
    return build_semantic_frame(prompt, term, sense_id, [])


# --- 1. connector grammar repair -------------------------------------------


def test_connector_comma_inserted_before_pronoun_main_clause():
    prompt = "The baseball player lifted the bat before the swing"
    text = f"{prompt} he steadied himself"
    repaired, rule = repair_connector_grammar(prompt, text, ENDING_BEFORE_SUBJECT)
    assert rule == "connector_comma"
    assert "before the swing," in repaired
    assert repaired == "The baseball player lifted the bat before the swing, he steadied himself"


def test_connector_comma_not_inserted_for_verb_completion():
    # "before the bird spread its wings" is a single subordinate clause: no comma.
    prompt = "The crane was silent for a long time before the bird"
    text = f"{prompt} spread its wings"
    repaired, rule = repair_connector_grammar(prompt, text, ENDING_BEFORE_SUBJECT)
    assert rule is None
    assert repaired == text


def test_connector_comma_end_to_end():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt(
        "The baseball player lifted the bat before the swing"
    )
    assert result.decision == "continue"
    assert "before the swing," in result.continuation


# --- 2 & 3. subject / action validation ------------------------------------


def test_body_part_whole_animal_action_rejected():
    result = validate_subject_action("its wings searched for insects")
    assert result.ok is False
    assert result.body_part == "wings"
    assert result.verb == "searched"


def test_allowed_body_part_action_passes():
    assert validate_subject_action("its wings spread wide").ok is True
    assert validate_subject_action("its flippers moved through the water").ok is True


def test_body_part_action_repaired_to_safe_clause():
    prompt = "The bat rested in the corner and only after sunset its wings"
    text = f"{prompt} searched for insects"
    frame = _frame(prompt, "bat", "animal")
    result = repair_surface_candidate(prompt, text, frame)
    assert result.safe is True
    assert "subject_action_bodypart_rewrite" in result.rules
    assert result.text == f"{prompt} {safe_body_part_clause('wings')}"
    # The repaired text must itself pass role validation.
    assert validate_subject_action(result.text).ok is True


# --- 4. repeated object repair ---------------------------------------------


def test_repeated_object_becomes_pronoun():
    prompt = "She pressed the wax seal onto the envelope to"
    text = f"{prompt} close the envelope"
    new_text, rule = repair_repetition(prompt, text, "closure_stamp")
    assert rule == "coreference_pronoun"
    assert new_text == "She pressed the wax seal onto the envelope to close it"


# --- 5. fish / prey repair --------------------------------------------------


def test_prey_repetition_repaired():
    prompt = "The seal swam through ocean water chasing fish to"
    text = f"{prompt} catch another fish"
    new_text, rule = repair_repetition(prompt, text, "animal")
    assert rule == "coreference_prey"
    assert new_text == "The seal swam through ocean water chasing fish to catch its prey"


# --- 6 & 7. attachment repair / unsafe routes to audit ----------------------


def test_attachment_drift_routes_to_audit():
    prompt = "The rock was ignored until the climber saw it on the cliff"
    text = f"{prompt} and lay near the river"
    frame = _frame(prompt, "rock", "stone")
    result = repair_surface_candidate(prompt, text, frame)
    assert result.safe is False
    assert result.audit_reason == AUDIT_REASON
    assert "attachment_drift" in result.reasons


def test_attachment_drift_end_to_end_audits():
    engine = ControlledContinuationEngine()
    result = engine.continue_prompt(
        "The rock was ignored until the climber saw it on the cliff"
    )
    assert result.decision == "audit"
    assert result.continuation == ""
    assert any("no_safe_repaired_candidate" in r for r in result.reasons)


def test_compound_predicate_over_single_subject_is_not_drift():
    # Same neutral phrase, but no intervening subordinate actor -> safe.
    prompt = "The heavy rock rolled from the mountain to the ground"
    text = f"{prompt} and lay near the river"
    frame = _frame(prompt, "rock", "stone")
    result = repair_surface_candidate(prompt, text, frame)
    assert result.safe is True


def test_non_positional_compound_over_local_actor_is_not_drift():
    # "the drummer came on stage and filled the stadium" attaches cleanly.
    prompt = "The rock sounded flat until the drummer came on stage"
    text = f"{prompt} and filled the stadium"
    frame = _frame(prompt, "rock", "music")
    result = repair_surface_candidate(prompt, text, frame)
    assert result.safe is True


# --- 8. determinism ---------------------------------------------------------


def test_repair_is_deterministic():
    prompt = "The bat rested in the corner and only after sunset its wings"
    text = f"{prompt} searched for insects"
    frame = _frame(prompt, "bat", "animal")
    first = repair_surface_candidate(prompt, text, frame)
    second = repair_surface_candidate(prompt, text, frame)
    assert first.text == second.text
    assert first.rules == second.rules
    assert first.reasons == second.reasons


def test_engine_repair_is_deterministic_end_to_end():
    engine = ControlledContinuationEngine()
    prompt = "The seal swam through ocean water chasing fish to"
    a = engine.continue_prompt(prompt)
    b = engine.continue_prompt(prompt)
    assert a.continuation == b.continuation
    assert a.memory_hits == b.memory_hits


# --- clean candidates pass through untouched --------------------------------


def test_clean_candidate_unchanged_and_marked_clean():
    prompt = "The customer reached the bank teller with cash to"
    text = f"{prompt} open an account"
    frame = _frame(prompt, "bank", "financial_institution")
    result = repair_surface_candidate(prompt, text, frame)
    assert result.safe is True
    assert result.applied is False
    assert result.text == text
    assert "surface_repair=clean" in result.reasons
