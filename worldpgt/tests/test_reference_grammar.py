"""Closed reference grammar: slot detection is enumerable and conservative."""

from __future__ import annotations

from worldpgt.dialogue import reference_grammar as G
from worldpgt.tests._dialogue_mocks import MockSurfaceIndex

INDEX = MockSurfaceIndex({
    "SpaceX": ("organization", []),
    "Tesla": ("organization", []),
    "Elon Musk": ("person", ["Musk"]),
    "Stop It Now": ("organization", []),
})


def _classes(question: str) -> list[str]:
    return [s.ref_class for s in G.detect_slots(question, INDEX).slots]


def test_person_pronoun_detected_with_gate():
    parse = G.detect_slots("What else did he found?", INDEX)
    assert [s.ref_class for s in parse.slots] == [G.PRONOUN_PERSON]
    assert parse.slots[0].type_gate == G.PERSON_TYPES
    assert parse.has_exclusion


def test_thing_pronoun_and_possessive_flag():
    parse = G.detect_slots("What are its products?", INDEX)
    assert [s.ref_class for s in parse.slots] == [G.PRONOUN_THING]
    assert parse.slots[0].possessive
    assert "person" not in parse.slots[0].type_gate


def test_plural_pronoun():
    assert _classes("What do they develop?") == [G.PRONOUN_PLURAL]


def test_typed_demonstrative_beats_bare_pronoun_overlap():
    parse = G.detect_slots("What does that company develop?", INDEX)
    assert [s.ref_class for s in parse.slots] == [G.DEMONSTRATIVE_TYPED]
    assert parse.slots[0].type_gate == frozenset({"organization"})
    assert parse.slots[0].surface.lower() == "that company"


def test_role_descriptor_maps_relation():
    parse = G.detect_slots("What did the founder study?", INDEX)
    assert [s.ref_class for s in parse.slots] == [G.ROLE_DESCRIPTOR]
    assert parse.slots[0].role_relation == "founded_by"
    assert parse.slots[0].type_gate == G.PERSON_TYPES


def test_contrastive_with_and_without_noun():
    with_noun = G.detect_slots("Who founded the other company?", INDEX).slots
    assert [s.ref_class for s in with_noun] == [G.CONTRASTIVE]
    assert with_noun[0].type_gate == frozenset({"organization"})

    bare = G.detect_slots("Who founded the other one?", INDEX).slots
    assert [s.ref_class for s in bare] == [G.CONTRASTIVE]
    assert bare[0].type_gate is None  # inherits focus type in the resolver


def test_selective_only_question_initial():
    parse = G.detect_slots("Which one develops reusable rockets?", INDEX)
    assert [s.ref_class for s in parse.slots] == [G.SELECTIVE]


def test_bare_demonstrative_start_and_prepositional():
    assert _classes("Tell me more about that") == [G.DEMONSTRATIVE_BARE]


def test_relative_that_is_not_a_reference():
    # "that" as a relative pronoun must not become a slot.
    parse = G.detect_slots("Which company that Musk founded builds rockets?", INDEX)
    assert G.DEMONSTRATIVE_BARE not in [s.ref_class for s in parse.slots]


def test_topic_shift_detected_for_known_entity_only():
    shift = G.detect_slots("What about Tesla?", INDEX)
    assert shift.topic_shift_surface == "Tesla"
    assert shift.slots == ()

    no_shift = G.detect_slots("What about his brother?", INDEX)
    assert no_shift.topic_shift_surface is None
    assert [s.ref_class for s in no_shift.slots] == [G.PRONOUN_PERSON]


def test_elliptical_only_when_no_content_residue():
    ell = G.detect_slots("Who founded?", INDEX)
    assert [s.ref_class for s in ell.slots] == [G.ELLIPTICAL]
    assert ell.slots[0].role_relation == "founded_by"

    full = G.detect_slots("Who founded the first private space company?", INDEX)
    assert G.ELLIPTICAL not in [s.ref_class for s in full.slots]


def test_pronoun_inside_entity_surface_is_not_a_slot():
    parse = G.detect_slots("Tell me about Stop It Now.", INDEX)
    assert parse.slots == ()


def test_no_slots_for_plain_question():
    parse = G.detect_slots("What does SpaceX develop?", INDEX)
    assert parse.slots == ()
    assert parse.topic_shift_surface is None


def test_unknown_reference_form_is_not_fuzzy_matched():
    # "the aforementioned entity" is not in the grammar → no slot, and the
    # question simply parses as an ordinary (unresolvable-entity) question.
    parse = G.detect_slots("What does the aforementioned entity develop?", INDEX)
    assert parse.slots == ()
