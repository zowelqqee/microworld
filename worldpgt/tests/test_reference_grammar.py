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
    # Permissive gate: a role's holder isn't always a person (an "owner" can
    # be a company) -- the role_relation itself narrows candidates, not the
    # type gate.
    assert parse.slots[0].type_gate is None


def test_role_descriptor_open_vocabulary_via_shared_relation_table():
    # These nouns resolve without a private word list -- they go through
    # the same relation_policy keyword table the rest of Microworld uses,
    # not a duplicated dialogue-only dict.
    for question, expected_relation in [
        ("Who is the leader?", "leader_of"),
        ("Who is the head?", "leader_of"),
        ("Who is the owner?", "owned_by"),
    ]:
        parse = G.detect_slots(question, INDEX)
        assert [s.ref_class for s in parse.slots] == [G.ROLE_DESCRIPTOR], question
        assert parse.slots[0].role_relation == expected_relation, question


def test_role_descriptor_does_not_fire_on_self_contained_question():
    # "the founder of PayPal" names its anchor explicitly right there -- this
    # is a complete question for ordinary parsing, not an elliptical dialogue
    # reference, so no role_descriptor slot should form.
    parse = G.detect_slots("Where was the founder of PayPal born?", INDEX)
    assert G.ROLE_DESCRIPTOR not in [s.ref_class for s in parse.slots]


def test_type_noun_does_not_collide_with_relation_keyword_table():
    # "product"/"service" happen to have "product_of"/"service_of" entries in
    # the shared relation table, but that's a keyword-table coincidence, not
    # role semantics -- a non-person type must win outright.
    for question, expected_type in [
        ("Tell me about that product.", "product"),
        ("Tell me about that service.", "service"),
    ]:
        parse = G.detect_slots(question, INDEX)
        assert [s.ref_class for s in parse.slots] == [G.DEMONSTRATIVE_TYPED], question
        assert parse.slots[0].type_gate == frozenset({expected_type}), question


def test_unrecognized_role_noun_form_no_slot():
    # "creator" is not in the shared relation table (no exact entry, and the
    # embedding fallback is deliberately not used here because it false-positives
    # on ordinary abstract nouns) -- honest gap, no slot, never fuzzy-matched.
    assert G.detect_slots("Who is the creator?", INDEX).slots == ()


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
