"""Dialogue resolver: pure-function resolution with margin-gated honesty."""

from __future__ import annotations

from worldpgt.dialogue import constants as C
from worldpgt.dialogue.bound_index import BoundSurfaceIndex
from worldpgt.dialogue.resolver import resolve_question
from worldpgt.dialogue.state import AnswerEntity, DialogueState, TurnRecord
from worldpgt.tests._dialogue_mocks import MockGraphReader, MockSurfaceIndex

INDEX = MockSurfaceIndex({
    "SpaceX": ("organization", []),
    "Blue Origin": ("organization", []),
    "Tesla": ("organization", []),
    "Starlink": ("organization", []),
    "Elon Musk": ("person", ["Musk"]),
    "Jeff Bezos": ("person", ["Bezos"]),
    "Falcon 9": ("vehicle", []),
})


def _commit_about(state: DialogueState, entity: str, *, extra=()) -> None:
    state.commit(TurnRecord(
        question=f"Tell me about {entity}.",
        user_named=(entity,),
        answer_entities=(AnswerEntity(entity, "answer_subject"), *extra),
        topic_op=("set", entity),
        question_subject=entity,
    ))


def test_no_slots_is_a_structural_noop():
    resolved = resolve_question("What does SpaceX develop?", DialogueState(), INDEX)
    assert resolved.outcome == "no_slots"
    assert resolved.bindings == ()


def test_cold_start_pronoun_audits():
    resolved = resolve_question("Who founded it?", DialogueState(), INDEX)
    assert resolved.outcome == "unresolved"
    assert resolved.slots[0].outcome == "no_candidate"


def test_pronoun_resolves_to_topic_with_full_trace():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    resolved = resolve_question("What does it build?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.resolved_references == ["[it → SpaceX]"]
    top = resolved.slots[0].candidates[0]
    assert top.canonical == "SpaceX"
    assert dict(top.breakdown)["active_topic"] == C.ACTIVE_TOPIC
    assert top.total == sum(points for _n, points in top.breakdown)


def test_person_gate_excludes_organizations():
    state = DialogueState()
    _commit_about(state, "SpaceX",
                  extra=(AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),))
    resolved = resolve_question("What else did he found?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.resolved_references == ["[he → Elon Musk]"]
    named = {c.canonical for c in resolved.slots[0].candidates}
    assert "SpaceX" not in named  # organizations never reach a person pronoun


def test_topic_shift_redirects_pronouns():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    shift = resolve_question("What about Tesla?", state, INDEX)
    assert shift.outcome == "resolved"
    assert shift.directives.topic_op == ("set", "Tesla")
    assert shift.directives.reformulated_question == "Tell me about Tesla."
    _commit_about(state, "Tesla")

    resolved = resolve_question("Who founded it?", state, INDEX)
    assert resolved.resolved_references == ["[it → Tesla]"]
    scores = {c.canonical: c.total for c in resolved.slots[0].candidates}
    assert scores["Tesla"] - scores["SpaceX"] >= C.RESOLVE_MARGIN


def test_ambiguous_margin_audits_with_candidates():
    state = DialogueState()
    # Two organizations registered symmetrically in one turn — near-tied.
    state.commit(TurnRecord(
        question="Compare SpaceX and Blue Origin.",
        user_named=("SpaceX", "Blue Origin"),
        answer_entities=(
            AnswerEntity("SpaceX", "answer_subject"),
            AnswerEntity("Blue Origin", "answer_subject"),
        ),
    ))
    resolved = resolve_question("Who founded it?", state, INDEX)
    assert resolved.outcome == "unresolved"
    slot = resolved.slots[0]
    assert slot.outcome == "ambiguous"
    assert slot.margin is not None and slot.margin < C.RESOLVE_MARGIN
    assert {c.canonical for c in slot.candidates} == {"SpaceX", "Blue Origin"}
    assert resolved.bindings == ()  # nothing binds on an unresolved question


def test_plural_resolves_exactly_two_active():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    _commit_about(state, "Blue Origin")
    resolved = resolve_question("What do they develop?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert set(resolved.slots[0].entities) == {"SpaceX", "Blue Origin"}
    assert resolved.bindings[0].canonicals == resolved.slots[0].entities


def test_selective_produces_directive_set():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    _commit_about(state, "Blue Origin")
    resolved = resolve_question("Which one develops reusable rockets?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.slots[0].outcome == "resolved_set"
    assert set(resolved.directives.selective_set) == {"SpaceX", "Blue Origin"}


def test_contrastive_excludes_focus_and_pivots_topic():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    _commit_about(state, "Blue Origin")
    # "Which one develops reusable rockets?" → answer SpaceX (last answer).
    state.commit(TurnRecord(
        question="Which one develops reusable rockets?",
        answer_entities=(AnswerEntity("SpaceX", "answer_subject"),),
        relation_intent="develops",
    ))
    resolved = resolve_question("Who founded the other one?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.resolved_references == ["[the other one → Blue Origin]"]
    assert resolved.directives.topic_op == ("set", "Blue Origin")


def test_contrastive_pool_is_topics_only():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    _commit_about(state, "Blue Origin")
    # Starlink named as a relation object — never a topic, so it must not
    # widen the contrast set.
    state.commit(TurnRecord(
        question="Which company owns Starlink?",
        user_named=("Starlink",),
        answer_entities=(AnswerEntity("SpaceX", "answer_subject"),),
        relation_intent="owned_by",
        surfaced_relations=(("Starlink", "owned_by", "SpaceX"),),
    ))
    resolved = resolve_question("Who founded the other one?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.slots[0].entities == ("Blue Origin",)


def test_what_else_exclusion_is_directionless():
    state = DialogueState()
    _commit_about(state, "SpaceX",
                  extra=(AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),))
    state.commit(TurnRecord(
        question="Who founded it?",
        question_subject="SpaceX",
        relation_intent="founded_by",
        answer_entities=(AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),),
        surfaced_relations=(("SpaceX", "founded_by", "Elon Musk"),),
    ))
    resolved = resolve_question("What else did he found?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.directives.exclusion_subject == "Elon Musk"
    assert resolved.directives.exclusion_relation == "founded_by"
    # Musk appears as the *object* of (SpaceX, founded_by, Musk); the surfaced
    # subject SpaceX is what gets excluded.
    assert resolved.directives.exclude_objects == ("SpaceX",)


def test_role_descriptor_dialogue_pass():
    state = DialogueState()
    _commit_about(state, "SpaceX",
                  extra=(AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),))
    resolved = resolve_question("Where did the founder study?", state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.slots[0].strategy == "dialogue_role"
    assert resolved.slots[0].entities == ("Elon Musk",)


def test_role_descriptor_graph_pass_single_and_multiple():
    state = DialogueState()
    _commit_about(state, "SpaceX")

    single = MockGraphReader([("SpaceX", "founded_by", "Elon Musk")])
    resolved = resolve_question("Where did the founder study?", state, INDEX, single)
    assert resolved.outcome == "resolved"
    assert resolved.slots[0].strategy == "graph_verified"
    assert resolved.slots[0].entities == ("Elon Musk",)

    multiple = MockGraphReader([
        ("SpaceX", "founded_by", "Elon Musk"),
        ("SpaceX", "founded_by", "Jeff Bezos"),
    ])
    resolved = resolve_question("Where did the founder study?", state, INDEX, multiple)
    assert resolved.outcome == "unresolved"
    assert resolved.slots[0].outcome == "ambiguous"


def test_elliptical_binds_virtual_span_to_topic():
    state = DialogueState()
    _commit_about(state, "Starlink")
    question = "Who founded?"
    resolved = resolve_question(question, state, INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.directives.answer_style == "followup"
    span = resolved.bindings[0]
    assert (span.start, span.end) == (len(question), len(question))
    assert span.canonicals == ("Starlink",)


def test_same_question_mention_enables_intra_turn_anaphora():
    resolved = resolve_question(
        "SpaceX builds rockets, but who founded it?", DialogueState(), INDEX)
    assert resolved.outcome == "resolved"
    assert resolved.resolved_references == ["[it → SpaceX]"]
    assert dict(resolved.slots[0].candidates[0].breakdown)["same_question_mention"] == \
        C.SAME_QUESTION_MENTION


def test_resolution_is_pure_and_repeatable():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    snapshot = state.to_dict()
    first = resolve_question("What does it build?", state, INDEX)
    second = resolve_question("What does it build?", state, INDEX)
    assert first == second
    assert state.to_dict() == snapshot  # resolution never mutates state


# ── BoundSurfaceIndex transport ──────────────────────────────────────────────


def test_bound_index_injects_and_passes_through():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    question = "What does it build?"
    resolved = resolve_question(question, state, INDEX)
    bound = BoundSurfaceIndex(INDEX, resolved.bindings)

    rows = bound.find_in_text(question)
    assert ("it", "SpaceX") in {(surface, canonical) for surface, canonical, _s, _e in rows}
    assert bound.resolve("it") == "SpaceX"
    assert bound.resolve("Tesla") == "Tesla"  # inner passthrough
    span = resolved.bindings[0]
    assert bound.is_bound_span(span.start, span.end)
    assert not bound.is_bound_span(0, 4)
    assert bound.entity_type("SpaceX") == "organization"


def test_bound_index_multi_canonical_span():
    state = DialogueState()
    _commit_about(state, "SpaceX")
    _commit_about(state, "Blue Origin")
    question = "What do they develop?"
    resolved = resolve_question(question, state, INDEX)
    bound = BoundSurfaceIndex(INDEX, resolved.bindings)
    canonicals = [c for _s, c, _st, _e in bound.find_in_text(question)]
    assert set(canonicals) == {"SpaceX", "Blue Origin"}
