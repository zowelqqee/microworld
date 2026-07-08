"""Salience scoring: integer, itemized, reproducible from visible state."""

from __future__ import annotations

from worldpgt.dialogue import constants as C
from worldpgt.dialogue.salience import base_salience, slot_salience, type_gate_passes
from worldpgt.dialogue.state import AnswerEntity, DialogueState, TurnRecord


def _state_with_topic() -> DialogueState:
    state = DialogueState()
    state.commit(TurnRecord(
        question="Tell me about SpaceX.",
        user_named=("SpaceX",),
        answer_entities=(AnswerEntity("SpaceX", "answer_subject"),),
        topic_op=("set", "SpaceX"),
        question_subject="SpaceX",
    ))
    return state


def test_active_topic_dominates_breakdown():
    state = _state_with_topic()
    total, parts = base_salience(state.entities["SpaceX"], state)
    names = dict(parts)
    assert names["active_topic"] == C.ACTIVE_TOPIC
    assert names["last_answer_entity"] == C.LAST_ANSWER_ENTITY
    assert names["last_question_subject"] == C.LAST_QUESTION_SUBJECT
    assert names["user_named"] == C.USER_NAMED
    assert total == sum(points for _n, points in parts)


def test_recency_penalty_is_linear_and_floored():
    state = _state_with_topic()
    state.commit(TurnRecord(question="x", user_named=("Tesla",), topic_op=("set", "Tesla")))
    spacex = state.entities["SpaceX"]

    floored = False
    # Up to the eviction horizon (idle == EVICT_AFTER_TURNS drops the entity).
    for extra_turns in range(1, C.EVICT_AFTER_TURNS - 1):
        state.commit(TurnRecord(question=f"t{extra_turns}", user_named=("Tesla",)))
        _total, parts = base_salience(state.entities["SpaceX"], state)
        idle = state.confirmed_counter - spacex.last_mention_confirmed
        expected = max(C.RECENCY_FLOOR, -C.RECENCY_PENALTY_PER_TURN * idle)
        assert dict(parts)["recency"] == expected
        floored = floored or expected == C.RECENCY_FLOOR
    # The floor is actually reachable before eviction — never exceeded.
    assert floored


def test_mention_bonus_caps():
    state = DialogueState()
    for _ in range(C.MENTION_BONUS_CAP + 3):
        state.commit(TurnRecord(question="x", user_named=("SpaceX",)))
    _total, parts = base_salience(state.entities["SpaceX"], state)
    assert dict(parts)["mentions"] == C.MENTION_BONUS * C.MENTION_BONUS_CAP


def test_sticky_referent_bonus():
    state = _state_with_topic()
    state.last_referent["pronoun_thing"] = "SpaceX"
    total, parts = slot_salience(
        state.entities["SpaceX"], state, ref_class="pronoun_thing")
    assert dict(parts)["sticky_referent"] == C.STICKY_REFERENT
    assert total == sum(points for _n, points in parts)


def test_role_match_requires_active_anchor():
    state = DialogueState()
    state.commit(TurnRecord(
        question="Who founded SpaceX?",
        user_named=("SpaceX",),
        topic_op=("set", "SpaceX"),
        answer_entities=(AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),),
    ))
    musk = state.entities["Elon Musk"]
    with_anchor = slot_salience(
        musk, state, ref_class="role_descriptor",
        role_relation="founded_by", role_anchors_active=frozenset({"SpaceX"}))
    without_anchor = slot_salience(
        musk, state, ref_class="role_descriptor",
        role_relation="founded_by", role_anchors_active=frozenset())
    assert dict(with_anchor[1]).get("role_match") == C.ROLE_MATCH
    assert "role_match" not in dict(without_anchor[1])


def test_type_gate_is_a_hard_filter():
    assert type_gate_passes("organization", frozenset({"organization"}))
    assert not type_gate_passes("person", frozenset({"organization"}))
    assert type_gate_passes("person", None)  # any *known* type
    assert not type_gate_passes(None, None)  # unknown type never passes


def test_scoring_is_deterministic():
    state_a = _state_with_topic()
    state_b = _state_with_topic()
    assert base_salience(state_a.entities["SpaceX"], state_a) == \
        base_salience(state_b.entities["SpaceX"], state_b)
