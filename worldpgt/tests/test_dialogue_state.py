"""DialogueState: commit is the only mutator; replay reconstructs exactly."""

from __future__ import annotations

from worldpgt.dialogue import constants as C
from worldpgt.dialogue.state import AnswerEntity, DialogueState, TurnRecord


def _turn(question="q", **kwargs) -> TurnRecord:
    return TurnRecord(question=question, **kwargs)


def test_commit_registers_user_named_and_answer_entities():
    state = DialogueState()
    state.commit(_turn(
        user_named=("SpaceX",),
        answer_entities=(
            AnswerEntity("SpaceX", "answer_subject"),
            AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),
        ),
        surfaced_relations=(("SpaceX", "founded_by", "Elon Musk"),),
        topic_op=("set", "SpaceX"),
    ))
    assert state.turn_counter == 1
    assert state.confirmed_counter == 1
    assert state.active_topic == "SpaceX"
    assert set(state.entities) == {"SpaceX", "Elon Musk"}
    musk = state.entities["Elon Musk"]
    assert musk.dialogue_roles == (("founded_by", "SpaceX", 1),)
    assert state.mentioned_relations == (("SpaceX", "founded_by", "Elon Musk", 1),)


def test_audit_turn_freezes_decay_clock():
    state = DialogueState()
    state.commit(_turn(user_named=("SpaceX",), topic_op=("set", "SpaceX")))
    before = state.entities["SpaceX"].last_mention_confirmed
    for _ in range(5):
        state.commit(_turn(answer_decision="audit"))
    # Audit turns advance turn_counter but not the confirmed clock, so the
    # entity has not aged at all.
    assert state.turn_counter == 6
    assert state.confirmed_counter == 1
    assert state.confirmed_counter - before == 0
    assert "SpaceX" in state.entities


def test_topic_change_marks_was_topic_and_previous():
    state = DialogueState()
    state.commit(_turn(user_named=("SpaceX",), topic_op=("set", "SpaceX")))
    state.commit(_turn(user_named=("Tesla",), topic_op=("set", "Tesla")))
    assert state.active_topic == "Tesla"
    assert state.previous_topic == "SpaceX"
    assert state.entities["SpaceX"].was_topic
    assert not state.entities["Tesla"].was_topic


def test_user_named_upgrades_introduction_source():
    state = DialogueState()
    state.commit(_turn(answer_entities=(AnswerEntity("Elon Musk", "answer_object"),)))
    assert state.entities["Elon Musk"].introduction_source == "answer_object"
    state.commit(_turn(user_named=("Elon Musk",)))
    assert state.entities["Elon Musk"].introduction_source == "user_named"
    assert state.entities["Elon Musk"].mention_count == 2


def test_eviction_after_idle_confirmed_turns():
    state = DialogueState()
    state.commit(_turn(user_named=("Old Corp",)))
    evicted: list[str] = []
    for i in range(C.EVICT_AFTER_TURNS):
        evicted += state.commit(_turn(user_named=(f"E{i}",)))
    assert "Old Corp" in evicted
    assert "Old Corp" not in state.entities


def test_active_topic_never_evicted_by_idle():
    state = DialogueState()
    state.commit(_turn(user_named=("SpaceX",), topic_op=("set", "SpaceX")))
    for i in range(C.EVICT_AFTER_TURNS + 2):
        state.commit(_turn(user_named=(f"E{i}",)))
    assert "SpaceX" in state.entities


def test_registry_cap_evicts_lowest_salience_first():
    state = DialogueState()
    for i in range(C.REGISTRY_CAP + 3):
        state.commit(_turn(user_named=(f"E{i:02d}",)))
    assert len(state.entities) <= C.REGISTRY_CAP
    # The most recent entities survive; the stalest were evicted.
    assert f"E{C.REGISTRY_CAP + 2:02d}" in state.entities
    assert "E00" not in state.entities


def test_last_referent_tracking():
    state = DialogueState()
    state.commit(_turn(resolved_referents=(("pronoun_thing", "SpaceX"),)))
    assert state.last_referent["pronoun_thing"] == "SpaceX"


def test_serialization_round_trip():
    state = DialogueState()
    state.commit(_turn(
        user_named=("SpaceX",),
        answer_entities=(AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),),
        surfaced_relations=(("SpaceX", "founded_by", "Elon Musk"),),
        topic_op=("set", "SpaceX"),
        question_subject="SpaceX",
        relation_intent="founded_by",
        resolved_referents=(("pronoun_thing", "SpaceX"),),
        entity_type_hints=(("Elon Musk", "person"),),
    ))
    restored = DialogueState.from_dict(state.to_dict())
    assert restored.to_dict() == state.to_dict()


def test_replay_equals_live_state():
    records = [
        _turn("Tell me about SpaceX.", user_named=("SpaceX",), topic_op=("set", "SpaceX"),
              answer_entities=(AnswerEntity("SpaceX", "answer_subject"),)),
        _turn("Who founded it?", question_subject="SpaceX", relation_intent="founded_by",
              answer_entities=(AnswerEntity("Elon Musk", "answer_object", "founded_by", "SpaceX"),),
              surfaced_relations=(("SpaceX", "founded_by", "Elon Musk"),),
              resolved_referents=(("pronoun_thing", "SpaceX"),)),
        _turn("audit turn", answer_decision="audit"),
        _turn("What about Tesla?", user_named=("Tesla",), topic_op=("set", "Tesla")),
    ]
    live = DialogueState()
    for record in records:
        live.commit(record)
    assert DialogueState.replay(records).to_dict() == live.to_dict()


def test_turn_record_round_trip():
    record = _turn(
        user_named=("A",),
        answer_entities=(AnswerEntity("B", "answer_object", "owned_by", "A", "person"),),
        surfaced_relations=(("A", "owned_by", "B"),),
        topic_op=("set", "A"),
        resolved_referents=(("pronoun_person", "B"),),
        entity_type_hints=(("B", "person"),),
    )
    assert TurnRecord.from_dict(record.to_dict()) == record
