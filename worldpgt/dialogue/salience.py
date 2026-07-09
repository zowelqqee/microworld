"""Deterministic salience scoring for dialogue entity candidates.

Pure functions over :class:`~worldpgt.dialogue.state.DialogueState`. All
arithmetic is integer; every point awarded appears in the returned breakdown
under the exact constant name from :mod:`worldpgt.dialogue.constants`, so a
trace is a complete proof of the score.
"""

from __future__ import annotations

from worldpgt.dialogue import constants as C
from worldpgt.dialogue.state import DialogueState, EntityActivation

Breakdown = tuple[tuple[str, int], ...]


def base_salience(
    activation: EntityActivation, state: DialogueState
) -> tuple[int, Breakdown]:
    """Slot-independent salience of one entity. Deterministic in
    (activation, state); stores nothing."""

    parts: list[tuple[str, int]] = []

    if state.active_topic == activation.canonical:
        parts.append(("active_topic", C.ACTIVE_TOPIC))
    elif activation.was_topic:
        parts.append(("was_topic", C.WAS_TOPIC))

    if state.last_answer is not None and activation.canonical in state.last_answer.entities:
        parts.append(("last_answer_entity", C.LAST_ANSWER_ENTITY))

    if state.last_question is not None and state.last_question.subject == activation.canonical:
        parts.append(("last_question_subject", C.LAST_QUESTION_SUBJECT))

    if activation.introduction_source == "user_named":
        parts.append(("user_named", C.USER_NAMED))

    idle = state.confirmed_counter - activation.last_mention_confirmed
    if idle > 0:
        penalty = max(C.RECENCY_FLOOR, -C.RECENCY_PENALTY_PER_TURN * idle)
        parts.append(("recency", penalty))

    bonus = C.MENTION_BONUS * min(activation.mention_count, C.MENTION_BONUS_CAP)
    if bonus:
        parts.append(("mentions", bonus))

    total = sum(points for _name, points in parts)
    return total, tuple(parts)


def slot_salience(
    activation: EntityActivation,
    state: DialogueState,
    *,
    ref_class: str,
    role_relation: str | None = None,
    role_anchors_active: frozenset[str] = frozenset(),
    same_question_named: frozenset[str] = frozenset(),
) -> tuple[int, Breakdown]:
    """Base salience plus slot-dependent bonuses (role match, sticky referent,
    same-question mention). Still pure and integer-only."""

    total, parts = base_salience(activation, state)
    extra: list[tuple[str, int]] = list(parts)

    if role_relation is not None:
        for relation, anchor, _turn in activation.dialogue_roles:
            if relation == role_relation and anchor in role_anchors_active:
                extra.append(("role_match", C.ROLE_MATCH))
                total += C.ROLE_MATCH
                break

    if state.last_referent.get(ref_class) == activation.canonical:
        extra.append(("sticky_referent", C.STICKY_REFERENT))
        total += C.STICKY_REFERENT

    if activation.canonical in same_question_named:
        extra.append(("same_question_mention", C.SAME_QUESTION_MENTION))
        total += C.SAME_QUESTION_MENTION

    return total, tuple(extra)


def type_gate_passes(
    etype: str | None,
    gate: frozenset[str] | None,
) -> bool:
    """Hard filter, applied before any scoring. ``None`` gate means "any
    *known* type". ``etype`` is resolved by the caller from the activation's
    session hint or the trusted entity index — both deterministic reads."""

    if gate is None:
        return etype is not None
    return etype in gate
