"""Serving-path glue for the dialogue-v2 layer.

Everything the API server needs to run the resolver per request:

* :class:`OverlayGraphReader` — the read-only role-holder lookup over loaded
  overlay items (built once at startup);
* :func:`serialize_bindings` — stage-A transport: renders already-decided
  bindings into an effective question for the unchanged orchestrator. The
  *decision* is the resolver's (traced, margin-gated); this function only
  serializes it. Stage B (threading ``BoundSurfaceIndex`` through the
  orchestrator) replaces this without changing any resolution behavior;
* :func:`build_turn_record` — turns one answered request into the
  :class:`~worldpgt.dialogue.state.TurnRecord` that commits into
  ``DialogueState`` (port of the v1 ``_record_turn`` answer-scanning logic);
* :func:`dialogue_mode` — the ``MICROWORLD_DIALOGUE_V2`` flag
  (``off`` | ``shadow`` | ``on``, default ``shadow``).
"""

from __future__ import annotations

import os

from worldpgt.dialogue.resolver import ResolvedQuestion
from worldpgt.dialogue.state import AnswerEntity, TurnRecord

_DEFINITION_QUERY_TYPES = frozenset({"definition", "open_synthesis"})


def dialogue_mode() -> str:
    value = os.environ.get("MICROWORLD_DIALOGUE_V2", "shadow").strip().lower()
    return value if value in {"off", "shadow", "on"} else "shadow"


class OverlayGraphReader:
    """Role-holder lookup over overlay relation items, both directions
    (overlay stores (SpaceX, founded_by, Musk) but (Musk, leader_of, Tesla)).
    Read-only; built once from the already-loaded overlay items."""

    def __init__(self, overlay_items: list[dict]) -> None:
        self._forward: dict[tuple[str, str], list[str]] = {}
        self._reverse: dict[tuple[str, str], list[str]] = {}
        for item in overlay_items:
            if item.get("overlay_type") != "overlay_relation":
                continue
            subject = str(item.get("subject") or "").strip()
            predicate = str(item.get("predicate") or "").strip()
            obj = str(item.get("object") or "").strip()
            if not subject or not predicate or not obj:
                continue
            self._forward.setdefault((subject.lower(), predicate), [])
            if obj not in self._forward[(subject.lower(), predicate)]:
                self._forward[(subject.lower(), predicate)].append(obj)
            self._reverse.setdefault((obj.lower(), predicate), [])
            if subject not in self._reverse[(obj.lower(), predicate)]:
                self._reverse[(obj.lower(), predicate)].append(subject)

    def role_holders(self, anchor: str, relation: str) -> tuple[str, ...]:
        key = (anchor.lower(), relation)
        out: list[str] = []
        for holder in self._forward.get(key, []) + self._reverse.get(key, []):
            if holder not in out:
                out.append(holder)
        return tuple(out)


def serialize_bindings(question: str, resolved: ResolvedQuestion) -> str:
    """Render resolver decisions into an effective question (stage-A transport).

    Canonical names are exact index keys, so the parser re-recognizes them
    with a guaranteed round-trip; no resolution happens downstream.
    """

    if resolved.directives.reformulated_question:
        return resolved.directives.reformulated_question
    if not resolved.bindings:
        return question

    out = question
    virtual: list[str] = []
    offset = 0
    for span in sorted(resolved.bindings, key=lambda b: b.start):
        if span.start == span.end == len(question):
            # Elliptical: the missing subject is appended, not substituted.
            virtual.extend(span.canonicals)
            continue
        replacement = " and ".join(span.canonicals)
        if span.possessive:
            replacement += "'s"
        start = span.start + offset
        end = span.end + offset
        out = out[:start] + replacement + out[end:]
        offset += len(replacement) - (span.end - span.start)

    if virtual:
        stripped = out.rstrip()
        trailing = "?" if stripped.endswith("?") else ""
        stripped = stripped.rstrip("?.! ").rstrip()
        out = f"{stripped} {' and '.join(virtual)}{trailing or '?'}"
    return out


def build_turn_record(
    *,
    question: str,
    resolved: ResolvedQuestion | None,
    semantic_query,
    answer,
    surface_index,
    answer_text_entities: list[str],
    entity_type_hints: dict[str, str] | None = None,
) -> TurnRecord:
    """Build the commit record for one served turn.

    ``answer_text_entities`` are the canonical entities found in the rendered
    answer text (the server already scans for them); ``entity_type_hints``
    carries session-only hints such as the web-search person hint.
    """

    hints_tuple = tuple(sorted((entity_type_hints or {}).items()))
    user_named = _dedupe(
        canonical for _s, canonical, _st, _e in surface_index.find_in_text(question)
    )
    resolved_referents: tuple[tuple[str, str], ...] = ()
    directives_topic: tuple[str, ...] = ("keep",)
    if resolved is not None:
        resolved_referents = tuple(
            (res.slot.ref_class, res.entities[0])
            for res in resolved.slots
            if res.outcome == "resolved"
            and len(res.entities) == 1
            and res.slot.ref_class != "topic_shift"
        )
        directives_topic = resolved.directives.topic_op

    if answer.decision == "audit":
        return TurnRecord(
            question=question,
            user_named=user_named,
            answer_decision="audit",
            entity_type_hints=hints_tuple,
        )

    subject = getattr(semantic_query, "entity_a", None)
    relation = getattr(semantic_query, "relation_intent", None)
    query_type = getattr(semantic_query, "query_type", None)

    answer_entities: list[AnswerEntity] = []
    surfaced: list[tuple[str, str, str]] = []
    if subject:
        answer_entities.append(AnswerEntity(subject, "answer_subject"))
    for canonical in answer_text_entities:
        if canonical == subject or any(e.canonical == canonical for e in answer_entities):
            continue
        if relation and subject and answer.decision == "answer":
            answer_entities.append(AnswerEntity(canonical, "answer_object", relation, subject))
            surfaced.append((subject, relation, canonical))
        else:
            answer_entities.append(AnswerEntity(canonical, "answer_object"))

    # Topic rule: an explicit resolver op wins; otherwise definition-style
    # questions ("tell me about X") set the topic; relation questions keep it.
    topic_op: tuple[str, ...] = ("keep",)
    if directives_topic and directives_topic[0] == "set":
        topic_op = directives_topic
    elif query_type in _DEFINITION_QUERY_TYPES and subject:
        topic_op = ("set", subject)

    return TurnRecord(
        question=question,
        user_named=user_named,
        answer_decision=answer.decision,
        answer_entities=tuple(answer_entities),
        surfaced_relations=tuple(surfaced),
        topic_op=topic_op,
        question_subject=subject,
        relation_intent=relation,
        resolved_referents=resolved_referents,
        entity_type_hints=hints_tuple,
    )


def unresolved_answer_text(resolved: ResolvedQuestion) -> str:
    """Human-readable audit payload for an unresolved dialogue reference,
    including the candidate scores so the refusal doubles as a
    disambiguation prompt."""

    for res in resolved.slots:
        if res.outcome in ("ambiguous", "no_candidate"):
            reference = res.slot.surface or "<omitted subject>"
            if res.candidates:
                listed = ", ".join(f"{c.canonical} ({c.total})" for c in res.candidates)
                return (
                    f"unresolved_dialogue_reference: could not determine what "
                    f"{reference!r} refers to — candidates: {listed}"
                )
            return (
                f"unresolved_dialogue_reference: could not determine what "
                f"{reference!r} refers to — no active dialogue candidates"
            )
    return "unresolved_dialogue_reference"


def _dedupe(items) -> tuple[str, ...]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return tuple(out)
