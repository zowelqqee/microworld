"""Explicit, replayable dialogue state for Microworld QA sessions.

``DialogueState`` is the *only* memory the dialogue layer has. It holds
canonical entity names and turn indices — pointers into trusted memory, never
facts — so nothing in it can be promoted into an overlay. It is mutated in
exactly one place (:meth:`DialogueState.commit`, fed a :class:`TurnRecord`)
and is fully reconstructible from the transcript via
:meth:`DialogueState.replay`; the benchmark asserts ``replay(records) ==
live_state`` after every session.

Decay is measured on a *confirmed-turn* clock (audit turns do not advance it),
preserving the v1 behavior where confirmed entities stay referable across any
number of intervening audit turns.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from worldpgt.dialogue import constants as C

IntroductionSource = str  # "user_named" | "answer_subject" | "answer_object"

TOPIC_KEEP: tuple[str, ...] = ("keep",)


@dataclass(frozen=True)
class EntityActivation:
    """Immutable per-entity dialogue facts. Salience is *computed* from these
    at read time (see :mod:`worldpgt.dialogue.salience`); scores are never
    stored, so every score is reproducible from visible state."""

    canonical: str
    entity_type: str | None
    introduced_turn: int
    last_mention_confirmed: int  # confirmed-turn clock value at last mention
    mention_count: int
    introduction_source: IntroductionSource
    dialogue_roles: tuple[tuple[str, str, int], ...] = ()  # (relation, anchor, turn)
    was_topic: bool = False

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "entity_type": self.entity_type,
            "introduced_turn": self.introduced_turn,
            "last_mention_confirmed": self.last_mention_confirmed,
            "mention_count": self.mention_count,
            "introduction_source": self.introduction_source,
            "dialogue_roles": [list(r) for r in self.dialogue_roles],
            "was_topic": self.was_topic,
        }

    @staticmethod
    def from_dict(data: dict) -> "EntityActivation":
        return EntityActivation(
            canonical=data["canonical"],
            entity_type=data.get("entity_type"),
            introduced_turn=data["introduced_turn"],
            last_mention_confirmed=data["last_mention_confirmed"],
            mention_count=data["mention_count"],
            introduction_source=data["introduction_source"],
            dialogue_roles=tuple(tuple(r) for r in data.get("dialogue_roles", [])),
            was_topic=data.get("was_topic", False),
        )


@dataclass(frozen=True)
class AnswerEntity:
    """One entity surfaced by an answer, with its dialogue role.

    ``role_relation``/``role_anchor`` record *how* the entity entered the
    dialogue: an entity answering ``founded_by(SpaceX)`` gets role
    ``("founded_by", "SpaceX")``, which later lets "the founder" resolve to it
    without a graph read.
    """

    canonical: str
    source: IntroductionSource  # "answer_subject" | "answer_object"
    role_relation: str | None = None
    role_anchor: str | None = None
    entity_type_hint: str | None = None  # session-only hint (e.g. web search)


@dataclass(frozen=True)
class TurnRecord:
    """The sole input to :meth:`DialogueState.commit` — everything one turn is
    allowed to change, captured explicitly so sessions replay exactly."""

    question: str
    user_named: tuple[str, ...] = ()  # canonical entities named by the user
    answer_decision: str = "answer"  # "answer" | "audit" | "no"
    answer_entities: tuple[AnswerEntity, ...] = ()
    surfaced_relations: tuple[tuple[str, str, str], ...] = ()  # (s, p, o)
    topic_op: tuple[str, ...] = TOPIC_KEEP  # ("keep",) | ("set", canonical)
    question_subject: str | None = None  # entity_a of the resolved query
    relation_intent: str | None = None
    resolved_referents: tuple[tuple[str, str], ...] = ()  # (ref_class, canonical)
    entity_type_hints: tuple[tuple[str, str], ...] = ()  # (canonical, type)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "user_named": list(self.user_named),
            "answer_decision": self.answer_decision,
            "answer_entities": [
                {
                    "canonical": e.canonical,
                    "source": e.source,
                    "role_relation": e.role_relation,
                    "role_anchor": e.role_anchor,
                    "entity_type_hint": e.entity_type_hint,
                }
                for e in self.answer_entities
            ],
            "surfaced_relations": [list(t) for t in self.surfaced_relations],
            "topic_op": list(self.topic_op),
            "question_subject": self.question_subject,
            "relation_intent": self.relation_intent,
            "resolved_referents": [list(t) for t in self.resolved_referents],
            "entity_type_hints": [list(t) for t in self.entity_type_hints],
        }

    @staticmethod
    def from_dict(data: dict) -> "TurnRecord":
        return TurnRecord(
            question=data["question"],
            user_named=tuple(data.get("user_named", [])),
            answer_decision=data.get("answer_decision", "answer"),
            answer_entities=tuple(
                AnswerEntity(
                    canonical=e["canonical"],
                    source=e["source"],
                    role_relation=e.get("role_relation"),
                    role_anchor=e.get("role_anchor"),
                    entity_type_hint=e.get("entity_type_hint"),
                )
                for e in data.get("answer_entities", [])
            ),
            surfaced_relations=tuple(tuple(t) for t in data.get("surfaced_relations", [])),
            topic_op=tuple(data.get("topic_op", TOPIC_KEEP)),
            question_subject=data.get("question_subject"),
            relation_intent=data.get("relation_intent"),
            resolved_referents=tuple(tuple(t) for t in data.get("resolved_referents", [])),
            entity_type_hints=tuple(tuple(t) for t in data.get("entity_type_hints", [])),
        )


@dataclass(frozen=True)
class LastQuestion:
    question: str
    subject: str | None
    relation_intent: str | None
    turn: int


@dataclass(frozen=True)
class LastAnswer:
    decision: str
    entities: tuple[str, ...]
    # The answer's *new information*: the first answer_object entity when one
    # exists, else the first entity. This is what focus-style references
    # ("that", "the other one") contrast against.
    primary: str | None
    relation_intent: str | None
    turn: int


@dataclass
class DialogueState:
    """Structured, inspectable per-session dialogue memory. No hidden state:
    ``to_dict()`` round-trips everything, and the /session debug endpoint
    serves it verbatim."""

    turn_counter: int = 0  # all turns, including audits
    confirmed_counter: int = 0  # non-audit turns only; drives decay/eviction
    entities: dict[str, EntityActivation] = field(default_factory=dict)
    active_topic: str | None = None
    previous_topic: str | None = None
    last_question: LastQuestion | None = None
    last_answer: LastAnswer | None = None
    mentioned_relations: tuple[tuple[str, str, str, int], ...] = ()
    last_referent: dict[str, str] = field(default_factory=dict)  # ref_class → canonical
    entity_type_hints: dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------ #

    def commit(self, record: TurnRecord) -> list[str]:
        """Apply one turn. Returns the canonical names evicted this turn (for
        the turn trace). Deterministic; the only mutator of DialogueState."""

        self.turn_counter += 1
        is_confirmed = record.answer_decision != "audit"
        if is_confirmed:
            self.confirmed_counter += 1

        for canonical, etype in record.entity_type_hints:
            self.entity_type_hints.setdefault(canonical, etype)

        # Question entities: the user named them, so they register even on an
        # audit turn (the user manifestly has them in mind).
        for canonical in record.user_named:
            self._touch(canonical, source="user_named", type_hint=None)

        if is_confirmed:
            for ent in record.answer_entities:
                roles = ()
                if ent.role_relation and ent.role_anchor:
                    roles = ((ent.role_relation, ent.role_anchor, self.turn_counter),)
                self._touch(
                    ent.canonical,
                    source=ent.source,
                    type_hint=ent.entity_type_hint,
                    new_roles=roles,
                )

            new_triples = tuple(
                (s, p, o, self.turn_counter) for s, p, o in record.surfaced_relations
            )
            if new_triples:
                self.mentioned_relations = self.mentioned_relations + new_triples

            primary = next(
                (e.canonical for e in record.answer_entities if e.source == "answer_object"),
                record.answer_entities[0].canonical if record.answer_entities else None,
            )
            self.last_answer = LastAnswer(
                decision=record.answer_decision,
                entities=tuple(e.canonical for e in record.answer_entities),
                primary=primary,
                relation_intent=record.relation_intent,
                turn=self.turn_counter,
            )

        self.last_question = LastQuestion(
            question=record.question,
            subject=record.question_subject,
            relation_intent=record.relation_intent,
            turn=self.turn_counter,
        )

        if record.topic_op and record.topic_op[0] == "set":
            target = record.topic_op[1]
            if target != self.active_topic:
                if self.active_topic is not None:
                    self.previous_topic = self.active_topic
                    old = self.entities.get(self.active_topic)
                    if old is not None:
                        self.entities[self.active_topic] = replace(old, was_topic=True)
                self.active_topic = target

        for ref_class, canonical in record.resolved_referents:
            self.last_referent[ref_class] = canonical

        return self._evict()

    # ------------------------------------------------------------------ #

    def _touch(
        self,
        canonical: str,
        *,
        source: IntroductionSource,
        type_hint: str | None,
        new_roles: tuple[tuple[str, str, int], ...] = (),
    ) -> None:
        if type_hint:
            self.entity_type_hints.setdefault(canonical, type_hint)
        existing = self.entities.get(canonical)
        if existing is None:
            self.entities[canonical] = EntityActivation(
                canonical=canonical,
                entity_type=self.entity_type_hints.get(canonical),
                introduced_turn=self.turn_counter,
                last_mention_confirmed=self.confirmed_counter,
                mention_count=1,
                introduction_source=source,
                dialogue_roles=new_roles,
            )
            return
        # user_named upgrades a weaker introduction source; roles accumulate.
        upgraded_source = (
            "user_named" if source == "user_named" else existing.introduction_source
        )
        merged_roles = existing.dialogue_roles + tuple(
            r for r in new_roles if r[:2] not in {er[:2] for er in existing.dialogue_roles}
        )
        etype = existing.entity_type or self.entity_type_hints.get(canonical)
        self.entities[canonical] = replace(
            existing,
            entity_type=etype,
            last_mention_confirmed=self.confirmed_counter,
            mention_count=existing.mention_count + 1,
            introduction_source=upgraded_source,
            dialogue_roles=merged_roles,
        )

    def _evict(self) -> list[str]:
        # Import here to avoid a module cycle (salience imports state types).
        from worldpgt.dialogue.salience import base_salience

        evicted: list[str] = []
        for canonical, act in list(self.entities.items()):
            if canonical == self.active_topic:
                continue
            idle = self.confirmed_counter - act.last_mention_confirmed
            if idle >= C.EVICT_AFTER_TURNS:
                evicted.append(canonical)
                del self.entities[canonical]

        if len(self.entities) > C.REGISTRY_CAP:
            overflow = len(self.entities) - C.REGISTRY_CAP
            # Lowest salience first; ties by older introduced_turn, then by
            # canonical purely to make the sort total (a true score+turn tie).
            ranked = sorted(
                (a for a in self.entities.values() if a.canonical != self.active_topic),
                key=lambda a: (
                    base_salience(a, self)[0],
                    -a.introduced_turn,
                    a.canonical,
                ),
                reverse=False,
            )
            for act in ranked[:overflow]:
                evicted.append(act.canonical)
                del self.entities[act.canonical]

        return evicted

    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        return {
            "turn_counter": self.turn_counter,
            "confirmed_counter": self.confirmed_counter,
            "entities": {k: v.to_dict() for k, v in sorted(self.entities.items())},
            "active_topic": self.active_topic,
            "previous_topic": self.previous_topic,
            "last_question": (
                {
                    "question": self.last_question.question,
                    "subject": self.last_question.subject,
                    "relation_intent": self.last_question.relation_intent,
                    "turn": self.last_question.turn,
                }
                if self.last_question
                else None
            ),
            "last_answer": (
                {
                    "decision": self.last_answer.decision,
                    "entities": list(self.last_answer.entities),
                    "primary": self.last_answer.primary,
                    "relation_intent": self.last_answer.relation_intent,
                    "turn": self.last_answer.turn,
                }
                if self.last_answer
                else None
            ),
            "mentioned_relations": [list(t) for t in self.mentioned_relations],
            "last_referent": dict(sorted(self.last_referent.items())),
            "entity_type_hints": dict(sorted(self.entity_type_hints.items())),
        }

    @staticmethod
    def from_dict(data: dict) -> "DialogueState":
        state = DialogueState(
            turn_counter=data["turn_counter"],
            confirmed_counter=data["confirmed_counter"],
            entities={
                k: EntityActivation.from_dict(v) for k, v in data.get("entities", {}).items()
            },
            active_topic=data.get("active_topic"),
            previous_topic=data.get("previous_topic"),
            mentioned_relations=tuple(
                tuple(t) for t in data.get("mentioned_relations", [])
            ),
            last_referent=dict(data.get("last_referent", {})),
            entity_type_hints=dict(data.get("entity_type_hints", {})),
        )
        lq = data.get("last_question")
        if lq:
            state.last_question = LastQuestion(
                question=lq["question"],
                subject=lq.get("subject"),
                relation_intent=lq.get("relation_intent"),
                turn=lq["turn"],
            )
        la = data.get("last_answer")
        if la:
            state.last_answer = LastAnswer(
                decision=la["decision"],
                entities=tuple(la.get("entities", [])),
                primary=la.get("primary"),
                relation_intent=la.get("relation_intent"),
                turn=la["turn"],
            )
        return state

    @staticmethod
    def replay(records: "list[TurnRecord] | tuple[TurnRecord, ...]") -> "DialogueState":
        """Reconstruct state by folding commits over a fresh state. The
        benchmark asserts this equals the live state after every session."""

        state = DialogueState()
        for record in records:
            state.commit(record)
        return state
