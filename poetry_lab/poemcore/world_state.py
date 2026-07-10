"""Deterministic temporal state for the narrative reasoning experiment.

This module deliberately has no dependency on ``worldpgt``.  It preserves the
useful shapes of production inference (facts with provenance, explicit
violations, replayable state) while adding the one dimension prose needs:
discrete time.  A transition never mutates its parent; rejected candidates
return the parent as their accepted state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal, Mapping


FactSource = Literal["asserted", "inferred", "retracted"]


def _norm(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def _fact_sort_key(fact: "StateFact") -> tuple[object, ...]:
    return (fact.t, _norm(fact.subject), _norm(fact.predicate), _norm(fact.object), fact.source, fact.rule)


@dataclass(frozen=True)
class StateFact:
    """A fact that holds at one discrete state index."""

    subject: str
    predicate: str
    object: str
    t: int
    source: FactSource = "asserted"
    rule: str = ""
    chain: tuple[tuple[str, str, str], ...] = ()

    @property
    def key(self) -> tuple[str, str]:
        return (_norm(self.subject), _norm(self.predicate))

    @property
    def triple(self) -> tuple[str, str, str]:
        return (self.subject, self.predicate, self.object)

    def at(self, t: int, *, source: FactSource | None = None) -> "StateFact":
        return replace(self, t=t, source=source or self.source)

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "t": self.t,
            "source": self.source,
            "rule": self.rule,
            "chain": [list(item) for item in self.chain],
        }


@dataclass(frozen=True)
class EntityState:
    """Small denormalized entity index for O(1) narrative safety checks."""

    name: str
    introduced_at: int | None = None
    entity_type: str = "unknown"
    last_seen_at: int | None = None
    last_action_at: int | None = None
    location: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "introduced_at": self.introduced_at,
            "entity_type": self.entity_type,
            "last_seen_at": self.last_seen_at,
            "last_action_at": self.last_action_at,
            "location": self.location,
        }


@dataclass(frozen=True)
class ProofStep:
    rule: str
    conclusion: tuple[str, str, str]
    premises: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "rule": self.rule,
            "conclusion": list(self.conclusion),
            "premises": [list(item) for item in self.premises],
        }


@dataclass(frozen=True)
class Violation:
    code: str
    message: str
    t: int
    facts: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "t": self.t,
            "facts": [list(item) for item in self.facts],
        }


@dataclass(frozen=True)
class StateDelta:
    """A proposed transition.  Assertions are stamped with ``t + 1`` on apply."""

    assertions: tuple[StateFact, ...] = ()
    retracts: frozenset[tuple[str, str]] = frozenset()
    remote_events: frozenset[tuple[str, str, str]] = frozenset()
    label: str = ""

    def to_dict(self) -> dict:
        return {
            "assertions": [item.to_dict() for item in sorted(self.assertions, key=_fact_sort_key)],
            "retracts": [list(item) for item in sorted(self.retracts)],
            "remote_events": [list(item) for item in sorted(self.remote_events)],
            "label": self.label,
        }


@dataclass(frozen=True)
class TransitionResult:
    """Audit record for one candidate evaluation."""

    accepted: bool
    state: "WorldState"
    asserted: tuple[StateFact, ...] = ()
    inferred: tuple[StateFact, ...] = ()
    retracted: tuple[StateFact, ...] = ()
    proof_steps: tuple[ProofStep, ...] = ()
    violations: tuple[Violation, ...] = ()

    def to_dict(self) -> dict:
        return {
            "accepted": self.accepted,
            "state": self.state.to_dict(),
            "asserted": [item.to_dict() for item in self.asserted],
            "inferred": [item.to_dict() for item in self.inferred],
            "retracted": [item.to_dict() for item in self.retracted],
            "proof_steps": [item.to_dict() for item in self.proof_steps],
            "violations": [item.to_dict() for item in self.violations],
        }


_STATIC_PREDICATES = frozenset({"introduced", "located_at", "co_located", "present", "is_a"})
_LOCATION_PREDICATE = "located_at"
_MOVE_PREDICATE = "moves_to"
_INTERACTION_PREDICATES = frozenset({"speaking_to", "questions", "asks", "answers", "meets", "sees"})


@dataclass(frozen=True)
class WorldState:
    """Copy-on-write state at one time index.

    ``asserted_facts`` and ``inferred_facts`` contain only facts holding now.
    ``retracted_facts`` and ``proof_history`` are append-only audit trails.
    """

    t: int = 0
    asserted_facts: frozenset[StateFact] = frozenset()
    inferred_facts: frozenset[StateFact] = frozenset()
    retracted_facts: tuple[StateFact, ...] = ()
    entities: Mapping[str, EntityState] = field(default_factory=dict)
    proof_history: tuple[ProofStep, ...] = ()
    transitions: tuple[StateDelta, ...] = ()

    @classmethod
    def from_initial_facts(cls, facts: tuple[StateFact, ...] | list[StateFact]) -> "WorldState":
        """Create a t=0 scene snapshot without treating placement as teleportation."""
        asserted = frozenset(item.at(0, source="asserted") for item in facts)
        state = cls(t=0, asserted_facts=asserted)
        return replace(state, entities=_entity_index(state, {}, (), ()))

    @property
    def facts(self) -> frozenset[StateFact]:
        return self.asserted_facts | self.inferred_facts

    def location_of(self, entity: str) -> str | None:
        record = self.entities.get(_norm(entity))
        return record.location if record else None

    def is_introduced(self, entity: str) -> bool:
        record = self.entities.get(_norm(entity))
        return bool(record and record.introduced_at is not None)

    def apply(self, delta: StateDelta) -> TransitionResult:
        """Evaluate a candidate and commit it only when it is valid."""
        next_t = self.t + 1
        asserted = tuple(sorted((item.at(next_t, source="asserted") for item in delta.assertions), key=_fact_sort_key))
        manual_retracts = {(_norm(subject), _norm(predicate)) for subject, predicate in delta.retracts}
        moves = [fact for fact in asserted if _norm(fact.predicate) == _MOVE_PREDICATE]
        location_assertions = [fact for fact in asserted if _norm(fact.predicate) == _LOCATION_PREDICATE]
        violations: list[Violation] = []
        proofs: list[ProofStep] = []
        inferred: list[StateFact] = []

        action_facts = [fact for fact in asserted if _norm(fact.predicate) not in _STATIC_PREDICATES]
        introduced_now = {_norm(fact.subject) for fact in asserted if _norm(fact.predicate) == "introduced"}
        for fact in action_facts:
            if _norm(fact.subject) not in introduced_now and not self.is_introduced(fact.subject):
                violations.append(_violation("unintroduced_entity", next_t, fact, f"{fact.subject} acts before introduction"))

        move_destinations: dict[str, set[str]] = {}
        for move in moves:
            move_destinations.setdefault(_norm(move.subject), set()).add(_norm(move.object))
            conclusion = (move.subject, _LOCATION_PREDICATE, move.object)
            proofs.append(ProofStep("moves_to_implies_location", conclusion, (move.triple,)))
            inferred.append(StateFact(*conclusion, t=next_t, source="inferred", rule="moves_to_implies_location", chain=(move.triple,)))

        explicit_destinations: dict[str, set[str]] = {}
        for fact in location_assertions:
            explicit_destinations.setdefault(_norm(fact.subject), set()).add(_norm(fact.object))
            old = self.location_of(fact.subject)
            if old and _norm(old) != _norm(fact.object) and _norm(fact.subject) not in move_destinations:
                violations.append(_violation("location_change_without_move", next_t, fact, f"{fact.subject} changes location without moves_to"))

        all_destinations: dict[str, set[str]] = {}
        for source in (move_destinations, explicit_destinations):
            for subject, places in source.items():
                all_destinations.setdefault(subject, set()).update(places)
        for subject, places in all_destinations.items():
            if len(places) > 1:
                facts = tuple(fact.triple for fact in (*moves, *location_assertions) if _norm(fact.subject) == subject)
                violations.append(Violation("bilocation", f"{subject} has multiple locations at t={next_t}", next_t, facts))

        for fact in asserted:
            if _norm(fact.predicate) not in _INTERACTION_PREDICATES:
                continue
            pair = (_norm(fact.subject), _norm(fact.predicate), _norm(fact.object))
            left, right = self.location_of(fact.subject), self.location_of(fact.object)
            if left and right and _norm(left) != _norm(right) and pair not in delta.remote_events:
                violations.append(_violation("participants_not_colocated", next_t, fact, f"{fact.subject} and {fact.object} interact from different locations"))
            if _norm(fact.predicate) == "speaking_to":
                conclusion = (fact.subject, "co_located", fact.object)
                proofs.append(ProofStep("speaking_to_implies_co_located", conclusion, (fact.triple,)))
                inferred.append(StateFact(*conclusion, t=next_t, source="inferred", rule="speaking_to_implies_co_located", chain=(fact.triple,)))

        retraction_keys = set(manual_retracts)
        for subject in move_destinations:
            retraction_keys.add((subject, _LOCATION_PREDICATE))
        retained: list[StateFact] = []
        retracted: list[StateFact] = []
        for fact in self.facts:
            if fact.key in retraction_keys:
                retracted.append(fact.at(next_t, source="retracted"))
            else:
                retained.append(fact.at(next_t))

        # A direct location assertion is retained only when valid; a move's
        # derived location is the sole current location after its retraction.
        asserted_current = [fact for fact in retained if fact.source == "asserted"] + list(asserted)
        inferred_current = [fact for fact in retained if fact.source == "inferred"] + inferred
        candidate = WorldState(
            t=next_t,
            asserted_facts=frozenset(asserted_current),
            inferred_facts=frozenset(inferred_current),
            retracted_facts=self.retracted_facts + tuple(sorted(retracted, key=_fact_sort_key)),
            proof_history=self.proof_history + tuple(proofs),
            transitions=self.transitions + (delta,),
        )
        candidate = replace(candidate, entities=_entity_index(candidate, self.entities, asserted, action_facts))
        accepted = not violations
        return TransitionResult(
            accepted=accepted,
            state=candidate if accepted else self,
            asserted=asserted,
            inferred=tuple(sorted(inferred, key=_fact_sort_key)),
            retracted=tuple(sorted(retracted, key=_fact_sort_key)),
            proof_steps=tuple(proofs),
            violations=tuple(sorted(violations, key=lambda item: (item.code, item.message))),
        )

    @classmethod
    def replay(
        cls,
        transitions: tuple[StateDelta, ...] | list[StateDelta],
        *,
        initial: "WorldState | None" = None,
    ) -> "WorldState":
        state = initial or cls()
        for delta in transitions:
            result = state.apply(delta)
            if not result.accepted:
                raise ValueError(f"replay rejected transition: {result.violations[0].code}")
            state = result.state
        return state

    def to_dict(self) -> dict:
        return {
            "t": self.t,
            "asserted_facts": [item.to_dict() for item in sorted(self.asserted_facts, key=_fact_sort_key)],
            "inferred_facts": [item.to_dict() for item in sorted(self.inferred_facts, key=_fact_sort_key)],
            "retracted_facts": [item.to_dict() for item in sorted(self.retracted_facts, key=_fact_sort_key)],
            "entities": {key: value.to_dict() for key, value in sorted(self.entities.items())},
            "proof_history": [item.to_dict() for item in self.proof_history],
            "transitions": [item.to_dict() for item in self.transitions],
        }


def _violation(code: str, t: int, fact: StateFact, message: str) -> Violation:
    return Violation(code=code, message=message, t=t, facts=(fact.triple,))


def _entity_index(
    state: WorldState,
    previous: Mapping[str, EntityState],
    asserted_now: tuple[StateFact, ...],
    action_facts: list[StateFact] | tuple[StateFact, ...],
) -> dict[str, EntityState]:
    entities = dict(previous)
    facts = sorted(state.facts, key=_fact_sort_key)
    introduced = { _norm(fact.subject): fact for fact in facts if _norm(fact.predicate) == "introduced" }
    locations = { _norm(fact.subject): fact.object for fact in facts if _norm(fact.predicate) == _LOCATION_PREDICATE }
    touched = {_norm(fact.subject) for fact in asserted_now}
    touched.update(_norm(fact.object) for fact in asserted_now if _norm(fact.predicate) in _INTERACTION_PREDICATES)
    actors = {_norm(fact.subject) for fact in action_facts}
    for name in sorted(set(entities) | set(introduced) | set(locations) | touched):
        old = entities.get(name, EntityState(name=name))
        intro = introduced.get(name)
        entities[name] = EntityState(
            name=old.name if old.name else (intro.subject if intro else name),
            introduced_at=old.introduced_at if old.introduced_at is not None else (intro.t if intro else None),
            entity_type=old.entity_type,
            last_seen_at=state.t if name in touched else old.last_seen_at,
            last_action_at=state.t if name in actors else old.last_action_at,
            location=locations.get(name),
        )
    return entities
