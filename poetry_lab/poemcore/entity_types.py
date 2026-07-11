"""Entity type classifier — the QA knowledge layer, re-grounded on the corpus.

Production MicroWorld types an entity from its Wikipedia *definition* text
(``worldpgt/knowledge/entity_type_classifier.py``: keyword rules, first match
wins, ``"other"`` as the non-blocking default). We have no definitions here —
only the novel — so the same classifier shape is kept and its *input* is
swapped: instead of matching keywords against a definition string, the rules
match against evidence counters that ``ingest`` already accumulates while
reading the prose.

    production: definition text  -> keyword rules -> canonical type
    here:       corpus evidence  -> ratio rules   -> canonical type

That is the same transfer discipline as every other port in this project: keep
the mechanism, re-domain the inputs, never import the source's data.

**Advisory, not a gate.** This is the one place the QA analogy is deliberately
*not* followed. QA uses the type to decide whether it may answer at all
(unsupported -> audit/refuse). Narrative planning uses it only to pick a
*role*: a place should not be forced into the grammatical subject slot of an
action, a person should. An ``"unknown"`` type changes nothing — the planner
falls back to exactly its previous behaviour. Nothing is ever refused.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

# Canonical types. Deliberately far fewer than production's eleven: the novel
# only distinguishes who acts, where things happen, and what gets acted upon.
CanonicalEntityType = str  # "person" | "place" | "object" | "unknown"

# Prepositions whose object is a location ("в Москве", "на Патриарших"). The
# locative-preposition test is the corpus analogue of production's `place`
# keyword row ("city", "region", "island", ...). English forms added
# alongside so an English-only corpus (e.g. Shakespeare) gets the same signal.
LOCATIVE_PREPOSITIONS = frozenset({
    "в", "во", "на", "из", "к", "ко", "по", "у", "под", "над",
    "in", "at", "on", "from", "near", "by", "within", "beneath", "above",
})

# An entity is a person when it repeatedly occupies the agent slot: the token
# immediately precedes a finite verb ("Маргарита летела"), or immediately
# follows a speech verb in inversion ("сказал Воланд"). This is the corpus
# analogue of production's `person` keyword row ("entrepreneur", "founder", ...).
# Matched via `str.startswith`, so English forms are listed individually
# rather than as a shared stem (English past tense isn't a suffix the way
# Russian's is: "say"/"said" don't share a prefix).
SPEECH_VERB_STEMS = ("сказал", "ответил", "спросил", "проговорил", "воскликнул",
                     "прошептал", "закричал", "промолвил", "отозвал", "заметил",
                     "say", "said", "answer", "answered", "ask", "asked",
                     "repl", "exclaim", "whisper", "cri", "shout", "speak",
                     "spoke", "declar", "murmur", "utter")

# Minimum observations before a ratio is trusted. Below this the evidence is
# noise and the type stays "unknown" — the same conservative bar the ingest
# proper-name detector already applies.
_MIN_EVIDENCE = 3
# Share of an entity's typed evidence that must agree for the type to be taken.
_DOMINANCE = 0.6


@dataclass
class EntityEvidence:
    """Per-entity counters accumulated while reading the corpus."""

    agent: Counter = field(default_factory=Counter)      # precedes a verb / follows a speech verb
    locative: Counter = field(default_factory=Counter)   # follows a locative preposition
    patient: Counter = field(default_factory=Counter)     # follows a verb (acted upon)

    def to_dict(self) -> dict:
        return {
            "agent": dict(self.agent),
            "locative": dict(self.locative),
            "patient": dict(self.patient),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EntityEvidence":
        ev = cls()
        ev.agent = Counter(data.get("agent", {}))
        ev.locative = Counter(data.get("locative", {}))
        ev.patient = Counter(data.get("patient", {}))
        return ev


def classify_entity_type(name: str, evidence: EntityEvidence) -> CanonicalEntityType:
    """Return the type that best matches *name*'s corpus evidence.

    Mirrors ``classify_entity_type(definition)`` in production: rules are tried
    in priority order, the first sufficiently-supported one wins, and the
    default is a non-committal type rather than a guess.
    """

    agent = evidence.agent.get(name, 0)
    locative = evidence.locative.get(name, 0)
    patient = evidence.patient.get(name, 0)
    total = agent + locative + patient
    if total < _MIN_EVIDENCE:
        return "unknown"

    # Order matters, as in production. Agency is checked first: a character can
    # be mentioned after "в" ("в Иване что-то дрогнуло") without being a place,
    # but a place is almost never the agent of a finite verb.
    if agent / total >= _DOMINANCE:
        return "person"
    if locative / total >= _DOMINANCE:
        return "place"
    if patient / total >= _DOMINANCE:
        return "object"
    # Mixed evidence: prefer agency when it is the plurality, since the planner's
    # only destructive mistake is putting a *place* in the subject slot.
    if agent > locative and agent > patient:
        return "person"
    if locative > agent and locative > patient:
        return "place"
    return "unknown"


def build_entity_types(
    names: list[str], evidence: EntityEvidence
) -> dict[str, CanonicalEntityType]:
    """Type every recognised proper name. Unknown types are kept explicitly so
    the planner can tell 'never seen' from 'seen, undecidable'."""

    return {name: classify_entity_type(name, evidence) for name in names}
