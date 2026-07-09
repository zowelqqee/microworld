"""Core dataclasses for the reasoning layer (explanatory chains, graph
patterns, counterfactual traces).

All three capabilities are *structural*: they never invent facts. Every step
in every output is either a verified overlay fact, an explicit inference-rule
application, or a measured observation about the graph itself.

Deterministic. No ML. No network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

# ---------------------------------------------------------------------------
# Part 1 — Explanatory chains
# ---------------------------------------------------------------------------

ExplanationStepKind = Literal[
    "fact",      # a verified overlay fact (direct triple)
    "rule",      # an inference-rule application connecting prior steps
    "pattern",   # a discovered graph pattern used as class-level context
    "note",      # an honest meta remark (e.g. where a partial chain stopped)
]

# Decision mirrors the three-outcome philosophy, with "partial" as the honest
# middle ground the spec requires: the chain did not close, and we say so.
ExplanationDecision = Literal["answer", "partial", "audit"]

FactStatus = Literal["direct", "inferred", "absent"]


@dataclass(frozen=True)
class ExplanationStep:
    """One verified link in an explanation chain."""

    kind: ExplanationStepKind
    subject: str = ""
    predicate: str = ""
    object: str = ""
    text: str = ""
    rule: str = ""          # rule_id when kind == "rule"
    pattern_id: str = ""    # pattern id when kind == "pattern"
    confidence: float = 1.0

    def display(self) -> str:
        if self.text:
            return self.text
        return f"{self.subject} | {self.predicate} | {self.object}"

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "text": self.text,
            "rule": self.rule,
            "pattern_id": self.pattern_id,
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ExplanationChain:
    """A deterministic explanation of why a fact exists / makes sense.

    ``decision`` semantics:
      answer  — the chain closed: every link verified, context loops back to
                the subject.
      partial — the fact itself is verified but the explanatory chain did not
                close; ``frontier`` lists the nodes exploration reached.
      audit   — the fact is not in the graph; nothing to explain.
    """

    subject: str
    predicate: str
    object: str
    fact_status: FactStatus
    decision: ExplanationDecision
    steps: list[ExplanationStep] = field(default_factory=list)
    frontier: list[str] = field(default_factory=list)
    audit_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "fact_status": self.fact_status,
            "decision": self.decision,
            "steps": [s.to_dict() for s in self.steps],
            "frontier": list(self.frontier),
            "audit_reason": self.audit_reason,
        }


# ---------------------------------------------------------------------------
# Part 2 — Graph patterns
# ---------------------------------------------------------------------------

GraphPatternKind = Literal[
    "class_implication",  # entities that are C tend to have predicate p
    "cooccurrence",       # entities with (p1 → something that is_a D) tend to have p2
]


@dataclass
class PatternCounterExample:
    """An entity that satisfies a pattern's condition but not its consequent."""

    entity: str
    note: str

    def to_dict(self) -> dict:
        return {"entity": self.entity, "note": self.note}

    @staticmethod
    def from_dict(data: dict) -> "PatternCounterExample":
        return PatternCounterExample(
            entity=str(data.get("entity") or ""),
            note=str(data.get("note") or ""),
        )


@dataclass
class GraphPattern:
    """An unnamed structural regularity observed in the fact graph.

    Not a fact — an *observation about the graph*, with explicit evidence:
      supporting_evidence — concrete verified triples that instantiate it.
      confidence          — share of condition-matching entities that also
                            satisfy the consequent.
      counter_examples    — entities that match the condition but violate the
                            consequent.
    """

    pattern_id: str
    kind: GraphPatternKind
    description: str
    condition: dict            # {"class": C} | {"predicate": p1, "object_class": D}
    consequent: dict           # {"predicate": p2, "object_class": D2 | None}
    support: int               # entities matching condition AND consequent
    population: int            # entities matching condition
    confidence: float          # support / population
    matched_entities: list[str] = field(default_factory=list)
    counter_examples: list[PatternCounterExample] = field(default_factory=list)
    supporting_evidence: list[list[str]] = field(default_factory=list)  # [s, p, o] triples
    as_of: str = ""            # optional timestamp injected by the nightly runner

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "kind": self.kind,
            "description": self.description,
            "condition": dict(self.condition),
            "consequent": dict(self.consequent),
            "support": self.support,
            "population": self.population,
            "confidence": round(self.confidence, 4),
            "matched_entities": list(self.matched_entities),
            "counter_examples": [c.to_dict() for c in self.counter_examples],
            "supporting_evidence": [list(t) for t in self.supporting_evidence],
            "as_of": self.as_of,
        }

    @staticmethod
    def from_dict(data: dict) -> "GraphPattern":
        return GraphPattern(
            pattern_id=str(data.get("pattern_id") or ""),
            kind=data.get("kind", "class_implication"),
            description=str(data.get("description") or ""),
            condition=dict(data.get("condition") or {}),
            consequent=dict(data.get("consequent") or {}),
            support=int(data.get("support") or 0),
            population=int(data.get("population") or 0),
            confidence=float(data.get("confidence") or 0.0),
            matched_entities=[str(e) for e in (data.get("matched_entities") or [])],
            counter_examples=[
                PatternCounterExample.from_dict(c)
                for c in (data.get("counter_examples") or [])
            ],
            supporting_evidence=[
                [str(x) for x in t] for t in (data.get("supporting_evidence") or [])
            ],
            as_of=str(data.get("as_of") or ""),
        )


# ---------------------------------------------------------------------------
# Part 3 — Counterfactual traces
# ---------------------------------------------------------------------------

CounterfactualDecision = Literal["analysis", "audit"]


@dataclass
class RemovedFact:
    """A concrete overlay fact hypothetically removed from the graph."""

    subject: str
    predicate: str
    object: str

    def display(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
        }


@dataclass
class LostInference:
    """An inferred fact that stops being derivable without the removed fact."""

    subject: str
    predicate: str
    object: str
    rule: str
    chain: list[list[str]] = field(default_factory=list)
    removed_links: list[list[str]] = field(default_factory=list)

    def display(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object} (rule: {self.rule})"

    def to_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "rule": self.rule,
            "chain": [list(t) for t in self.chain],
            "removed_links": [list(t) for t in self.removed_links],
        }


@dataclass
class AffectedPattern:
    """A graph pattern whose evidence includes a removed fact."""

    pattern_id: str
    description: str
    old_confidence: float
    new_confidence: Optional[float]  # None → pattern no longer meets thresholds
    old_support: int
    new_support: int
    removed_evidence: list[list[str]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "description": self.description,
            "old_confidence": round(self.old_confidence, 4),
            "new_confidence": (
                round(self.new_confidence, 4)
                if self.new_confidence is not None
                else None
            ),
            "old_support": self.old_support,
            "new_support": self.new_support,
            "removed_evidence": [list(t) for t in self.removed_evidence],
        }


@dataclass
class CounterfactualTrace:
    """Structural analysis: what in the world model rests on a given fact.

    Never speculates about an alternative world — only reports which verified
    facts would be removed, which inferences stop being derivable, and which
    observed patterns lose evidence.
    """

    target_subject: str
    target_predicate: Optional[str]
    target_object: Optional[str]
    decision: CounterfactualDecision
    removed_facts: list[RemovedFact] = field(default_factory=list)
    lost_inferences: list[LostInference] = field(default_factory=list)
    affected_patterns: list[AffectedPattern] = field(default_factory=list)
    dependent_entities: list[str] = field(default_factory=list)
    audit_reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "target_subject": self.target_subject,
            "target_predicate": self.target_predicate,
            "target_object": self.target_object,
            "decision": self.decision,
            "removed_facts": [f.to_dict() for f in self.removed_facts],
            "lost_inferences": [f.to_dict() for f in self.lost_inferences],
            "affected_patterns": [p.to_dict() for p in self.affected_patterns],
            "dependent_entities": list(self.dependent_entities),
            "audit_reason": self.audit_reason,
        }
