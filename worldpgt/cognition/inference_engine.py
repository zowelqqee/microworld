"""Inference engine over the overlay fact graph.

This module owns the public result types (``InferredFact``,
``InferenceWorkspace``) and the ``run_inference`` entry point. The *rules*
themselves are no longer hardcoded here — they are data, interpreted by
``worldpgt.cognition.rule_interpreter`` (a small Datalog-style unification
engine). See ``worldpgt/knowledge/base_inference_rules.json`` for the rule set.

All inferred facts are ephemeral: they never persist to the overlay and are
always marked ``source="inferred"`` with an explicit proof ``chain``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

if TYPE_CHECKING:
    from worldpgt.cognition.rule_interpreter import InferenceRule


@dataclass(frozen=True)
class InferredFact:
    """One derived fact with explicit provenance."""

    subject: str
    predicate: str
    object: str
    source: Literal["inferred"] = "inferred"
    rule: str = ""
    chain: tuple[tuple[str, str, str], ...] = ()
    confidence: float = 0.7

    def as_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "source": self.source,
            "rule": self.rule,
            "chain": [list(step) for step in self.chain],
            "confidence": round(self.confidence, 4),
        }


@dataclass(frozen=True)
class InferenceWorkspace:
    """Ephemeral collection of inferred facts for one QA context."""

    facts: tuple[InferredFact, ...] = ()

    def for_subject(self, subject: str) -> tuple[InferredFact, ...]:
        """All inferred facts whose subject matches (case-insensitive)."""
        norm = subject.strip().lower()
        return tuple(f for f in self.facts if f.subject.strip().lower() == norm)

    def involving(self, entity: str) -> tuple[InferredFact, ...]:
        """All inferred facts where entity appears as subject or object."""
        norm = entity.strip().lower()
        return tuple(
            f for f in self.facts
            if f.subject.strip().lower() == norm or f.object.strip().lower() == norm
        )

    def by_rule(self, rule_id: str) -> tuple[InferredFact, ...]:
        """All inferred facts produced by a given rule id."""
        return tuple(f for f in self.facts if f.rule == rule_id)

    def without_facts(
        self, removed_keys: "frozenset[tuple[str, str, str]]"
    ) -> "InferenceWorkspace":
        """Drop cached facts whose proof chain cites a removed base triple.

        Cheap incremental substitute for re-running inference on a reduced
        overlay: each ``InferredFact`` already records the exact chain of base
        triples that justified it, so we can tell which ones a removal
        invalidates without redoing the rule search. This assumes that chain
        is a fact's only derivation path — true for this engine's
        single-derivation bookkeeping — so in the rare case an alternate,
        unrecorded derivation exists it may under-approximate (drop a fact
        that a full recompute would keep), but it never invents facts that
        shouldn't be there.
        """
        def _depends_on_removed(fact: InferredFact) -> bool:
            return any(
                (_norm(s), _norm(p), _norm(o)) in removed_keys
                for s, p, o in fact.chain
            )

        return InferenceWorkspace(
            facts=tuple(f for f in self.facts if not _depends_on_removed(f))
        )


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def run_inference(
    overlay_items: list[dict],
    rules: "list[InferenceRule] | None" = None,
) -> InferenceWorkspace:
    """Apply inference rules to the overlay; return an ephemeral workspace.

    When *rules* is None, the base rule set from
    ``knowledge/base_inference_rules.json`` is loaded. Callers may pass a custom
    list of ``InferenceRule`` to run a different rule set (e.g. one extra rule
    added purely as data, with no code change).
    """
    # Deferred import avoids a module-load cycle: rule_interpreter imports
    # InferredFact from this module.
    from worldpgt.cognition.rule_interpreter import apply_rules, load_base_rules

    if rules is None:
        rules = load_base_rules()
    return InferenceWorkspace(facts=tuple(apply_rules(overlay_items, rules)))
