"""Read-only fact graph shared by the reasoning layer.

Thin, deterministic wrapper around ``rule_interpreter.normalize_overlay`` —
the same normalization the inference engine uses, so explanations, patterns,
and counterfactuals all see exactly the facts the rest of the system reasons
over (including definition-lifted ``is_a`` / ``owned_by`` facts and inverse
twins such as ``founded_by``).

Deterministic. No ML. No network.
"""

from __future__ import annotations

from worldpgt.cognition.rule_interpreter import normalize_overlay


def norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


class FactGraph:
    """Indexed, order-stable view over the normalized overlay triples."""

    def __init__(self, overlay_items: list[dict]) -> None:
        self._normalized = normalize_overlay(overlay_items)
        self._out: dict[str, list] = {}
        self._in: dict[str, list] = {}
        self._display: dict[str, str] = {}
        for fact in self._normalized.facts:
            ns, no = norm(fact.subject), norm(fact.object)
            self._out.setdefault(ns, []).append(fact)
            self._in.setdefault(no, []).append(fact)
            self._display.setdefault(ns, fact.subject)
            self._display.setdefault(no, fact.object)
        # Deterministic per-node ordering regardless of overlay item order.
        for bucket in (self._out, self._in):
            for key in bucket:
                bucket[key].sort(key=lambda f: (f.predicate, norm(f.object), norm(f.subject)))

    # -- accessors ---------------------------------------------------------

    @property
    def facts(self) -> tuple:
        return self._normalized.facts

    @property
    def entity_types(self) -> dict[str, str]:
        return self._normalized.entity_types

    def display(self, node: str) -> str:
        return self._display.get(norm(node), node)

    def is_known(self, node: str) -> bool:
        return norm(node) in self._display

    def outgoing(self, node: str) -> list:
        return list(self._out.get(norm(node), []))

    def incoming(self, node: str) -> list:
        return list(self._in.get(norm(node), []))

    def classes_of(self, node: str) -> list[str]:
        """Sorted ``is_a`` / ``type_of`` objects for a node."""
        out = sorted(
            {
                fact.object
                for fact in self._out.get(norm(node), [])
                if fact.predicate in ("is_a", "type_of")
            },
            key=norm,
        )
        return out

    def find(self, subject: str, predicate: str | None = None, obj: str | None = None) -> list:
        """All facts matching the given (subject, predicate?, object?) query."""
        results = []
        for fact in self._out.get(norm(subject), []):
            if predicate is not None and norm(fact.predicate) != norm(predicate):
                continue
            if obj is not None and norm(fact.object) != norm(obj):
                continue
            results.append(fact)
        return results

    def has_fact(self, subject: str, predicate: str, obj: str) -> bool:
        return bool(self.find(subject, predicate, obj))

    def node_count(self) -> int:
        return len(self._display)

    def fact_count(self) -> int:
        return len(self._normalized.facts)
