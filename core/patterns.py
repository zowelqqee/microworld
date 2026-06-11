"""
Pattern discovery engine for finding frequent relation chains.

Answers "what structures actually exist in this graph" before attempting
any link prediction — this is a pure exploratory analysis tool.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .relations import Relation


@dataclass
class Pattern:
    """A frequently occurring relation chain discovered in the graph."""

    relations: tuple[str, ...]         # e.g. ("is_a", "capable_of")
    count:     int                     # unique chain instances found
    support:   float                   # count / total_relations
    examples:  list[tuple[str, ...]] = field(default_factory=list)  # up to 5

    def __repr__(self) -> str:
        chain = " -> ".join(self.relations)
        return f"Pattern({chain!r}, count={self.count}, support={self.support:.4f})"


class PatternDiscoveryEngine:
    """
    Discovers frequent relation bigrams and trigrams from any relation graph.

    No lifecycle assumptions — works on ConceptNet, synthetic worlds, or
    any set of (source, relation_type, target) triples.
    """

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = list(relations)
        # outgoing[node] = [(relation_type, target), ...]
        self._outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in self._relations:
            self._outgoing[r.source].append((r.relation_type, r.target))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_relation_bigrams(self, min_count: int = 5) -> list[Pattern]:
        """
        Find all 2-hop chains  A --r1--> B --r2--> C  and aggregate by
        (r1, r2).  Returns Pattern objects with count >= min_count,
        sorted by count descending.
        """
        counts: dict[tuple[str, str], int] = defaultdict(int)
        examples: dict[tuple[str, str], list[tuple[str, ...]]] = defaultdict(list)

        for r1 in self._relations:
            for r2_type, r2_tgt in self._outgoing[r1.target]:
                key = (r1.relation_type, r2_type)
                counts[key] += 1
                if len(examples[key]) < 5:
                    examples[key].append((r1.source, r1.target, r2_tgt))

        return self._build(counts, examples, min_count)

    def discover_relation_trigrams(self, min_count: int = 5) -> list[Pattern]:
        """
        Find all 3-hop chains  A --r1--> B --r2--> C --r3--> D  and
        aggregate by (r1, r2, r3).  Returns Pattern objects with
        count >= min_count, sorted by count descending.
        """
        counts: dict[tuple[str, str, str], int] = defaultdict(int)
        examples: dict[tuple[str, str, str], list[tuple[str, ...]]] = defaultdict(list)

        for r1 in self._relations:
            for r2_type, mid in self._outgoing[r1.target]:
                for r3_type, end in self._outgoing[mid]:
                    key = (r1.relation_type, r2_type, r3_type)
                    counts[key] += 1
                    if len(examples[key]) < 5:
                        examples[key].append((r1.source, r1.target, mid, end))

        return self._build(counts, examples, min_count)

    def discover_patterns(self, min_count: int = 5) -> dict[str, list[Pattern]]:
        """Run both bigram and trigram discovery; return under shared keys."""
        return {
            "bigrams":  self.discover_relation_bigrams(min_count=min_count),
            "trigrams": self.discover_relation_trigrams(min_count=min_count),
        }

    def compute_coverage(self, bigrams: list[Pattern]) -> float:
        """
        Fraction of relation instances that participate in at least one
        discovered bigram chain (as left OR right edge).

        Pass the output of discover_relation_bigrams() — only patterns
        meeting the original min_count threshold count as 'discovered'.
        """
        if not self._relations:
            return 0.0
        valid: set[tuple[str, str]] = {p.relations for p in bigrams}
        participating: set[tuple[str, str, str]] = set()
        for r1 in self._relations:
            for r2_type, r2_tgt in self._outgoing[r1.target]:
                if (r1.relation_type, r2_type) in valid:
                    participating.add((r1.source, r1.relation_type, r1.target))
                    participating.add((r1.target, r2_type, r2_tgt))
        return len(participating) / len(self._relations)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build(
        self,
        counts: dict,
        examples: dict,
        min_count: int,
    ) -> list[Pattern]:
        total = len(self._relations)
        results = []
        for key, count in counts.items():
            if count < min_count:
                continue
            results.append(
                Pattern(
                    relations=key,
                    count=count,
                    support=round(count / total, 6) if total > 0 else 0.0,
                    examples=list(examples[key]),
                )
            )
        return sorted(results, key=lambda p: -p.count)
