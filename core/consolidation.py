from dataclasses import dataclass, field
from .relations import Relation
from .abstractions import RELATION_PATTERNS

# Relation types that form meaningful causal chains.
CAUSAL_RELATION_TYPES = frozenset({
    "produces_result", "contains", "grows_into",
    "causes_effect", "enables", "requires", "reduces", "prevents",
})


@dataclass
class ConsolidatedPattern:
    name: str
    kind: str        # "relation_cluster" | "cycle" | "causal_chain"
    members: list[str]
    evidence: list[str]
    confidence: float

    def __repr__(self) -> str:
        return (
            f"ConsolidatedPattern(kind={self.kind!r}, name={self.name!r}, "
            f"conf={self.confidence:.2f}, members={len(self.members)})"
        )


class ConsolidationEngine:
    """
    Analyzes an accumulated relation graph and surfaces higher-level patterns:
    relation clusters, cycles, and multi-hop causal chains.
    """

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = relations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def consolidate(self) -> list[ConsolidatedPattern]:
        """Run all analyses and return every discovered pattern."""
        patterns: list[ConsolidatedPattern] = []
        patterns.extend(self.find_relation_clusters())
        patterns.extend(self.find_cycles())
        patterns.extend(self.find_causal_chains())
        return patterns

    def find_relation_clusters(self) -> list[ConsolidatedPattern]:
        """Group relations by type; one pattern per relation_type."""
        groups: dict[str, list[Relation]] = {}
        for rel in self._relations:
            groups.setdefault(rel.relation_type, []).append(rel)

        patterns = []
        for rel_type, rels in groups.items():
            label = RELATION_PATTERNS.get(rel_type, rel_type)
            members = [f"{r.source}->{r.target}" for r in rels]
            evidence = [e for r in rels for e in r.evidence]
            confidence = min(1.0, len(rels) / 5)
            patterns.append(ConsolidatedPattern(
                name=label,
                kind="relation_cluster",
                members=members,
                evidence=evidence,
                confidence=confidence,
            ))
        return patterns

    def find_cycles(self, max_depth: int = 4) -> list[ConsolidatedPattern]:
        """Detect simple cycles in the relation graph."""
        all_nodes = {n for r in self._relations for n in (r.source, r.target)}
        seen: set[frozenset] = set()
        cycles: list[list[str]] = []

        for start in sorted(all_nodes):
            self._dfs_cycles(start, start, [start], 0, max_depth, seen, cycles)

        patterns = []
        for nodes in cycles:
            display = " -> ".join(nodes) + " -> " + nodes[0]
            evidence = self._collect_cycle_evidence(nodes)
            patterns.append(ConsolidatedPattern(
                name=f"cycle: {display}",
                kind="cycle",
                members=list(nodes),
                evidence=evidence,
                confidence=0.8,
            ))
        return patterns

    def find_causal_chains(self, max_depth: int = 4) -> list[ConsolidatedPattern]:
        """Find maximal directed causal paths of length >= 2 relations (3+ nodes)."""
        all_nodes = {n for r in self._relations for n in (r.source, r.target)}
        seen: set[tuple] = set()
        patterns = []

        for start in sorted(all_nodes):
            raw_chains: list[tuple[str, ...]] = []
            self._dfs_chains(start, [start], {start}, 0, max_depth, raw_chains)
            for chain in raw_chains:
                if len(chain) < 3 or chain in seen:
                    continue
                seen.add(chain)
                display = " -> ".join(chain)
                evidence = self._collect_chain_evidence(chain)
                # longer chain = more connections verified = higher confidence
                confidence = min(1.0, (len(chain) - 1) / 4)
                patterns.append(ConsolidatedPattern(
                    name=f"chain: {display}",
                    kind="causal_chain",
                    members=list(chain),
                    evidence=evidence,
                    confidence=confidence,
                ))
        return patterns

    def score_relation_strength(self) -> dict[str, float]:
        """
        Return a confidence score per relation.
        More evidence entries → higher confidence (capped at 1.0).
        """
        scores: dict[str, float] = {}
        for rel in self._relations:
            key = f"{rel.source}--{rel.relation_type}-->{rel.target}"
            scores[key] = min(1.0, len(rel.evidence) / 3)
        return scores

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dfs_cycles(
        self,
        start: str,
        current: str,
        path: list[str],
        depth: int,
        max_depth: int,
        seen: set[frozenset],
        cycles: list[list[str]],
    ) -> None:
        if depth >= max_depth:
            return
        for rel in self._relations:
            if rel.source != current:
                continue
            if rel.target == start and len(path) >= 2:
                key = self._cycle_key(path)
                if key not in seen:
                    seen.add(key)
                    cycles.append(list(path))
            elif rel.target not in path:
                self._dfs_cycles(
                    start, rel.target, path + [rel.target],
                    depth + 1, max_depth, seen, cycles,
                )

    def _dfs_chains(
        self,
        current: str,
        path: list[str],
        visited: set[str],
        depth: int,
        max_depth: int,
        chains: list[tuple[str, ...]],
    ) -> None:
        outgoing = [
            r for r in self._relations
            if r.source == current and r.relation_type in CAUSAL_RELATION_TYPES
        ]
        if not outgoing or depth >= max_depth:
            if len(path) >= 3:
                chains.append(tuple(path))
            return
        for rel in outgoing:
            if rel.target in visited:
                # dead end for this branch; record what we have so far
                if len(path) >= 3:
                    chains.append(tuple(path))
                continue
            self._dfs_chains(
                rel.target,
                path + [rel.target],
                visited | {rel.target},
                depth + 1,
                max_depth,
                chains,
            )

    @staticmethod
    def _cycle_key(nodes: list[str]) -> frozenset:
        """Canonical key for a cycle regardless of starting point."""
        n = len(nodes)
        return frozenset((nodes[i], nodes[(i + 1) % n]) for i in range(n))

    def _collect_cycle_evidence(self, nodes: list[str]) -> list[str]:
        evidence: list[str] = []
        n = len(nodes)
        for i in range(n):
            src, tgt = nodes[i], nodes[(i + 1) % n]
            for rel in self._relations:
                if rel.source == src and rel.target == tgt:
                    evidence.extend(rel.evidence)
        return evidence

    def _collect_chain_evidence(self, chain: tuple[str, ...]) -> list[str]:
        evidence: list[str] = []
        for i in range(len(chain) - 1):
            src, tgt = chain[i], chain[i + 1]
            for rel in self._relations:
                if rel.source == src and rel.target == tgt:
                    evidence.extend(rel.evidence)
        return evidence
