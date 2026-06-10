from dataclasses import dataclass, field
from collections import defaultdict
from .relations import Relation

LIFECYCLE_RELS: tuple[str, ...] = ("produces_result", "contains", "grows_into")

# Never use member_of edges themselves as grouping patterns (would recurse).
_SKIP_REL_TYPES: frozenset[str] = frozenset({"member_of"})


@dataclass
class Concept:
    id: str
    kind: str                    # "relation_group" | "lifecycle_group"
    members: list[str]
    common_relations: list[str]  # human-readable pattern labels
    confidence: float

    def __repr__(self) -> str:
        return (
            f"Concept(id={self.id!r}, kind={self.kind!r}, "
            f"members={self.members}, conf={self.confidence:.2f})"
        )


class ConceptEngine:
    """
    Forms named concepts from repeated relational patterns.

    Relation groups: entities that share the same outgoing (rel_type, target).
    Lifecycle groups: entities that are the "tree" slot in the same lifecycle
                      cycle type, partitioned by connector.
    """

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = relations

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def discover_concepts(self) -> list[Concept]:
        """Run all discovery methods and return every discovered concept."""
        return self.discover_relation_groups() + self.discover_lifecycle_groups()

    def discover_relation_groups(self) -> list[Concept]:
        """
        Group entities whose outgoing edges share the same (relation_type, target).
        Minimum group size: 2.
        """
        pattern_sources: dict[tuple[str, str], list[str]] = defaultdict(list)
        for rel in self._relations:
            if rel.relation_type in _SKIP_REL_TYPES:
                continue
            key = (rel.relation_type, rel.target)
            if rel.source not in pattern_sources[key]:
                pattern_sources[key].append(rel.source)

        concepts: list[Concept] = []
        for (rel_type, target), sources in sorted(pattern_sources.items()):
            if len(sources) < 2:
                continue
            concept_id = f"concept_rel_{rel_type}_{target}"
            confidence = min(1.0, (len(sources) - 1) / 4)
            concepts.append(Concept(
                id=concept_id,
                kind="relation_group",
                members=list(sources),
                common_relations=[f"{rel_type} -> {target}"],
                confidence=confidence,
            ))
        return concepts

    def discover_lifecycle_groups(self) -> list[Concept]:
        """
        Find all complete lifecycle cycles and group trees by shared connector.
        Returns one concept per distinct connector value with >= 2 members.
        """
        instances = self._find_lifecycle_instances()
        if len(instances) < 2:
            return []

        # Partition by connector (slot 2).
        by_connector: dict[str, list[str]] = defaultdict(list)
        for A, _B, C in instances:
            if A not in by_connector[C]:
                by_connector[C].append(A)

        r0, r1, r2 = LIFECYCLE_RELS
        concepts: list[Concept] = []
        for connector, trees in sorted(by_connector.items()):
            if len(trees) < 2:
                continue
            confidence = min(1.0, (len(trees) - 1) / 4)
            concepts.append(Concept(
                id=f"concept_lifecycle_{connector}",
                kind="lifecycle_group",
                members=list(trees),
                common_relations=[f"{r0} → {r1} → {r2} (connector: {connector})"],
                confidence=confidence,
            ))
        return concepts

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_lifecycle_instances(self) -> list[tuple[str, str, str]]:
        """Return (tree, fruit, connector) triples for every complete 3-cycle."""
        r0, r1, r2 = LIFECYCLE_RELS
        instances: list[tuple[str, str, str]] = []
        for rel0 in self._relations:
            if rel0.relation_type != r0:
                continue
            A, B = rel0.source, rel0.target
            for rel1 in self._relations:
                if rel1.source != B or rel1.relation_type != r1:
                    continue
                C = rel1.target
                if C == A:
                    continue
                for rel2 in self._relations:
                    if rel2.source == C and rel2.relation_type == r2 and rel2.target == A:
                        instances.append((A, B, C))
        return instances
