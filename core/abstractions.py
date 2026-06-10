from dataclasses import dataclass, field
from .relations import Relation

# Maps relation_type -> human-readable pattern label.
# Replace or augment this dict to plug in embedding-based clustering later.
RELATION_PATTERNS: dict[str, str] = {
    "produces_result": "source -> result",
    "causes_effect": "cause -> effect",
    "contains": "whole -> part",
    "grows_into": "seed -> grown form",
    "requires": "dependent -> dependency",
    "prevents": "agent -> blocked outcome",
    "reduces": "agent -> diminished thing",
    "enables": "agent -> facilitated outcome",
}


@dataclass
class Abstraction:
    pattern: str          # e.g. "source -> result"
    relation_type: str
    instances: list[Relation] = field(default_factory=list)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{r.source}->{r.target}" for r in self.instances)
        return f"Abstraction({self.pattern!r}: [{pairs}])"


class AbstractionMiner:
    """Groups relations into named abstract patterns."""

    def mine(self, relations: list[Relation]) -> list[Abstraction]:
        buckets: dict[str, Abstraction] = {}
        for rel in relations:
            pattern = RELATION_PATTERNS.get(rel.relation_type, rel.relation_type)
            if rel.relation_type not in buckets:
                buckets[rel.relation_type] = Abstraction(
                    pattern=pattern,
                    relation_type=rel.relation_type,
                )
            buckets[rel.relation_type].instances.append(rel)
        return list(buckets.values())

    def analogies(self, abstractions: list[Abstraction]) -> list[str]:
        """Return human-readable analogy strings for each abstraction group."""
        lines = []
        for ab in abstractions:
            if len(ab.instances) < 2:
                continue
            for i, a in enumerate(ab.instances):
                for b in ab.instances[i + 1:]:
                    lines.append(
                        f"{a.source} -> {a.target} is analogous to "
                        f"{b.source} -> {b.target} "
                        f'because both match "{ab.pattern}"'
                    )
        return lines
