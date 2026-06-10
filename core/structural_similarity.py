from typing import Optional
from .relations import Relation

# member_of edges are derived from concept formation, not from observed structure.
# Excluding them keeps structural similarity grounded in the base relation graph
# and avoids a feedback loop (concepts → similarity → concepts).
_SKIP_REL_TYPES: frozenset[str] = frozenset({"member_of"})


class StructuralSimilarityEngine:
    """
    Discovers similarity between entities purely from graph structure — no ML,
    no embeddings, no oracle.  Two entities are similar when they play the same
    relational role (same relation types, same neighbours).

    An entity's profile is a set of structural feature strings:
      out:<rel_type>              — has an outgoing edge of this type
      out:<rel_type>:<target>     — points to this specific neighbour
      in:<rel_type>               — has an incoming edge of this type
      in:<rel_type>:<source>      — is pointed to by this specific neighbour

    Similarity is the Jaccard overlap of two profiles.
    """

    def __init__(
        self,
        relations: list[Relation],
        similarity_graph=None,
    ) -> None:
        self._relations = relations
        self._similarity_graph = similarity_graph
        self._profiles: dict[str, set[str]] = {}
        self._build_profiles()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def entity_profile(self, entity: str) -> set[str]:
        """Return the structural feature set for *entity* (empty if unknown)."""
        return set(self._profiles.get(entity, set()))

    def similarity(self, a: str, b: str) -> float:
        """Jaccard similarity of the two entity profiles; 0.0 if either is empty."""
        pa = self._profiles.get(a, set())
        pb = self._profiles.get(b, set())
        if not pa or not pb:
            return 0.0
        intersection = len(pa & pb)
        union = len(pa | pb)
        return intersection / union if union else 0.0

    def discover_similarities(
        self, min_score: float = 0.5
    ) -> list[tuple[str, str, float]]:
        """
        Compute Jaccard similarity for every unordered pair of entities and
        return those scoring >= *min_score*, sorted by score (desc) then name.
        """
        entities = sorted(self._profiles.keys())
        results: list[tuple[str, str, float]] = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                score = self.similarity(a, b)
                if score >= min_score and score > 0.0:
                    results.append((a, b, score))
        results.sort(key=lambda t: (-t[2], t[0], t[1]))
        return results

    def add_discovered_similarities(self, min_score: float = 0.5) -> int:
        """
        Discover structural similarities and register them in the attached
        SimilarityGraph.  Returns the number of pairs added.
        Raises if no SimilarityGraph was provided.
        """
        if self._similarity_graph is None:
            raise ValueError(
                "add_discovered_similarities requires a SimilarityGraph; "
                "construct the engine with similarity_graph=..."
            )
        discovered = self.discover_similarities(min_score=min_score)
        for a, b, score in discovered:
            self._similarity_graph.add_similarity(a, b, score)
        return len(discovered)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_profiles(self) -> None:
        for rel in self._relations:
            if rel.relation_type in _SKIP_REL_TYPES:
                continue
            rt, src, tgt = rel.relation_type, rel.source, rel.target
            # source-side (outgoing) features
            src_profile = self._profiles.setdefault(src, set())
            src_profile.add(f"out:{rt}")
            src_profile.add(f"out:{rt}:{tgt}")
            # target-side (incoming) features
            tgt_profile = self._profiles.setdefault(tgt, set())
            tgt_profile.add(f"in:{rt}")
            tgt_profile.add(f"in:{rt}:{src}")
