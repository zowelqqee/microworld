from typing import Optional
from .objects import WorldObject
from .events import Event
from .relations import Relation
from .parser import TinyParser
from .normalizer import EntityNormalizer
from .similarity import SimilarityGraph


class World:
    """Stores objects, events, and relations observed so far."""

    def __init__(self, normalizer: Optional[EntityNormalizer] = None) -> None:
        self._parser = TinyParser()
        self._normalizer = normalizer or EntityNormalizer()
        self._objects: dict[str, WorldObject] = {}
        self._events: list[Event] = []
        self._relations: list[Relation] = []
        self._similarity_graph = SimilarityGraph(normalizer=self._normalizer)
        self._concepts: list = []  # populated by discover_concepts()

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def observe(self, text: str) -> Optional[Event]:
        event = self._parser.parse(text)
        if event is None:
            return None
        # Normalize to canonical names; raw_text is preserved as-is
        event.subject = self._normalizer.normalize(event.subject)
        event.object = self._normalizer.normalize(event.object)
        self.add_event(event)
        return event

    def add_event(self, event: Event) -> None:
        self._events.append(event)
        self._ensure_object(event.subject)
        self._ensure_object(event.object)

        rel_type = self._parser.relation_type(event.verb)
        if rel_type:
            self._upsert_relation(event.subject, rel_type, event.object, event.raw_text)

    def add_relation(self, relation: Relation) -> None:
        self._relations.append(relation)

    # ------------------------------------------------------------------
    # Queries — all name inputs are normalized before lookup
    # ------------------------------------------------------------------

    def get_object(self, name: str) -> Optional[WorldObject]:
        return self._objects.get(self._normalizer.normalize(name))

    def get_relations(self) -> list[Relation]:
        return list(self._relations)

    def get_objects(self) -> list[WorldObject]:
        return list(self._objects.values())

    def explain_object(self, name: str) -> str:
        canonical = self._normalizer.normalize(name)
        obj = self._objects.get(canonical)
        if obj is None:
            return f"Unknown object: {name!r}"
        lines = [f"Object: {obj.name}"]
        if obj.kind:
            lines.append(f"  kind: {obj.kind}")
        outgoing = [r for r in self._relations if r.source == canonical]
        incoming = [r for r in self._relations if r.target == canonical]
        for r in outgoing:
            lines.append(f"  --{r.relation_type}--> {r.target}")
        for r in incoming:
            lines.append(f"  <--{r.relation_type}-- {r.source}")
        aliases = self._normalizer.get_aliases(canonical)
        if aliases:
            lines.append(f"  aliases: {', '.join(aliases)}")
        return "\n".join(lines)

    def explain_relation(self, source: str, target: str) -> str:
        src = self._normalizer.normalize(source)
        tgt = self._normalizer.normalize(target)
        matches = [r for r in self._relations if r.source == src and r.target == tgt]
        if not matches:
            return f"No relation found between {source!r} and {target!r}"
        lines = []
        for r in matches:
            evidence = "; ".join(r.evidence) if r.evidence else "none"
            lines.append(
                f"{r.source} --{r.relation_type}--> {r.target}  (evidence: {evidence})"
            )
        return "\n".join(lines)

    def explain_chain(self, source: str, target: str, max_depth: int = 6) -> list[str]:
        """Normalize inputs and delegate to CausalReasoner."""
        from .causal import CausalReasoner  # local import avoids circular dep
        src = self._normalizer.normalize(source)
        tgt = self._normalizer.normalize(target)
        return CausalReasoner(self._relations).explain_chain(src, tgt)

    def consolidate(self):
        """Run ConsolidationEngine over all accumulated relations."""
        from .consolidation import ConsolidationEngine
        return ConsolidationEngine(self._relations).consolidate()

    def add_similarity(self, a: str, b: str, score: float = 1.0) -> None:
        """Register pairwise similarity between two entity names."""
        self._similarity_graph.add_similarity(a, b, score)

    def similarity(self, a: str, b: str) -> float:
        """Return the stored similarity score between two entities."""
        return self._similarity_graph.similarity(a, b)

    def structural_similarity(self, a: str, b: str) -> float:
        """
        Compute structural (Jaccard-over-profiles) similarity between two
        entities directly from the current relation graph. No oracle.
        """
        from .structural_similarity import StructuralSimilarityEngine
        a = self._normalizer.normalize(a)
        b = self._normalizer.normalize(b)
        return StructuralSimilarityEngine(self._relations).similarity(a, b)

    def discover_structural_similarities(self, min_score: float = 0.5):
        """Return discovered (a, b, score) structural similarity pairs."""
        from .structural_similarity import StructuralSimilarityEngine
        return StructuralSimilarityEngine(self._relations).discover_similarities(
            min_score=min_score
        )

    def add_structural_similarities(self, min_score: float = 0.5) -> int:
        """
        Discover structural similarities and load them into the world's
        SimilarityGraph so prediction can use them without a manual oracle.
        Returns the number of pairs added.
        """
        from .structural_similarity import StructuralSimilarityEngine
        engine = StructuralSimilarityEngine(
            self._relations, similarity_graph=self._similarity_graph
        )
        return engine.add_discovered_similarities(min_score=min_score)

    def predict_missing_links(self):
        """Infer likely missing relations from known repeating patterns."""
        from .prediction import PredictionEngine
        return PredictionEngine(
            self._relations,
            similarity_graph=self._similarity_graph,
            concepts=self._concepts,
        ).predict_missing_links()

    def discover_concepts(self):
        """
        Form concepts from repeated relational patterns, register member_of
        edges, and cache concepts for use by predict_missing_links().
        Returns the list of discovered Concept objects.
        """
        from .concepts import ConceptEngine
        self._concepts = ConceptEngine(self._relations).discover_concepts()
        for concept in self._concepts:
            self._ensure_object(concept.id)
            for member in concept.members:
                self._ensure_object(member)
                self._upsert_relation(member, "member_of", concept.id, concept.id)
        return self._concepts

    def predict_from_patterns(self, min_count: int = 5) -> list:
        """Predict transitive links via discovered frequent relation bigrams."""
        from .pattern_prediction import PatternBasedPredictor
        return PatternBasedPredictor(self._relations).predict_from_bigrams(
            min_count=min_count
        )

    def evaluate_prediction_recovery(
        self,
        ratio: float = 0.3,
        seed: int = 42,
        use_similarity: bool = True,
        use_concepts: bool = False,
    ):
        """
        Hide a fraction of relations, run predictions, return EvaluationResult.

        use_similarity — whether to allow SimilarityGraph in inference.
        use_concepts   — if True, discover_concepts() is called on the modified
                         world before prediction so concept-aware inference fires.
        """
        from .evaluation import PredictionEvaluator
        evaluator = PredictionEvaluator()
        modified_world, hidden = evaluator.hide_relations(self, ratio=ratio, seed=seed)
        if use_concepts:
            modified_world.discover_concepts()
        return evaluator.evaluate_predictions(
            modified_world, hidden,
            use_similarity=use_similarity,
            use_concepts=use_concepts,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_object(self, name: str) -> WorldObject:
        if name not in self._objects:
            self._objects[name] = WorldObject(id=name, name=name)
        return self._objects[name]

    def _upsert_relation(
        self, source: str, rel_type: str, target: str, raw_text: str
    ) -> None:
        for rel in self._relations:
            if rel.source == source and rel.relation_type == rel_type and rel.target == target:
                if raw_text not in rel.evidence:
                    rel.evidence.append(raw_text)
                return
        self._relations.append(
            Relation(source=source, relation_type=rel_type, target=target, evidence=[raw_text])
        )
