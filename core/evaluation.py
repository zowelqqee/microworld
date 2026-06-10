import random
from dataclasses import dataclass

from .relations import Relation

# Relation types that can be hidden; produces_result is the prediction anchor.
DEFAULT_HIDEABLE: frozenset[str] = frozenset({"contains", "grows_into"})


@dataclass
class EvaluationResult:
    total_hidden: int
    total_predictions: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float

    def __repr__(self) -> str:
        return (
            f"EvaluationResult("
            f"P={self.precision:.3f}, R={self.recall:.3f}, F1={self.f1:.3f} | "
            f"TP={self.true_positives}, FP={self.false_positives}, FN={self.false_negatives})"
        )


class PredictionEvaluator:
    """
    Benchmarks the PredictionEngine by hiding known relations, running
    predictions, and measuring precision / recall / F1.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def relation_key(obj) -> tuple[str, str, str]:
        """Canonical key for comparing a Relation or Prediction."""
        return (obj.source, obj.relation_type, obj.target)

    def hide_relations(
        self,
        world,
        relation_filter: frozenset[str] | None = None,
        ratio: float = 0.3,
        seed: int = 42,
    ):
        """
        Create a modified copy of *world* with a random subset of
        contains/grows_into relations removed.

        Returns (modified_world, hidden_relations).
        The original world is never mutated.
        """
        if relation_filter is None:
            relation_filter = DEFAULT_HIDEABLE

        all_rels = world.get_relations()
        hideable = [r for r in all_rels if r.relation_type in relation_filter]

        if not hideable:
            return self._build_world(world, all_rels), []

        rng = random.Random(seed)
        n_hide = max(1, round(len(hideable) * ratio))
        hidden = rng.sample(hideable, min(n_hide, len(hideable)))
        hidden_keys = {self.relation_key(r) for r in hidden}

        visible = [r for r in all_rels if self.relation_key(r) not in hidden_keys]
        return self._build_world(world, visible), hidden

    def evaluate_predictions(
        self,
        world,
        hidden_relations: list[Relation],
        use_similarity: bool = True,
        use_concepts: bool = True,
    ) -> "EvaluationResult":
        """
        Run predictions and score against *hidden_relations*.

        use_similarity — pass the world's SimilarityGraph to PredictionEngine.
        use_concepts   — pass the world's cached concepts to PredictionEngine.

        Setting a flag to False isolates that inference mode for clean benchmarking.
        """
        from .prediction import PredictionEngine

        similarity_graph = (
            getattr(world, "_similarity_graph", None) if use_similarity else None
        )
        concepts = getattr(world, "_concepts", []) if use_concepts else []

        predictions = PredictionEngine(
            world.get_relations(),
            similarity_graph=similarity_graph,
            concepts=concepts,
        ).predict_missing_links()

        hidden_keys = {self.relation_key(r) for r in hidden_relations}
        pred_keys = {self.relation_key(p) for p in predictions}

        tp = len(hidden_keys & pred_keys)
        fp = len(pred_keys - hidden_keys)
        fn = len(hidden_keys - pred_keys)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        return EvaluationResult(
            total_hidden=len(hidden_relations),
            total_predictions=len(predictions),
            true_positives=tp,
            false_positives=fp,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1=f1,
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_world(self, original_world, relations: list[Relation]):
        """Create a minimal World containing exactly the given relations."""
        from .world import World

        w = World(normalizer=original_world._normalizer)
        w._similarity_graph = original_world._similarity_graph
        for rel in relations:
            w._relations.append(rel)
            w._ensure_object(rel.source)
            w._ensure_object(rel.target)
        return w
