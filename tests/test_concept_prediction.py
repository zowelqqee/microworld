import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, ConceptEngine
from core.prediction import PredictionEngine
from core.relations import Relation


# ── helpers ──────────────────────────────────────────────────────────────────

def _lifecycle_world(*lifecycles) -> World:
    w = World()
    for tree, fruit, connector in lifecycles:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {connector}")
        w.observe(f"{connector} вырастает в {tree}")
    return w


# 3 семя-trees + 2 косточка-trees → concept_rel_contains_семя (3 members),
# concept_rel_contains_косточка (2 members)
MIXED = [
    ("яблоня",            "яблоко",     "семя"),
    ("груша",             "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",  "семя"),
    ("персиковое_дерево", "персик",     "косточка"),
    ("абрикосовое_дерево", "абрикос",   "косточка"),
]


def _mixed_world_with_partial(partial_tree="сливовое_дерево", partial_fruit="слива") -> World:
    w = _lifecycle_world(*MIXED)
    w.observe(f"{partial_tree} производит {partial_fruit}")
    return w


# ── Concept lookup internals ──────────────────────────────────────────────────

class TestConceptLookup:
    def test_find_best_concept_returns_none_without_similarity(self):
        w = _mixed_world_with_partial()
        w.discover_concepts()
        engine = PredictionEngine(
            w.get_relations(),
            similarity_graph=w._similarity_graph,
            concepts=w._concepts,
        )
        concept, score = engine._find_best_concept("слива")
        assert concept is None
        assert score == pytest.approx(0.0)

    def test_find_best_concept_with_similarity(self):
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        w.discover_concepts()
        engine = PredictionEngine(
            w.get_relations(),
            similarity_graph=w._similarity_graph,
            concepts=w._concepts,
        )
        concept, score = engine._find_best_concept("слива")
        assert concept is not None
        assert concept.id == "concept_rel_contains_косточка"
        assert score == pytest.approx(0.9)

    def test_concept_connector_extraction(self):
        w = _mixed_world_with_partial()
        w.discover_concepts()
        engine = PredictionEngine(w.get_relations(), concepts=w._concepts)
        concept = next(c for c in w._concepts if "косточка" in c.id)
        assert engine._concept_connector(concept) == "косточка"

    def test_concept_connector_extraction_семя(self):
        w = _mixed_world_with_partial()
        w.discover_concepts()
        engine = PredictionEngine(w.get_relations(), concepts=w._concepts)
        concept = next(c for c in w._concepts if "семя" in c.id and c.kind == "relation_group")
        assert engine._concept_connector(concept) == "семя"

    def test_only_relation_groups_used_for_concept_lookup(self):
        """Lifecycle-group concepts should not be used for connector selection."""
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "яблоня", 0.9)  # яблоня is in lifecycle group, not relation group
        w.discover_concepts()
        engine = PredictionEngine(
            w.get_relations(),
            similarity_graph=w._similarity_graph,
            concepts=w._concepts,
        )
        concept, score = engine._find_best_concept("слива")
        # яблоня is a member of concept_lifecycle_семя (kind=lifecycle_group) — should be skipped
        # so concept should be None or a relation_group
        if concept is not None:
            assert concept.kind == "relation_group"


# ── Concept-based prediction ──────────────────────────────────────────────────

class TestConceptPrediction:
    def test_concept_overrides_majority_vote(self):
        """слива similar to персик → concept_rel_contains_косточка → косточка not семя."""
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        w.discover_concepts()
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "косточка"

    def test_concept_reason_string(self):
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        w.discover_concepts()
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert "concept match" in p.reason
        assert "concept_rel_contains_косточка" in p.reason

    def test_concept_does_not_predict_majority_connector(self):
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        w.discover_concepts()
        preds = w.predict_missing_links()
        sliva_targets = {p.target for p in preds if p.source == "слива"}
        assert "семя" not in sliva_targets

    def test_closing_link_uses_concept_connector(self):
        """косточка -grows_into-> сливовое_дерево should also be predicted."""
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        w.discover_concepts()
        preds = w.predict_missing_links()
        closing = next(
            (p for p in preds if p.source == "косточка" and p.target == "сливовое_дерево"),
            None,
        )
        assert closing is not None

    def test_without_concepts_falls_back_to_similarity(self):
        """Same similarity, but concepts not discovered → similarity path used."""
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        # NOT calling discover_concepts()
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "косточка"
        assert "nearest lifecycle match" in p.reason

    def test_without_similarity_falls_back_to_majority(self):
        """No similarity added, discover_concepts called → falls back to majority (семя)."""
        w = _mixed_world_with_partial()
        w.discover_concepts()
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "семя"
        assert "majority" in p.reason

    def test_no_concepts_no_similarity_majority(self):
        """Neither similarity nor concepts → plain majority vote."""
        w = _mixed_world_with_partial()
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "семя"
        assert "majority" in p.reason


# ── Confidence model ──────────────────────────────────────────────────────────

class TestConceptConfidence:
    def test_concept_confidence_higher_than_majority_small_world(self):
        """
        With only 2 complete instances majority conf = 0.80.
        Concept of 2 members gives 0.85 + 0.02*2 = 0.89 > 0.80.
        """
        cycles = [
            ("яблоня", "яблоко",  "семя"),
            ("груша",  "груша_плод", "семя"),
        ]
        w = _lifecycle_world(*cycles)
        w.observe("сливовое_дерево производит слива")

        # majority (2 instances → conf 0.80)
        preds_maj = w.predict_missing_links()
        conf_maj = next(p for p in preds_maj if p.source == "слива").confidence
        assert conf_maj == pytest.approx(0.80)

        # concept-aware (семя concept has 2 members → conf 0.89)
        w.add_similarity("слива", "яблоко", 0.1)
        w.discover_concepts()
        preds_concept = w.predict_missing_links()
        conf_concept = next(p for p in preds_concept if p.source == "слива").confidence
        assert conf_concept > conf_maj

    def test_concept_confidence_formula(self):
        """concept_rel_contains_косточка has 2 members → 0.85 + 0.02*2 = 0.89."""
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        w.discover_concepts()
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        expected = min(0.95, 0.85 + 0.02 * 2)
        assert p.confidence == pytest.approx(expected)

    def test_concept_confidence_grows_with_concept_size(self):
        """Larger concept → higher confidence."""
        # 3-member косточка concept
        cycles_3 = [
            ("яблоня", "яблоко", "семя"),
            ("персиковое_дерево", "персик",  "косточка"),
            ("абрикосовое_дерево", "абрикос", "косточка"),
            ("вишня_дерево",       "вишня",   "косточка"),
        ]
        w3 = _lifecycle_world(*cycles_3)
        w3.observe("сливовое_дерево производит слива")
        w3.add_similarity("слива", "персик", 0.9)
        w3.discover_concepts()
        preds3 = w3.predict_missing_links()
        conf3 = next(p for p in preds3 if p.source == "слива").confidence

        # 2-member косточка concept
        w2 = _mixed_world_with_partial()
        w2.add_similarity("слива", "персик", 0.9)
        w2.discover_concepts()
        preds2 = w2.predict_missing_links()
        conf2 = next(p for p in preds2 if p.source == "слива").confidence

        assert conf3 > conf2

    def test_concept_confidence_capped_at_095(self):
        cycles = [("яблоня", "яблоко", "семя")] + [
            (f"дерево_{i}", f"плод_{i}", "косточка") for i in range(10)
        ]
        w = _lifecycle_world(*cycles)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "плод_0", 0.9)
        w.discover_concepts()
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert p.confidence <= 0.95


# ── Priority chain ────────────────────────────────────────────────────────────

class TestPriorityChain:
    def test_concept_beats_similarity_when_both_available(self):
        """
        слива similar to персик (косточка concept, 2 members)
        AND similar to яблоко (семя concept, 3 members).
        The higher-scoring concept wins regardless of concept size.
        """
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)   # косточка concept
        w.add_similarity("слива", "яблоко",  0.3)  # семя concept (lower score)
        w.discover_concepts()
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert p.target == "косточка"
        assert "concept match" in p.reason

    def test_similarity_beats_majority_when_no_concepts(self):
        w = _mixed_world_with_partial()
        w.add_similarity("слива", "персик", 0.9)
        # no discover_concepts
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert p.target == "косточка"
        assert "nearest lifecycle match" in p.reason

    def test_majority_used_when_nothing_else(self):
        w = _mixed_world_with_partial()
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert "majority" in p.reason


# ── evaluate_prediction_recovery integration ──────────────────────────────────

class TestEvalIntegration:
    def test_use_concepts_flag_accepted(self):
        w = _lifecycle_world(*MIXED)
        result = w.evaluate_prediction_recovery(ratio=0.3, seed=42, use_concepts=False)
        assert hasattr(result, "f1")

    def test_use_concepts_true_runs_without_error(self):
        w = _lifecycle_world(*MIXED)
        result = w.evaluate_prediction_recovery(ratio=0.3, seed=42, use_concepts=True)
        assert 0.0 <= result.f1 <= 1.0

    def test_similarity_graph_propagates_to_modified_world(self):
        """Similarities added before evaluation must be available in modified world."""
        w = _lifecycle_world(*MIXED)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        # The modified world inside the evaluator should still have the similarity graph
        from core.evaluation import PredictionEvaluator
        ev = PredictionEvaluator()
        mod, _ = ev.hide_relations(w, ratio=0.1, seed=42)
        assert mod._similarity_graph.similarity("слива", "персик") == pytest.approx(0.9)

    def test_concepts_empty_without_discover_call(self):
        """_concepts starts empty; prediction falls back to majority."""
        w = _lifecycle_world(*MIXED)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        # similarity exists, concepts are empty → similarity path
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert "nearest lifecycle match" in p.reason
