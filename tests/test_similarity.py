import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, SimilarityGraph
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


SEMYA_LIFECYCLES = [
    ("яблоня",   "яблоко",     "семя"),
    ("груша",    "груша_плод", "семя"),
]
MIXED_LIFECYCLES = [
    ("яблоня",            "яблоко",  "семя"),
    ("груша",             "груша_плод", "семя"),
    ("персиковое_дерево", "персик",  "косточка"),
]


# ── SimilarityGraph unit tests ────────────────────────────────────────────────

class TestSimilarityGraph:
    def test_stores_similarity(self):
        sg = SimilarityGraph()
        sg.add_similarity("A", "B", 0.8)
        assert sg.similarity("A", "B") == pytest.approx(0.8)

    def test_undirected(self):
        sg = SimilarityGraph()
        sg.add_similarity("A", "B", 0.7)
        assert sg.similarity("B", "A") == pytest.approx(0.7)

    def test_same_name_returns_one(self):
        sg = SimilarityGraph()
        assert sg.similarity("X", "X") == pytest.approx(1.0)

    def test_unknown_pair_returns_zero(self):
        sg = SimilarityGraph()
        assert sg.similarity("A", "B") == pytest.approx(0.0)

    def test_overwrite_score(self):
        sg = SimilarityGraph()
        sg.add_similarity("A", "B", 0.5)
        sg.add_similarity("A", "B", 0.9)
        assert sg.similarity("A", "B") == pytest.approx(0.9)

    def test_default_score_is_one(self):
        sg = SimilarityGraph()
        sg.add_similarity("A", "B")
        assert sg.similarity("A", "B") == pytest.approx(1.0)

    def test_most_similar_returns_best(self):
        sg = SimilarityGraph()
        sg.add_similarity("target", "A", 0.3)
        sg.add_similarity("target", "B", 0.8)
        sg.add_similarity("target", "C", 0.5)
        name, score = sg.most_similar("target", ["A", "B", "C"])
        assert name == "B"
        assert score == pytest.approx(0.8)

    def test_most_similar_no_match_returns_none(self):
        sg = SimilarityGraph()
        name, score = sg.most_similar("target", ["A", "B"])
        assert name is None
        assert score == pytest.approx(0.0)

    def test_most_similar_empty_candidates(self):
        sg = SimilarityGraph()
        name, score = sg.most_similar("target", [])
        assert name is None
        assert score == pytest.approx(0.0)

    def test_normalizer_integration(self):
        from core.normalizer import EntityNormalizer
        norm = EntityNormalizer()
        norm.add_alias("яблоку", "яблоко")
        sg = SimilarityGraph(normalizer=norm)
        sg.add_similarity("яблоку", "груша", 0.6)
        # "яблоку" normalizes to "яблоко", so lookup by canonical works
        assert sg.similarity("яблоко", "груша") == pytest.approx(0.6)


# ── Majority vote fallback ────────────────────────────────────────────────────

class TestMajorityFallback:
    def test_no_similarity_uses_majority(self):
        """Without similarity graph, engine uses majority-vote connector."""
        w = _lifecycle_world(*SEMYA_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        preds = w.predict_missing_links()
        targets = {p.target for p in preds}
        assert "семя" in targets

    def test_fallback_reason_mentions_template(self):
        w = _lifecycle_world(*SEMYA_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert "majority" in p.reason

    def test_fallback_confidence_formula(self):
        """2 complete instances → base 0.7 + 0.1 = 0.80."""
        w = _lifecycle_world(*SEMYA_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert p.confidence == pytest.approx(0.80)


# ── Similarity overrides majority ─────────────────────────────────────────────

class TestSimilarityOverride:
    def test_similarity_selects_nearest_connector(self):
        """слива similar to персик → connector should be косточка, not семя."""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        preds = w.predict_missing_links()
        # слива contains ? — should be косточка
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "косточка"

    def test_similarity_reason_mentions_nearest_match(self):
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert "nearest lifecycle match" in p.reason
        assert "персик" in p.reason
        assert "0.90" in p.reason

    def test_similarity_does_not_predict_majority_connector(self):
        """With similarity pointing to персик, семя should NOT be predicted for слива."""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        preds = w.predict_missing_links()
        sliva_preds = [p for p in preds if p.source == "слива"]
        targets = {p.target for p in sliva_preds}
        assert "семя" not in targets

    def test_closing_link_uses_same_connector(self):
        """косточка grows_into сливовое_дерево should also be predicted."""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        preds = w.predict_missing_links()
        closing = next(
            (p for p in preds
             if p.source == "косточка" and p.target == "сливовое_дерево"),
            None,
        )
        assert closing is not None

    def test_zero_similarity_falls_back_to_majority(self):
        """Explicit score of 0.0 is equivalent to no similarity — majority is used."""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.0)
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "семя"

    def test_lower_similarity_wins_over_no_similarity(self):
        """Any positive score beats majority-vote for the matched slot."""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.1)
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "косточка"


# ── Confidence formula ────────────────────────────────────────────────────────

class TestSimilarityConfidence:
    def test_confidence_increases_with_score(self):
        scores = [0.2, 0.5, 0.8, 1.0]
        confs = []
        for score in scores:
            w = _lifecycle_world(*MIXED_LIFECYCLES)
            w.observe("сливовое_дерево производит слива")
            w.add_similarity("слива", "персик", score)
            preds = w.predict_missing_links()
            p = next(p for p in preds if p.source == "слива")
            confs.append(p.confidence)
        assert confs == sorted(confs)

    def test_confidence_formula_at_09(self):
        """score=0.9 → 0.65 + 0.30*0.9 = 0.92"""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert p.confidence == pytest.approx(0.65 + 0.30 * 0.9, abs=1e-9)

    def test_confidence_capped_at_095(self):
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 1.0)
        preds = w.predict_missing_links()
        p = next(p for p in preds if p.source == "слива")
        assert p.confidence <= 0.95

    def test_similarity_confidence_higher_than_majority_at_high_score(self):
        """score=1.0 → similarity conf 0.95 > majority conf 0.80 (2 instances)."""
        w = _lifecycle_world(*SEMYA_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        # Without similarity
        preds_maj = w.predict_missing_links()
        conf_maj = next(p for p in preds_maj if p.source == "слива").confidence

        w.add_similarity("слива", "яблоко", 1.0)
        preds_sim = w.predict_missing_links()
        conf_sim = next(p for p in preds_sim if p.source == "слива").confidence

        assert conf_sim > conf_maj


# ── World integration ─────────────────────────────────────────────────────────

class TestWorldIntegration:
    def test_add_similarity_exists(self):
        w = World()
        assert callable(w.add_similarity)

    def test_similarity_method_exists(self):
        w = World()
        assert callable(w.similarity)

    def test_similarity_roundtrip(self):
        w = World()
        w.add_similarity("A", "B", 0.75)
        assert w.similarity("A", "B") == pytest.approx(0.75)
        assert w.similarity("B", "A") == pytest.approx(0.75)

    def test_predict_uses_similarity_graph(self):
        """World.predict_missing_links should route through similarity graph."""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.observe("сливовое_дерево производит слива")
        w.add_similarity("слива", "персик", 0.9)
        preds = w.predict_missing_links()
        sliva_pred = next((p for p in preds if p.source == "слива"), None)
        assert sliva_pred is not None
        assert sliva_pred.target == "косточка"

    def test_similarity_graph_shared_across_calls(self):
        """add_similarity before predict_missing_links should be visible."""
        w = _lifecycle_world(*MIXED_LIFECYCLES)
        w.add_similarity("слива", "персик", 0.9)
        w.observe("сливовое_дерево производит слива")
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert p.target == "косточка"
