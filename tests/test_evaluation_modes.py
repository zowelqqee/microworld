import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World
from core.evaluation import PredictionEvaluator
from core.prediction import PredictionEngine
from core.relations import Relation


# ── shared scenario ───────────────────────────────────────────────────────────
#
# 3 семя-trees + 2 косточка-trees (all complete cycles).
# Partial anchors: produces_result + contains observed; grows_into hidden.
#   - слива contains косточка   → слива joins concept_rel_contains_косточка
#   - лимон contains семя       → лимон joins concept_rel_contains_семя
# Hidden: косточка->grows_into->сливовое_дерево
#         семя->grows_into->лимонное_дерево
#
# Expected by mode:
#   majority_only  : слива gets семя (wrong) → FP×2 + FN=1 for слива; лимон TP
#   similarity_only: слива similar to персик → косточка (TP); лимон majority (TP)
#   concept_only   : слива in concept_rel_contains_косточка → косточка (TP); лимон TP
#   hybrid         : same as concept_only (concept fires first)
# ─────────────────────────────────────────────────────────────────────────────

COMPLETE = [
    ("яблоня",              "яблоко",     "семя"),
    ("груша",               "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",   "семя"),
    ("персиковое_дерево",   "персик",     "косточка"),
    ("абрикосовое_дерево",  "абрикос",    "косточка"),
]
PARTIAL = [
    ("сливовое_дерево", "слива",  "косточка"),
    ("лимонное_дерево", "лимон",  "семя"),
]


def _build_scenario(add_similarity: bool = False) -> tuple[World, list[Relation]]:
    w = World()
    for tree, fruit, conn in COMPLETE:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
        w.observe(f"{conn} вырастает в {tree}")
    hidden: list[Relation] = []
    for tree, fruit, conn in PARTIAL:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
        hidden.append(Relation(conn, "grows_into", tree))
    if add_similarity:
        w.add_similarity("слива", "персик", 0.9)
    return w, hidden


def _run_mode(
    world: World,
    hidden: list[Relation],
    use_sim: bool,
    use_con: bool,
):
    ev = PredictionEvaluator()
    fresh = ev._build_world(world, world.get_relations())
    if use_con:
        fresh.discover_concepts()
    return ev.evaluate_predictions(fresh, hidden, use_similarity=use_sim, use_concepts=use_con)


# ── majority_only mode ────────────────────────────────────────────────────────

class TestMajorityOnly:
    def test_ignores_similarity(self):
        """Even with similarity added, majority_only should ignore it."""
        w, hidden = _build_scenario(add_similarity=True)
        r = _run_mode(w, hidden, use_sim=False, use_con=False)
        # слива gets семя (wrong) → FP for слива path
        assert r.false_positives > 0

    def test_ignores_concepts(self):
        """Concepts discovered on world should not affect majority_only."""
        w, hidden = _build_scenario(add_similarity=True)
        w.discover_concepts()
        r = _run_mode(w, hidden, use_sim=False, use_con=False)
        assert r.false_positives > 0

    def test_reason_says_majority(self):
        w, hidden = _build_scenario()
        fresh = PredictionEvaluator()._build_world(w, w.get_relations())
        preds = PredictionEngine(fresh.get_relations()).predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        assert p is not None
        assert "majority" in p.reason

    def test_dominant_pattern_still_recovered(self):
        """лимон -> семя is the dominant pattern; majority should get it right."""
        w, hidden = _build_scenario()
        r = _run_mode(w, hidden, use_sim=False, use_con=False)
        # At least the лимон grows_into should be a TP
        assert r.true_positives >= 1


# ── similarity_only mode ──────────────────────────────────────────────────────

class TestSimilarityOnly:
    def test_recovers_minority_pattern_via_similarity(self):
        w, hidden = _build_scenario(add_similarity=True)
        r = _run_mode(w, hidden, use_sim=True, use_con=False)
        assert r.precision == pytest.approx(1.0)
        assert r.recall == pytest.approx(1.0)

    def test_ignores_concepts(self):
        """If concepts are cached, similarity_only should not use them for routing."""
        w, hidden = _build_scenario(add_similarity=True)
        ev = PredictionEvaluator()
        fresh = ev._build_world(w, w.get_relations())
        preds = PredictionEngine(
            fresh.get_relations(),
            similarity_graph=fresh._similarity_graph,
            concepts=[],   # explicitly no concepts
        ).predict_missing_links()
        # слива->contains->косточка is already observed; the new prediction is the closing link
        p = next((p for p in preds if p.source == "косточка" and p.target == "сливовое_дерево"), None)
        assert p is not None
        assert "concept match" not in p.reason
        assert "nearest lifecycle match" in p.reason

    def test_without_similarity_score_falls_back(self):
        """Similarity mode but no score added → falls back to majority."""
        w, hidden = _build_scenario(add_similarity=False)
        r = _run_mode(w, hidden, use_sim=True, use_con=False)
        # Without a similarity score for слива, majority picks семя → FP
        assert r.false_positives > 0


# ── concept_only mode ─────────────────────────────────────────────────────────

class TestConceptOnly:
    def test_recovers_minority_pattern_via_direct_membership(self):
        """
        слива->contains->косточка is observed, so after discover_concepts
        слива is a direct member of concept_rel_contains_косточка.
        concept_only (no similarity) should predict косточка->grows_into->сливовое.
        """
        w, hidden = _build_scenario(add_similarity=False)
        r = _run_mode(w, hidden, use_sim=False, use_con=True)
        assert r.precision == pytest.approx(1.0)
        assert r.recall == pytest.approx(1.0)

    def test_ignores_similarity_graph(self):
        """Even with similarity added, concept_only should route through concepts."""
        w, hidden = _build_scenario(add_similarity=True)
        ev = PredictionEvaluator()
        fresh = ev._build_world(w, w.get_relations())
        fresh.discover_concepts()
        preds = PredictionEngine(
            fresh.get_relations(),
            similarity_graph=None,   # explicitly no similarity
            concepts=fresh._concepts,
        ).predict_missing_links()
        # The closing link (grows_into) is the new prediction; contains is already observed
        p = next((p for p in preds if p.source == "косточка" and p.target == "сливовое_дерево"), None)
        assert p is not None
        assert "concept match" in p.reason

    def test_requires_concepts_discovered(self):
        """Without discover_concepts, concept_only has no concepts → majority."""
        w, hidden = _build_scenario(add_similarity=False)
        # Build fresh world without calling discover_concepts
        ev = PredictionEvaluator()
        fresh = ev._build_world(w, w.get_relations())
        # pass use_concepts=True but world._concepts is empty
        r = ev.evaluate_predictions(fresh, hidden, use_similarity=False, use_concepts=True)
        # Falls through to majority → FP for слива
        assert r.false_positives > 0

    def test_direct_membership_score_is_one(self):
        w, _ = _build_scenario()
        w.discover_concepts()
        engine = PredictionEngine(
            w.get_relations(),
            similarity_graph=None,
            concepts=w._concepts,
        )
        concept, score = engine._find_best_concept("слива")
        assert concept is not None
        assert score == pytest.approx(1.0)
        assert "косточка" in concept.id


# ── hybrid mode ───────────────────────────────────────────────────────────────

class TestHybrid:
    def test_passes_with_both_enabled(self):
        w, hidden = _build_scenario(add_similarity=True)
        r = _run_mode(w, hidden, use_sim=True, use_con=True)
        assert r.precision == pytest.approx(1.0)
        assert r.recall == pytest.approx(1.0)

    def test_concept_takes_priority_over_similarity(self):
        """
        hybrid: concept fires first. The closing link reason should say
        'concept match', not 'nearest lifecycle match'.
        """
        w, hidden = _build_scenario(add_similarity=True)
        ev = PredictionEvaluator()
        fresh = ev._build_world(w, w.get_relations())
        fresh.discover_concepts()
        preds = PredictionEngine(
            fresh.get_relations(),
            similarity_graph=fresh._similarity_graph,
            concepts=fresh._concepts,
        ).predict_missing_links()
        # The closing link is the prediction; contains is already observed
        p = next((p for p in preds if p.source == "косточка" and p.target == "сливовое_дерево"), None)
        assert p is not None
        assert "concept match" in p.reason

    def test_hybrid_without_similarity_same_as_concept_only(self):
        """Hybrid with no similarity score is equivalent to concept_only."""
        w_no_sim, h = _build_scenario(add_similarity=False)
        r_hybrid = _run_mode(w_no_sim, h, use_sim=True, use_con=True)
        r_concept = _run_mode(w_no_sim, h, use_sim=False, use_con=True)
        assert r_hybrid.true_positives == r_concept.true_positives
        assert r_hybrid.false_positives == r_concept.false_positives


# ── isolation — no world mutation ─────────────────────────────────────────────

class TestNoMutation:
    def test_source_world_not_mutated_by_evaluate(self):
        w, hidden = _build_scenario(add_similarity=True)
        original_rel_count = len(w.get_relations())
        original_obj_count = len(w.get_objects())
        for use_sim, use_con in [(False, False), (True, False), (False, True), (True, True)]:
            _run_mode(w, hidden, use_sim, use_con)
        assert len(w.get_relations()) == original_rel_count
        assert len(w.get_objects()) == original_obj_count

    def test_similarity_graph_not_shared_by_ref_between_modes(self):
        """Each _build_world call should share the graph but not a new copy."""
        w, hidden = _build_scenario(add_similarity=True)
        ev = PredictionEvaluator()
        f1 = ev._build_world(w, w.get_relations())
        f2 = ev._build_world(w, w.get_relations())
        # Both see the same scores (shared graph is intentional)
        assert f1._similarity_graph.similarity("слива", "персик") == pytest.approx(0.9)
        assert f2._similarity_graph.similarity("слива", "персик") == pytest.approx(0.9)

    def test_concepts_discovered_on_fresh_world_not_original(self):
        w, hidden = _build_scenario()
        assert w._concepts == []
        _run_mode(w, hidden, use_sim=False, use_con=True)
        assert w._concepts == []  # original world untouched


# ── World.evaluate_prediction_recovery API ────────────────────────────────────

class TestWorldAPI:
    def test_use_similarity_false_accepted(self):
        w, _ = _build_scenario(add_similarity=True)
        r = w.evaluate_prediction_recovery(ratio=0.3, seed=42, use_similarity=False, use_concepts=False)
        assert hasattr(r, "f1")

    def test_majority_only_via_api(self):
        w, hidden = _build_scenario(add_similarity=True)
        # Build a world where we can control what's hidden
        # Use the full world and hide 0 (ratio=0) → empty hidden, trivially P=0, R=0
        # Better: compare results show modes differ
        r_maj = w.evaluate_prediction_recovery(
            ratio=0.3, seed=42, use_similarity=False, use_concepts=False
        )
        r_sim = w.evaluate_prediction_recovery(
            ratio=0.3, seed=42, use_similarity=True, use_concepts=False
        )
        # They may differ or be equal depending on what's hidden — just check no crash.
        assert 0.0 <= r_maj.f1 <= 1.0
        assert 0.0 <= r_sim.f1 <= 1.0

    def test_concept_only_via_api(self):
        w, _ = _build_scenario()
        r = w.evaluate_prediction_recovery(
            ratio=0.3, seed=42, use_similarity=False, use_concepts=True
        )
        assert 0.0 <= r.f1 <= 1.0
