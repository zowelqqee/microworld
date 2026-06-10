import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World
from core.relations import Relation
from core.evaluation import PredictionEvaluator


# ── shared dataset ────────────────────────────────────────────────────────────

COMPLETE = [
    ("яблоня",              "яблоко",     "семя"),
    ("груша",               "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",   "семя"),
    ("персиковое_дерево",   "персик",     "косточка"),
    ("абрикосовое_дерево",  "абрикос",    "косточка"),
]
PARTIAL = [
    ("сливовое_дерево", "слива", "косточка"),
    ("лимонное_дерево", "лимон", "семя"),
]
MODES = [
    "majority_only",
    "manual_similarity",
    "structural_similarity",
    "concept_only",
    "hybrid_structural",
]


def build_world() -> tuple[World, list[Relation]]:
    w = World()
    for tree, fruit, conn in COMPLETE:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
        w.observe(f"{conn} вырастает в {tree}")
    for tree, fruit, conn in PARTIAL:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
    hidden = [Relation(conn, "grows_into", tree) for tree, fruit, conn in PARTIAL]
    return w, hidden


def run_mode(mode: str):
    w, hidden = build_world()
    use_sim = use_con = False
    if mode == "majority_only":
        pass
    elif mode == "manual_similarity":
        w.add_similarity("слива", "персик", 0.9)
        use_sim = True
    elif mode == "structural_similarity":
        w.add_structural_similarities(min_score=0.5)
        use_sim = True
    elif mode == "concept_only":
        w.discover_concepts()
        use_con = True
    elif mode == "hybrid_structural":
        w.add_structural_similarities(min_score=0.5)
        w.discover_concepts()
        use_sim = True
        use_con = True
    else:
        raise ValueError(mode)
    ev = PredictionEvaluator()
    return ev.evaluate_predictions(w, hidden, use_similarity=use_sim, use_concepts=use_con)


HIDDEN_KEYS = {
    ("косточка", "grows_into", "сливовое_дерево"),
    ("семя",     "grows_into", "лимонное_дерево"),
}


# ── all modes runnable ────────────────────────────────────────────────────────

class TestAllModesRun:
    @pytest.mark.parametrize("mode", MODES)
    def test_mode_produces_result(self, mode):
        r = run_mode(mode)
        assert 0.0 <= r.precision <= 1.0
        assert 0.0 <= r.recall <= 1.0
        assert 0.0 <= r.f1 <= 1.0

    def test_total_hidden_is_two(self):
        for mode in MODES:
            r = run_mode(mode)
            assert r.total_hidden == 2


# ── majority fails on minority pattern ────────────────────────────────────────

class TestMajorityFailsMinority:
    def test_majority_has_false_negative(self):
        r = run_mode("majority_only")
        assert r.false_negatives >= 1

    def test_majority_misses_pit_closing(self):
        """The косточка lifecycle closing for слива should NOT be recovered."""
        w, hidden = build_world()
        ev = PredictionEvaluator()
        from core.prediction import PredictionEngine
        preds = PredictionEngine(w.get_relations()).predict_missing_links()
        pred_keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("косточка", "grows_into", "сливовое_дерево") not in pred_keys

    def test_majority_predicts_contradiction(self):
        """Majority predicts слива contains семя, contradicting observed косточка."""
        w, _ = build_world()
        from core.prediction import PredictionEngine
        preds = PredictionEngine(w.get_relations()).predict_missing_links()
        pred_keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("слива", "contains", "семя") in pred_keys

    def test_majority_f1_below_one(self):
        assert run_mode("majority_only").f1 < 1.0


# ── structural improves over majority ─────────────────────────────────────────

class TestStructuralImproves:
    def test_structural_better_than_majority(self):
        maj = run_mode("majority_only")
        struct = run_mode("structural_similarity")
        assert struct.f1 > maj.f1

    def test_structural_recovers_pit_closing(self):
        struct = run_mode("structural_similarity")
        assert struct.true_positives == 2
        assert struct.false_negatives == 0

    def test_structural_perfect(self):
        struct = run_mode("structural_similarity")
        assert struct.precision == pytest.approx(1.0)
        assert struct.recall == pytest.approx(1.0)


# ── structural does not use manual oracle ─────────────────────────────────────

class TestStructuralNoOracle:
    def test_similarity_graph_has_structural_value_not_oracle(self):
        w, _ = build_world()
        w.add_structural_similarities(min_score=0.5)
        # structural слива~персик is 0.6 (Jaccard), NOT the manual oracle 0.9
        assert w.similarity("слива", "персик") == pytest.approx(0.6)
        assert w.similarity("слива", "персик") != pytest.approx(0.9)

    def test_no_manual_add_similarity_called(self):
        """The structural world must not contain the hand-injected 0.9 edge."""
        w, _ = build_world()
        w.add_structural_similarities(min_score=0.5)
        # No pair should equal exactly 0.9 (the oracle constant used elsewhere)
        for pair_score in w._similarity_graph._scores.values():
            assert pair_score != pytest.approx(0.9)

    def test_structural_matches_manual_outcome(self):
        """Structural reaches the same F1 as the oracle without the oracle."""
        man = run_mode("manual_similarity")
        struct = run_mode("structural_similarity")
        assert struct.f1 == pytest.approx(man.f1)


# ── hybrid not worse than structural ──────────────────────────────────────────

class TestHybridNotWorse:
    def test_hybrid_f1_geq_structural(self):
        struct = run_mode("structural_similarity")
        hybrid = run_mode("hybrid_structural")
        assert hybrid.f1 >= struct.f1

    def test_hybrid_perfect(self):
        hybrid = run_mode("hybrid_structural")
        assert hybrid.f1 == pytest.approx(1.0)

    def test_concept_only_also_recovers(self):
        """Direct concept membership recovers the minority pattern too."""
        concept = run_mode("concept_only")
        assert concept.true_positives == 2


# ── isolation: no shared mutation ─────────────────────────────────────────────

class TestIsolation:
    def test_fresh_worlds_are_independent(self):
        w1, _ = build_world()
        w2, _ = build_world()
        w1.add_similarity("слива", "персик", 0.9)
        w1.discover_concepts()
        # w2 must be pristine
        assert w2.similarity("слива", "персик") == pytest.approx(0.0)
        assert w2._concepts == []

    def test_mode_order_independence(self):
        """Running modes in different orders yields identical results."""
        forward = {m: run_mode(m) for m in MODES}
        backward = {m: run_mode(m) for m in reversed(MODES)}
        for m in MODES:
            assert forward[m].f1 == pytest.approx(backward[m].f1)
            assert forward[m].true_positives == backward[m].true_positives
            assert forward[m].false_positives == backward[m].false_positives

    def test_structural_mode_does_not_leak_into_majority(self):
        """A structural run must not affect a subsequent majority run."""
        run_mode("structural_similarity")
        maj = run_mode("majority_only")
        # majority must still fail on the minority pattern
        assert maj.false_negatives >= 1
        assert maj.f1 < 1.0

    def test_running_all_modes_does_not_change_majority_baseline(self):
        baseline = run_mode("majority_only")
        for m in MODES:
            run_mode(m)
        after = run_mode("majority_only")
        assert after.f1 == pytest.approx(baseline.f1)
        assert after.true_positives == baseline.true_positives


# ── ranking sanity ────────────────────────────────────────────────────────────

class TestRanking:
    def test_majority_is_worst(self):
        results = {m: run_mode(m).f1 for m in MODES}
        assert results["majority_only"] == min(results.values())

    def test_all_non_majority_modes_perfect(self):
        for m in ["manual_similarity", "structural_similarity", "concept_only", "hybrid_structural"]:
            assert run_mode(m).f1 == pytest.approx(1.0)
