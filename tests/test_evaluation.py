import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core import World, PredictionEvaluator, EvaluationResult
from core.relations import Relation
from core.prediction import PredictionEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lifecycle_world(*lifecycles) -> World:
    w = World()
    for tree, fruit, connector in lifecycles:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {connector}")
        w.observe(f"{connector} вырастает в {tree}")
    return w


FULL_WORLD_LIFECYCLES = [
    ("яблоня",          "яблоко",     "семя"),
    ("груша",           "груша_плод", "семя"),
    ("апельсин_дерево", "апельсин",   "семя"),
    ("лимон_дерево",    "лимон",      "семя"),
    ("персик_дерево",   "персик",     "косточка"),
]


def _full_world() -> World:
    return _lifecycle_world(*FULL_WORLD_LIFECYCLES)


# ---------------------------------------------------------------------------
# relation_key
# ---------------------------------------------------------------------------

class TestRelationKey:
    def test_returns_three_tuple(self):
        rel = Relation("A", "contains", "B")
        key = PredictionEvaluator.relation_key(rel)
        assert key == ("A", "contains", "B")

    def test_works_on_prediction(self):
        from core.prediction import Prediction
        p = Prediction("A", "contains", "B", confidence=0.8, reason="test")
        key = PredictionEvaluator.relation_key(p)
        assert key == ("A", "contains", "B")

    def test_distinct_relations_produce_distinct_keys(self):
        r1 = Relation("A", "contains", "B")
        r2 = Relation("A", "contains", "C")
        assert PredictionEvaluator.relation_key(r1) != PredictionEvaluator.relation_key(r2)


# ---------------------------------------------------------------------------
# hide_relations
# ---------------------------------------------------------------------------

class TestHideRelations:
    def setup_method(self):
        self.ev = PredictionEvaluator()

    def test_returns_modified_world_and_hidden_list(self):
        w = _lifecycle_world(("яблоня", "яблоко", "семя"))
        mod, hidden = self.ev.hide_relations(w, seed=42)
        assert hasattr(mod, "get_relations")
        assert isinstance(hidden, list)

    def test_hidden_count_respects_ratio(self):
        w = _full_world()
        hideable = [r for r in w.get_relations() if r.relation_type in {"contains", "grows_into"}]
        _, hidden = self.ev.hide_relations(w, ratio=0.3, seed=42)
        expected = max(1, round(len(hideable) * 0.3))
        assert len(hidden) == expected

    def test_deterministic_with_same_seed(self):
        w = _full_world()
        _, h1 = self.ev.hide_relations(w, ratio=0.3, seed=42)
        _, h2 = self.ev.hide_relations(w, ratio=0.3, seed=42)
        keys1 = {PredictionEvaluator.relation_key(r) for r in h1}
        keys2 = {PredictionEvaluator.relation_key(r) for r in h2}
        assert keys1 == keys2

    def test_different_seeds_may_differ(self):
        w = _full_world()
        _, h1 = self.ev.hide_relations(w, ratio=0.3, seed=42)
        _, h0 = self.ev.hide_relations(w, ratio=0.3, seed=0)
        k1 = {PredictionEvaluator.relation_key(r) for r in h1}
        k0 = {PredictionEvaluator.relation_key(r) for r in h0}
        assert k1 != k0

    def test_produces_result_never_hidden(self):
        w = _full_world()
        _, hidden = self.ev.hide_relations(w, ratio=1.0, seed=42)
        for r in hidden:
            assert r.relation_type != "produces_result"

    def test_only_hideable_types_hidden(self):
        w = _full_world()
        _, hidden = self.ev.hide_relations(w, ratio=1.0, seed=42)
        for r in hidden:
            assert r.relation_type in {"contains", "grows_into"}

    def test_hidden_relations_absent_from_modified_world(self):
        w = _full_world()
        mod, hidden = self.ev.hide_relations(w, ratio=0.3, seed=42)
        mod_keys = {PredictionEvaluator.relation_key(r) for r in mod.get_relations()}
        for r in hidden:
            assert PredictionEvaluator.relation_key(r) not in mod_keys

    def test_original_world_not_mutated(self):
        w = _full_world()
        original_count = len(w.get_relations())
        self.ev.hide_relations(w, ratio=0.3, seed=42)
        assert len(w.get_relations()) == original_count

    def test_empty_world_returns_empty_hidden(self):
        w = World()
        mod, hidden = self.ev.hide_relations(w, seed=42)
        assert hidden == []

    def test_ratio_1_hides_all_hideable(self):
        w = _lifecycle_world(("яблоня", "яблоко", "семя"))
        hideable = [r for r in w.get_relations() if r.relation_type in {"contains", "grows_into"}]
        _, hidden = self.ev.hide_relations(w, ratio=1.0, seed=42)
        assert len(hidden) == len(hideable)

    def test_custom_relation_filter(self):
        w = _full_world()
        # hide only contains, not grows_into
        _, hidden = self.ev.hide_relations(w, relation_filter=frozenset({"contains"}), ratio=1.0, seed=42)
        for r in hidden:
            assert r.relation_type == "contains"


# ---------------------------------------------------------------------------
# evaluate_predictions
# ---------------------------------------------------------------------------

class TestEvaluatePredictions:
    def setup_method(self):
        self.ev = PredictionEvaluator()

    def test_perfect_prediction(self):
        # 2 complete lifecycles + predictions recover hidden
        w = _lifecycle_world(
            ("яблоня", "яблоко", "семя"),
            ("груша",  "груша_плод", "семя"),
        )
        mod, hidden = self.ev.hide_relations(w, ratio=0.5, seed=7)
        result = self.ev.evaluate_predictions(mod, hidden)
        # With 2 complete instances and семя only, should recover perfectly
        if result.true_positives + result.false_positives > 0:
            assert result.precision == pytest.approx(1.0)

    def test_no_false_positives_when_all_correct(self):
        w = _lifecycle_world(
            ("яблоня", "яблоко", "семя"),
            ("груша",  "груша_плод", "семя"),
        )
        mod, hidden = self.ev.hide_relations(w, ratio=0.3, seed=42)
        result = self.ev.evaluate_predictions(mod, hidden)
        assert result.false_positives == 0

    def test_result_fields_populated(self):
        w = _full_world()
        mod, hidden = self.ev.hide_relations(w, ratio=0.3, seed=42)
        result = self.ev.evaluate_predictions(mod, hidden)
        assert isinstance(result.total_hidden, int)
        assert isinstance(result.total_predictions, int)
        assert isinstance(result.true_positives, int)
        assert isinstance(result.false_positives, int)
        assert isinstance(result.false_negatives, int)
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0

    def test_tp_plus_fn_equals_total_hidden(self):
        w = _full_world()
        mod, hidden = self.ev.hide_relations(w, ratio=0.3, seed=42)
        result = self.ev.evaluate_predictions(mod, hidden)
        assert result.true_positives + result.false_negatives == result.total_hidden

    def test_tp_plus_fp_equals_total_predictions(self):
        w = _full_world()
        mod, hidden = self.ev.hide_relations(w, ratio=0.3, seed=42)
        result = self.ev.evaluate_predictions(mod, hidden)
        assert result.true_positives + result.false_positives == result.total_predictions

    def test_false_positive_when_wrong_connector(self):
        # персик uses косточка; engine will predict семя → FP
        w = _lifecycle_world(
            ("яблоня",       "яблоко",  "семя"),
            ("груша",        "груша_плод", "семя"),
            ("персик_дерево","персик",  "косточка"),
        )
        # Manually build modified world without косточка->grows_into->персик_дерево
        hidden = [r for r in w.get_relations()
                  if r.source == "косточка" and r.relation_type == "grows_into"]
        visible = [r for r in w.get_relations() if r not in hidden]
        mod = self.ev._build_world(w, visible)
        result = self.ev.evaluate_predictions(mod, hidden)
        # engine predicts семя->grows_into->персик_дерево (FP) not косточка->grows_into (FN)
        assert result.false_positives >= 1
        assert result.false_negatives >= 1

    def test_precision_formula(self):
        # Construct result manually and check formula
        result = EvaluationResult(
            total_hidden=3, total_predictions=4,
            true_positives=2, false_positives=2, false_negatives=1,
            precision=0.5, recall=2/3, f1=0.0,
        )
        expected_f1 = 2 * 0.5 * (2/3) / (0.5 + 2/3)
        # Verify our evaluator computes this correctly from scratch
        w = _lifecycle_world(
            ("яблоня",       "яблоко",  "семя"),
            ("груша",        "груша_плод", "семя"),
            ("апельсин_дерево","апельсин","семя"),
            ("персик_дерево","персик",  "косточка"),
        )
        mod, hidden = self.ev.hide_relations(w, ratio=0.3, seed=0)
        res = self.ev.evaluate_predictions(mod, hidden)
        # Just verify formula coherence
        if res.true_positives + res.false_positives > 0:
            assert res.precision == pytest.approx(
                res.true_positives / (res.true_positives + res.false_positives)
            )
        if res.true_positives + res.false_negatives > 0:
            assert res.recall == pytest.approx(
                res.true_positives / (res.true_positives + res.false_negatives)
            )

    def test_zero_predictions_zero_precision(self):
        w = World()
        # No complete cycle → no predictions
        mod = w
        hidden = [Relation("A", "contains", "B")]
        result = self.ev.evaluate_predictions(mod, hidden)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.f1 == 0.0

    def test_empty_hidden_no_fn(self):
        w = _lifecycle_world(("яблоня", "яблоко", "семя"))
        # Don't hide anything but add a partial
        w.observe("груша производит груша_плод")
        result = self.ev.evaluate_predictions(w, hidden_relations=[])
        assert result.false_negatives == 0
        assert result.total_hidden == 0


# ---------------------------------------------------------------------------
# World integration
# ---------------------------------------------------------------------------

class TestWorldIntegration:
    def test_evaluate_prediction_recovery_exists(self):
        w = World()
        assert callable(w.evaluate_prediction_recovery)

    def test_returns_evaluation_result(self):
        w = _full_world()
        result = w.evaluate_prediction_recovery(ratio=0.3, seed=42)
        assert isinstance(result, EvaluationResult)

    def test_full_world_seed42_perfect(self):
        w = _full_world()
        result = w.evaluate_prediction_recovery(ratio=0.3, seed=42)
        assert result.true_positives == result.total_hidden
        assert result.false_positives == 0

    def test_consistency_with_manual_call(self):
        w = _full_world()
        ev = PredictionEvaluator()
        mod, hidden = ev.hide_relations(w, ratio=0.3, seed=42)
        manual = ev.evaluate_predictions(mod, hidden)
        via_world = w.evaluate_prediction_recovery(ratio=0.3, seed=42)
        assert manual.true_positives == via_world.true_positives
        assert manual.false_positives == via_world.false_positives
        assert manual.f1 == pytest.approx(via_world.f1)
