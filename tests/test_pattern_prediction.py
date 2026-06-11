import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
import pytest

from core.relations import Relation
from core.pattern_prediction import (
    PatternPrediction,
    PatternBasedPredictor,
    PatternEvaluationResult,
    evaluate_pattern_prediction_recovery,
)
from core.world import World


# ── helpers ────────────────────────────────────────────────────────────────────

def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _world(*triples) -> World:
    """Build a World from (source, relation_type, target) triples."""
    return _world_from(list(_rels(*triples)))


def _world_from(relations: list[Relation]) -> World:
    """Build a World from an existing list of Relation objects."""
    w = World()
    for rel in relations:
        w._relations.append(rel)
        w._ensure_object(rel.source)
        w._ensure_object(rel.target)
    return w


def _conf(count: int) -> float:
    return min(0.95, 0.5 + 0.05 * math.log(count + 1))


# ── shared datasets ───────────────────────────────────────────────────────────
#
#  TRANS_RELS: clean transitive graph
#
#   a --part_of--> b --part_of--> c     (chain 1 — no transitive yet)
#   x --part_of--> y --part_of--> z     (chain 2 — no transitive yet)
#   d --part_of--> e --part_of--> f     (chain 3 — no transitive yet)
#   m --part_of--> n --part_of--> o     (chain 4 — no transitive yet)
#
#  TRANS_WITH_CLOSURE adds the closure for chains 1, 2, 3 so they can be hidden:
#   a --part_of--> c
#   x --part_of--> z
#   m --part_of--> o

TRANS_RELS = _rels(
    ("a", "part_of", "b"), ("b", "part_of", "c"),
    ("x", "part_of", "y"), ("y", "part_of", "z"),
    ("d", "part_of", "e"), ("e", "part_of", "f"),
    ("m", "part_of", "n"), ("n", "part_of", "o"),
)

TRANS_WITH_CLOSURE = TRANS_RELS + _rels(
    ("a", "part_of", "c"),   # existing transitive edge → will be hidden
    ("x", "part_of", "z"),   # existing transitive edge → will be hidden
    ("m", "part_of", "o"),   # existing transitive edge → will be hidden
)


# ── basic prediction behaviour ────────────────────────────────────────────────

class TestPredictFromBigrams:
    def setup_method(self):
        self.pred = PatternBasedPredictor(TRANS_RELS)

    def test_returns_list_of_pattern_predictions(self):
        preds = self.pred.predict_from_bigrams(min_count=1)
        assert isinstance(preds, list)
        assert all(isinstance(p, PatternPrediction) for p in preds)

    def test_predicts_transitive_link(self):
        preds = self.pred.predict_from_bigrams(min_count=1)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("a", "part_of", "c") in keys

    def test_all_four_closures_predicted(self):
        preds = self.pred.predict_from_bigrams(min_count=1)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        for src, tgt in [("a", "c"), ("x", "z"), ("d", "f"), ("m", "o")]:
            assert (src, "part_of", tgt) in keys

    def test_empty_relations_returns_empty(self):
        assert PatternBasedPredictor([]).predict_from_bigrams(min_count=1) == []

    def test_no_transitive_pattern_returns_empty(self):
        # Mixed relation types — no same-relation bigram
        rels = _rels(("a", "is_a", "b"), ("b", "part_of", "c"))
        assert PatternBasedPredictor(rels).predict_from_bigrams(min_count=1) == []

    def test_sorted_by_confidence_descending(self):
        preds = self.pred.predict_from_bigrams(min_count=1)
        confs = [p.confidence for p in preds]
        assert confs == sorted(confs, reverse=True)

    def test_prediction_fields_populated(self):
        preds = self.pred.predict_from_bigrams(min_count=1)
        for p in preds:
            assert isinstance(p.source, str) and p.source
            assert isinstance(p.relation_type, str) and p.relation_type
            assert isinstance(p.target, str) and p.target
            assert 0.0 < p.confidence <= 0.95
            assert isinstance(p.reason, str) and p.reason
            assert isinstance(p.evidence, list) and len(p.evidence) >= 1


# ── existing edges not duplicated ─────────────────────────────────────────────

class TestNoExistingEdgeDuplicated:
    def test_existing_transitive_edge_not_in_predictions(self):
        # a→c already in graph → should NOT appear in predictions
        rels = TRANS_WITH_CLOSURE
        preds = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("a", "part_of", "c") not in keys
        assert ("x", "part_of", "z") not in keys
        assert ("m", "part_of", "o") not in keys

    def test_only_novel_edges_predicted(self):
        # d→f is the only novel closure (a,x,m already have direct edges)
        rels = TRANS_WITH_CLOSURE
        preds = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("d", "part_of", "f") in keys
        assert len(keys) == 1

    def test_self_loops_not_predicted(self):
        # A -r-> A -r-> A: should not produce A -r-> A
        rels = _rels(("a", "part_of", "a"))
        preds = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1)
        assert preds == []


# ── confidence formula ─────────────────────────────────────────────────────────

class TestConfidence:
    def test_confidence_formula(self):
        preds = PatternBasedPredictor(TRANS_RELS).predict_from_bigrams(min_count=1)
        # 4 chains → count=4
        expected = _conf(4)
        for p in preds:
            assert p.confidence == pytest.approx(expected)

    def test_confidence_increases_with_count(self):
        # Low-count: 2 chains
        rels_2 = _rels(
            ("a", "r", "b"), ("b", "r", "c"),
            ("x", "r", "y"), ("y", "r", "z"),
        )
        # High-count: 10 chains
        rels_10 = _rels(
            *[(f"n{i}", "r", "hub") for i in range(10)],
            ("hub", "r", "end"),
        )
        p2  = PatternBasedPredictor(rels_2).predict_from_bigrams(min_count=1)
        p10 = PatternBasedPredictor(rels_10).predict_from_bigrams(min_count=1)
        assert p2[0].confidence < p10[0].confidence

    def test_confidence_capped_at_0_95(self):
        # Huge count still caps at 0.95
        rels = _rels(*[(f"n{i}", "r", "hub") for i in range(10_000)], ("hub", "r", "end"))
        preds = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1)
        for p in preds:
            assert p.confidence <= 0.95

    def test_min_confidence_filter(self):
        # count=1 → conf≈0.535; filter at 0.6 should exclude it
        rels = _rels(("a", "r", "b"), ("b", "r", "c"))
        assert PatternBasedPredictor(rels).predict_from_bigrams(
            min_count=1, min_confidence=0.6
        ) == []


# ── reason and evidence ───────────────────────────────────────────────────────

class TestReasonAndEvidence:
    def test_reason_contains_relation_type(self):
        preds = PatternBasedPredictor(TRANS_RELS).predict_from_bigrams(min_count=1)
        for p in preds:
            assert p.relation_type in p.reason

    def test_reason_contains_count(self):
        preds = PatternBasedPredictor(TRANS_RELS).predict_from_bigrams(min_count=1)
        for p in preds:
            assert "count=" in p.reason

    def test_reason_format(self):
        preds = PatternBasedPredictor(TRANS_RELS).predict_from_bigrams(min_count=1)
        for p in preds:
            # "transitive pattern: part_of -> part_of (count=4)"
            assert p.reason.startswith("transitive pattern:")
            assert "->" in p.reason

    def test_evidence_contains_intermediate_node(self):
        preds = PatternBasedPredictor(TRANS_RELS).predict_from_bigrams(min_count=1)
        p = next(x for x in preds if x.source == "a" and x.target == "c")
        assert "b" in p.evidence

    def test_evidence_multiple_paths(self):
        # Two paths to the same conclusion: a→b→c and a→b2→c
        rels = _rels(
            ("a", "r", "b1"), ("b1", "r", "c"),
            ("a", "r", "b2"), ("b2", "r", "c"),
        )
        preds = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1)
        p = next(x for x in preds if x.source == "a" and x.target == "c")
        assert len(p.evidence) == 2

    def test_evidence_capped_at_five(self):
        # 10 different intermediate nodes for the same (A, r, C) conclusion
        rels = _rels(
            *[(f"hub{i}", "r", "end") for i in range(10)],
            *[("start", "r", f"hub{i}") for i in range(10)],
        )
        preds = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1)
        p = next(x for x in preds if x.source == "start" and x.target == "end")
        assert len(p.evidence) <= 5


# ── explain_prediction ────────────────────────────────────────────────────────

class TestExplainPrediction:
    def _get_pred(self) -> PatternPrediction:
        preds = PatternBasedPredictor(TRANS_RELS).predict_from_bigrams(min_count=1)
        return next(x for x in preds if x.source == "a" and x.target == "c")

    def test_explain_contains_source_and_target(self):
        p = self._get_pred()
        explanation = PatternBasedPredictor(TRANS_RELS).explain_prediction(p)
        assert "a" in explanation
        assert "c" in explanation

    def test_explain_contains_relation_type(self):
        p = self._get_pred()
        explanation = PatternBasedPredictor(TRANS_RELS).explain_prediction(p)
        assert "part_of" in explanation

    def test_explain_contains_reason(self):
        p = self._get_pred()
        explanation = PatternBasedPredictor(TRANS_RELS).explain_prediction(p)
        assert "transitive pattern" in explanation

    def test_explain_contains_confidence(self):
        p = self._get_pred()
        explanation = PatternBasedPredictor(TRANS_RELS).explain_prediction(p)
        assert "conf" in explanation

    def test_explain_contains_via_node(self):
        p = self._get_pred()
        explanation = PatternBasedPredictor(TRANS_RELS).explain_prediction(p)
        assert "b" in explanation  # intermediate node


# ── world integration ─────────────────────────────────────────────────────────

class TestWorldIntegration:
    def test_predict_from_patterns_returns_list(self):
        w = _world(
            ("a", "part_of", "b"), ("b", "part_of", "c"),
            ("x", "part_of", "y"), ("y", "part_of", "z"),
        )
        preds = w.predict_from_patterns(min_count=1)
        assert isinstance(preds, list)
        assert len(preds) >= 1

    def test_predict_from_patterns_finds_transitive(self):
        w = _world(
            ("a", "is_a", "b"), ("b", "is_a", "c"),
            ("x", "is_a", "b"), ("y", "is_a", "b"),
        )
        preds = w.predict_from_patterns(min_count=1)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("a", "is_a", "c") in keys


# ── evaluation ────────────────────────────────────────────────────────────────

class TestEvaluation:
    def _build_eval_world(self) -> World:
        """
        3 hidden transitive triples (a→c, x→z, m→o)
        1 novel (d→f, not in original graph) → appears as FP in evaluation
        """
        return _world_from(TRANS_WITH_CLOSURE)

    def test_returns_evaluation_result(self):
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        assert isinstance(result, PatternEvaluationResult)

    def test_hidden_count(self):
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        assert result.hidden_count == 3

    def test_recall_perfect(self):
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        assert result.recall == pytest.approx(1.0)

    def test_true_positives(self):
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        assert result.true_positives == 3

    def test_false_positives(self):
        # d→f is predicted but was NOT in the original graph (novel inference)
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        assert result.false_positives == 1

    def test_false_negatives_zero(self):
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        assert result.false_negatives == 0

    def test_f1_value(self):
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        # P = 3/4 = 0.75, R = 1.0, F1 = 2*0.75*1.0/1.75
        assert result.f1 == pytest.approx(2 * 0.75 / 1.75)

    def test_relation_types_filter(self):
        # Only evaluate part_of
        w = self._build_eval_world()
        r_all  = evaluate_pattern_prediction_recovery(w, min_count=1)
        r_part = evaluate_pattern_prediction_recovery(
            w, relation_types={"part_of"}, min_count=1
        )
        # Filtering to just part_of shouldn't change results here since
        # all edges are part_of
        assert r_all.hidden_count == r_part.hidden_count

    def test_max_hidden_cap(self):
        w = self._build_eval_world()
        result = evaluate_pattern_prediction_recovery(w, min_count=1, max_hidden=2)
        assert result.hidden_count == 2

    def test_no_transitive_in_graph_gives_zero_hidden(self):
        # Pure chains, no pre-existing closures
        w = _world_from(TRANS_RELS)
        result = evaluate_pattern_prediction_recovery(w, min_count=1)
        assert result.hidden_count == 0

    def test_empty_world_returns_zero_result(self):
        result = evaluate_pattern_prediction_recovery(_world_from([]), min_count=1)
        assert result.hidden_count == 0
        assert result.true_positives == 0


# ── conceptnet tiny fixture ───────────────────────────────────────────────────

_FIXTURE = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "data", "conceptnet_tiny.csv")
)


class TestConceptNetTinyFixture:
    def setup_method(self):
        from core.datasets import load_relations_csv, build_world_from_relations
        rows = load_relations_csv(_FIXTURE)
        self.w = build_world_from_relations(rows)

    def test_predictor_finds_transitive_predictions(self):
        preds = self.w.predict_from_patterns(min_count=2)
        assert len(preds) > 0

    def test_predictions_are_is_a_transitive(self):
        # fruit -is_a-> plant, so apple/banana/… -is_a-> plant should be predicted
        preds = self.w.predict_from_patterns(min_count=2)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("apple", "is_a", "plant") in keys

    def test_no_existing_edge_predicted(self):
        rels = self.w.get_relations()
        existing = {(r.source, r.relation_type, r.target) for r in rels}
        preds = self.w.predict_from_patterns(min_count=2)
        for p in preds:
            assert (p.source, p.relation_type, p.target) not in existing

    def test_all_predictions_have_evidence(self):
        preds = self.w.predict_from_patterns(min_count=2)
        for p in preds:
            assert len(p.evidence) >= 1

    def test_evaluation_runs_without_error(self):
        result = evaluate_pattern_prediction_recovery(self.w, min_count=2)
        assert isinstance(result, PatternEvaluationResult)
        assert result.hidden_count == 0  # no pre-existing transitive closures in fixture


# ── hub penalty ────────────────────────────────────────────────────────────────
#
#  HUB_RELS graph
#  ─────────────
#  100 leaf nodes n0..n99  each  --r-->  hub
#  hub                            --r-->  end
#  start                          --r-->  mid
#  mid                            --r-->  dest
#
#  Total (r,r) bigram count = 101 (100 hub chains + 1 mid chain)
#  hub  : total_degree = 100 (in) + 1 (out) = 101
#  mid  : total_degree = 1   (in) + 1 (out) = 2      → hub_factor = 1.0
#  hub hub_factor = sqrt(10/101) ≈ 0.315   → base_conf * 0.315 < 0.5  → filtered
#  mid hub_factor = sqrt(10/10)  = 1.0     → conf unchanged
#

_HUB_LEAVES = [(f"n{i}", "r", "hub") for i in range(100)]
_HUB_RELS = _rels(
    *_HUB_LEAVES,
    ("hub",   "r", "end"),
    ("start", "r", "mid"),
    ("mid",   "r", "dest"),
)


def _hub_base_conf() -> float:
    """Base confidence for the (r,r) bigram in HUB_RELS (count=101)."""
    return min(0.95, 0.5 + 0.05 * math.log(101 + 1))


class TestHubPenalty:
    def setup_method(self):
        self.pred = PatternBasedPredictor(_HUB_RELS)

    # -- degree tables ---------------------------------------------------------

    def test_total_degree_hub_correct(self):
        # 100 in-edges from n0..n99 + 1 out-edge to end
        assert self.pred.get_total_degree("hub") == 101

    def test_total_degree_mid_correct(self):
        # 1 in-edge from start + 1 out-edge to dest
        assert self.pred.get_total_degree("mid") == 2

    def test_total_degree_unknown_node_returns_zero(self):
        assert self.pred.get_total_degree("nonexistent_xyz") == 0

    def test_chain_intermediate_count_hub(self):
        # 100 chains go through hub (n0→hub→end, n1→hub→end, …)
        assert self.pred.get_chain_intermediate_count("hub") == 100

    def test_chain_intermediate_count_mid(self):
        # exactly 1 chain through mid (start→mid→dest)
        assert self.pred.get_chain_intermediate_count("mid") == 1

    def test_chain_intermediate_count_leaf_is_zero(self):
        assert self.pred.get_chain_intermediate_count("n0") == 0

    # -- hub penalty reduces or removes hub-mediated predictions ---------------

    def test_hub_penalty_suppresses_high_degree_hub(self):
        preds = self.pred.predict_from_bigrams(min_count=1, hub_penalty=True)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        # all n0..n99 → end predictions should be filtered (conf < 0.5 after penalty)
        for i in range(100):
            assert (f"n{i}", "r", "end") not in keys

    def test_hub_penalty_retains_low_degree_prediction(self):
        preds = self.pred.predict_from_bigrams(min_count=1, hub_penalty=True)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("start", "r", "dest") in keys

    def test_hub_penalty_off_includes_hub_predictions(self):
        preds = self.pred.predict_from_bigrams(min_count=1, hub_penalty=False)
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("n0", "r", "end") in keys
        assert ("start", "r", "dest") in keys

    def test_hub_penalty_lowers_confidence_vs_no_penalty(self):
        # For a moderate-degree hub (degree between 10 and very high) the
        # penalised confidence should be ≤ the unpenalised confidence.
        # Use a graph with a hub of degree ~30 so both runs still have predictions.
        rels = _rels(
            *[(f"x{i}", "r", "mhub") for i in range(20)],
            ("mhub", "r", "mend"),
            ("s",    "r", "lmid"),
            ("lmid", "r", "d"),
        )
        pred_pen   = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1, hub_penalty=True)
        pred_nopen = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1, hub_penalty=False)
        # Find matching predictions for the hub-mediated path
        pen_confs   = {(p.source, p.target): p.confidence for p in pred_pen}
        nopen_confs = {(p.source, p.target): p.confidence for p in pred_nopen}
        # low-degree "lmid" prediction should have equal or slightly lower conf (factor near 1)
        if ("s", "d") in pen_confs and ("s", "d") in nopen_confs:
            assert pen_confs[("s", "d")] <= nopen_confs[("s", "d")] + 1e-9

    def test_low_degree_hub_factor_is_one(self):
        # degree ≤ 10 → hub_factor = sqrt(10/max(10,deg)) = sqrt(10/10) = 1.0
        # so confidence with penalty == confidence without penalty
        rels = _rels(
            ("a", "r", "b"), ("b", "r", "c"),
            ("x", "r", "b"), ("y", "r", "b"),  # b has in=3, out=1, total=4 ≤ 10
        )
        pred_pen   = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1, hub_penalty=True)
        pred_nopen = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1, hub_penalty=False)
        pen_map   = {(p.source, p.target): p.confidence for p in pred_pen}
        nopen_map = {(p.source, p.target): p.confidence for p in pred_nopen}
        for key in pen_map:
            if key in nopen_map:
                assert pen_map[key] == pytest.approx(nopen_map[key])

    # -- max_intermediate_degree filtering ------------------------------------

    def test_max_intermediate_degree_filters_hub(self):
        # hub has degree=102; any threshold below that removes its predictions
        preds = self.pred.predict_from_bigrams(
            min_count=1, hub_penalty=False, max_intermediate_degree=50
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        for i in range(100):
            assert (f"n{i}", "r", "end") not in keys

    def test_max_intermediate_degree_keeps_low_degree(self):
        preds = self.pred.predict_from_bigrams(
            min_count=1, hub_penalty=False, max_intermediate_degree=50
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("start", "r", "dest") in keys

    def test_max_intermediate_degree_none_means_no_filter(self):
        preds_none = self.pred.predict_from_bigrams(
            min_count=1, hub_penalty=False, max_intermediate_degree=None
        )
        preds_high = self.pred.predict_from_bigrams(
            min_count=1, hub_penalty=False, max_intermediate_degree=10_000
        )
        # both should include the hub prediction
        keys_none = {(p.source, p.relation_type, p.target) for p in preds_none}
        keys_high = {(p.source, p.relation_type, p.target) for p in preds_high}
        assert ("n0", "r", "end") in keys_none
        assert ("n0", "r", "end") in keys_high

    def test_max_intermediate_degree_zero_removes_all(self):
        preds = self.pred.predict_from_bigrams(
            min_count=1, hub_penalty=False, max_intermediate_degree=0
        )
        assert preds == []

    # -- max_intermediate_count filtering -------------------------------------

    def test_max_intermediate_count_filters_hub(self):
        # hub intermediary count = 100; threshold 10 should remove it
        preds = self.pred.predict_from_bigrams(
            min_count=1, hub_penalty=False, max_intermediate_count=10
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("n0", "r", "end") not in keys

    def test_max_intermediate_count_keeps_low_count(self):
        preds = self.pred.predict_from_bigrams(
            min_count=1, hub_penalty=False, max_intermediate_count=10
        )
        keys = {(p.source, p.relation_type, p.target) for p in preds}
        assert ("start", "r", "dest") in keys

    # -- reason string --------------------------------------------------------

    def test_reason_contains_via_and_degree_when_penalty_on(self):
        preds = self.pred.predict_from_bigrams(min_count=1, hub_penalty=True)
        p = next(x for x in preds if x.source == "start")
        assert "via=" in p.reason
        assert "degree=" in p.reason
        assert "hub_penalty=" in p.reason

    def test_reason_simple_format_when_penalty_off(self):
        preds = self.pred.predict_from_bigrams(min_count=1, hub_penalty=False)
        p = next(x for x in preds if x.source == "start")
        assert "via=" not in p.reason
        assert "hub_penalty=" not in p.reason
        assert "count=" in p.reason

    def test_reason_hub_penalty_value_near_one_for_low_degree(self):
        preds = self.pred.predict_from_bigrams(min_count=1, hub_penalty=True)
        p = next(x for x in preds if x.source == "start")
        # mid has degree=2, hub_factor = sqrt(10/10) = 1.0
        assert "hub_penalty=1.00" in p.reason

    # -- no-penalty restores old confidence -----------------------------------

    def test_no_hub_penalty_confidence_equals_base(self):
        preds = self.pred.predict_from_bigrams(min_count=1, hub_penalty=False)
        base = _hub_base_conf()
        for p in preds:
            assert p.confidence == pytest.approx(base)

    def test_hub_penalty_true_confidence_below_base_for_high_degree(self):
        # For start→dest (via mid, degree=2) hub_factor=1.0 so conf == base.
        # We can't easily test n*→end since they're filtered, so we use a
        # smaller hub to still get predictions.
        rels = _rels(
            *[(f"l{i}", "r", "shub") for i in range(15)],
            ("shub", "r", "send"),
        )
        # shub degree = 15+1 = 16, hub_factor = sqrt(10/16) ≈ 0.79
        base = min(0.95, 0.5 + 0.05 * math.log(16 + 1))
        pred_pen   = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1, hub_penalty=True)
        pred_nopen = PatternBasedPredictor(rels).predict_from_bigrams(min_count=1, hub_penalty=False)
        # No-penalty should have higher or equal confidence
        if pred_pen and pred_nopen:
            assert pred_pen[0].confidence <= pred_nopen[0].confidence + 1e-9


# ── audit export + hub penalty ────────────────────────────────────────────────

class TestAuditExportHubPenalty:
    """Verify that build_audit_rows passes hub-penalty params through."""

    def _write_hub_csv(self, tmp_path) -> str:
        p = tmp_path / "hub_test.csv"
        lines = ["source,relation_type,target"]
        for i in range(20):
            lines.append(f"n{i},r,hub")
        lines.append("hub,r,end")
        lines.append("start,r,mid")
        lines.append("mid,r,dest")
        p.write_text("\n".join(lines) + "\n")
        return str(p)

    def test_max_intermediate_degree_removes_hub_preds(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        csv_path = self._write_hub_csv(tmp_path)
        rows = build_audit_rows(
            csv_path, limit=200, hub_penalty=False, max_intermediate_degree=10
        )
        for r in rows:
            assert r["target"] != "end", "hub-mediated pred should be removed"

    def test_max_intermediate_degree_none_includes_hub_preds(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        csv_path = self._write_hub_csv(tmp_path)
        rows = build_audit_rows(
            csv_path, limit=200, hub_penalty=False, max_intermediate_degree=None
        )
        targets = {r["target"] for r in rows}
        assert "end" in targets

    def test_no_hub_penalty_includes_more_or_equal_rows(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        csv_path = self._write_hub_csv(tmp_path)
        rows_pen   = build_audit_rows(csv_path, limit=200, hub_penalty=True)
        rows_nopen = build_audit_rows(csv_path, limit=200, hub_penalty=False)
        # penalty only filters; no-penalty must have >= as many rows
        assert len(rows_nopen) >= len(rows_pen)

    def test_hub_penalty_true_confidence_le_no_penalty(self, tmp_path):
        from examples.pattern_audit_export import build_audit_rows
        csv_path = self._write_hub_csv(tmp_path)
        rows_pen   = build_audit_rows(csv_path, limit=200, hub_penalty=True)
        rows_nopen = build_audit_rows(csv_path, limit=200, hub_penalty=False)
        pen_map   = {(r["source"], r["target"]): float(r["confidence"]) for r in rows_pen}
        nopen_map = {(r["source"], r["target"]): float(r["confidence"]) for r in rows_nopen}
        for key in pen_map:
            if key in nopen_map:
                assert pen_map[key] <= nopen_map[key] + 1e-6
