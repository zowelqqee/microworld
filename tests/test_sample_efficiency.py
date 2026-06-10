import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from core.sample_efficiency import (
    SampleEfficiencyExperiment,
    generate_instances,
    FAMILY_SPECS,
    MODES,
    DEFAULT_BUDGETS,
)


# ── deterministic world generation ─────────────────────────────────────────────

class TestDeterministicGeneration:
    def test_generate_is_deterministic(self):
        a = generate_instances(6)
        b = generate_instances(6)
        assert a == b

    def test_instance_count(self):
        insts = generate_instances(6)
        assert len(insts) == len(FAMILY_SPECS) * 6

    def test_shared_connector_per_family(self):
        insts = generate_instances(6)
        seed = [i for i in insts if i.family == "seed"]
        assert all(i.connector == "seed" for i in seed)
        pit = [i for i in insts if i.family == "pit"]
        assert all(i.connector == "pit" for i in pit)

    def test_unique_tree_and_fruit_names(self):
        insts = generate_instances(6)
        trees = [i.tree for i in insts]
        fruits = [i.fruit for i in insts]
        assert len(trees) == len(set(trees))
        assert len(fruits) == len(set(fruits))

    def test_experiment_world_deterministic(self):
        e1 = SampleEfficiencyExperiment(n_per_family=6)
        e2 = SampleEfficiencyExperiment(n_per_family=6)
        w1 = e1.build_world(0.6)
        w2 = e2.build_world(0.6)
        rels1 = sorted((r.source, r.relation_type, r.target) for r in w1.get_relations())
        rels2 = sorted((r.source, r.relation_type, r.target) for r in w2.get_relations())
        assert rels1 == rels2

    def test_requires_min_two_per_family(self):
        with pytest.raises(ValueError):
            SampleEfficiencyExperiment(n_per_family=1)


# ── deterministic budget split ─────────────────────────────────────────────────

class TestBudgetSplit:
    def setup_method(self):
        self.exp = SampleEfficiencyExperiment(n_per_family=6)

    def test_pool_excludes_held_out(self):
        held = {(i.family, i.index) for i in self.exp.held_out}
        for inst in self.exp.pool:
            assert (inst.family, inst.index) not in held

    def test_held_out_one_per_family(self):
        assert len(self.exp.held_out) == len(FAMILY_SPECS)
        families = {i.family for i in self.exp.held_out}
        assert families == {f[0] for f in FAMILY_SPECS}

    def test_budget_split_deterministic(self):
        a = self.exp.budget_pool(0.4)
        b = self.exp.budget_pool(0.4)
        assert a == b

    def test_budget_monotonic_prefix(self):
        """Lower budgets are prefixes of higher budgets."""
        small = self.exp.budget_pool(0.4)
        large = self.exp.budget_pool(0.8)
        assert large[: len(small)] == small

    def test_budget_size_scales(self):
        sizes = [len(self.exp.budget_pool(b)) for b in DEFAULT_BUDGETS]
        assert sizes == sorted(sizes)
        assert sizes[-1] == len(self.exp.pool)

    def test_budget_bounds_validated(self):
        with pytest.raises(ValueError):
            self.exp.budget_pool(1.5)

    def test_hidden_relations_fixed(self):
        """Held-out closing edges do not depend on budget."""
        h1 = self.exp.hidden_relations()
        h2 = self.exp.hidden_relations()
        keys1 = {(r.source, r.relation_type, r.target) for r in h1}
        keys2 = {(r.source, r.relation_type, r.target) for r in h2}
        assert keys1 == keys2
        assert len(keys1) == len(FAMILY_SPECS)

    def test_hidden_edges_absent_from_world(self):
        """The held-out closing edges must never be in any budgeted world."""
        hidden = {(r.source, r.relation_type, r.target) for r in self.exp.hidden_relations()}
        for budget in DEFAULT_BUDGETS:
            w = self.exp.build_world(budget)
            world_keys = {(r.source, r.relation_type, r.target) for r in w.get_relations()}
            assert hidden.isdisjoint(world_keys)


# ── mode isolation ─────────────────────────────────────────────────────────────

class TestModeIsolation:
    def setup_method(self):
        self.exp = SampleEfficiencyExperiment(n_per_family=6)

    def test_fresh_world_each_mode(self):
        """run_mode builds its own world; running one mode can't affect another."""
        majority_before = self.exp.run_mode(1.0, "majority_only")
        self.exp.run_mode(1.0, "structural_similarity")
        self.exp.run_mode(1.0, "hybrid_structural")
        majority_after = self.exp.run_mode(1.0, "majority_only")
        assert majority_before.f1 == pytest.approx(majority_after.f1)
        assert majority_before.true_positives == majority_after.true_positives

    def test_mode_order_independence(self):
        forward = {m: self.exp.run_mode(0.6, m).f1 for m in MODES}
        backward = {m: self.exp.run_mode(0.6, m).f1 for m in reversed(MODES)}
        for m in MODES:
            assert forward[m] == pytest.approx(backward[m])

    def test_same_hidden_set_all_modes(self):
        """Every mode is scored against the identical held-out count."""
        for budget in DEFAULT_BUDGETS:
            totals = {self.exp.run_mode(budget, m).total_hidden for m in MODES}
            assert totals == {len(FAMILY_SPECS)}

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            self.exp.run_mode(0.5, "nonsense_mode")


# ── no manual similarity oracle ────────────────────────────────────────────────

class TestNoManualSimilarity:
    def test_structural_world_has_no_hand_injected_edges(self):
        """
        The structural mode builds its similarity graph only from
        add_structural_similarities — never add_similarity. Verify every score
        is a Jaccard value (in (0,1)) and none is a hand-picked oracle constant.
        """
        exp = SampleEfficiencyExperiment(n_per_family=6)
        w = exp.build_world(1.0)
        w.add_structural_similarities(min_score=0.5)
        scores = list(w._similarity_graph._scores.values())
        assert scores  # something was discovered
        # Structural Jaccard values for these profiles are rational fractions,
        # never the 0.9 oracle constant used in earlier hand-fed demos.
        for s in scores:
            assert 0.0 < s <= 1.0
            assert s != pytest.approx(0.9)

    def test_no_add_similarity_called_anywhere(self):
        """A freshly built world starts with an empty similarity graph."""
        exp = SampleEfficiencyExperiment(n_per_family=6)
        w = exp.build_world(1.0)
        assert w._similarity_graph._scores == {}


# ── F1 table is produced ───────────────────────────────────────────────────────

class TestTableProduced:
    def test_run_returns_rows_for_every_cell(self):
        exp = SampleEfficiencyExperiment(n_per_family=6)
        rows = exp.run()
        assert len(rows) == len(DEFAULT_BUDGETS) * len(MODES)

    def test_row_fields_present(self):
        exp = SampleEfficiencyExperiment(n_per_family=6)
        rows = exp.run()
        for r in rows:
            for key in ("budget", "mode", "precision", "recall", "f1", "tp", "fp", "fn"):
                assert key in r
            assert 0.0 <= r["precision"] <= 1.0
            assert 0.0 <= r["recall"] <= 1.0
            assert 0.0 <= r["f1"] <= 1.0

    def test_custom_budgets_and_modes(self):
        exp = SampleEfficiencyExperiment(n_per_family=6)
        rows = exp.run(budgets=[0.5, 1.0], modes=["majority_only", "hybrid_structural"])
        assert len(rows) == 4


# ── the main claim: sample efficiency ──────────────────────────────────────────

class TestSampleEfficiencyClaim:
    def setup_method(self):
        self.exp = SampleEfficiencyExperiment(n_per_family=6)
        self.rows = self.exp.run()

    def _f1(self, budget, mode):
        for r in self.rows:
            if r["budget"] == budget and r["mode"] == mode:
                return r["f1"]
        raise KeyError((budget, mode))

    def test_hybrid_never_worse_than_majority(self):
        for budget in DEFAULT_BUDGETS:
            assert self._f1(budget, "hybrid_structural") >= self._f1(budget, "majority_only")

    def test_at_least_one_lower_budget_hybrid_geq_majority(self):
        # 0.2 is the lowest budget; hybrid >= majority must hold there.
        assert self._f1(0.2, "hybrid_structural") >= self._f1(0.2, "majority_only")

    def test_structural_improves_over_majority_somewhere(self):
        gains = [
            self._f1(b, "structural_similarity") - self._f1(b, "majority_only")
            for b in DEFAULT_BUDGETS
        ]
        assert max(gains) > 0.0

    def test_majority_is_flat(self):
        """Majority cannot improve with more data — one global connector only."""
        f1s = [self._f1(b, "majority_only") for b in DEFAULT_BUDGETS]
        assert max(f1s) - min(f1s) == pytest.approx(0.0)

    def test_structural_curve_is_monotonic(self):
        f1s = [self._f1(b, "structural_similarity") for b in DEFAULT_BUDGETS]
        assert f1s == sorted(f1s)

    def test_hybrid_reaches_perfect_recovery(self):
        assert self._f1(1.0, "hybrid_structural") == pytest.approx(1.0)

    def test_concept_matches_structural(self):
        """Concept membership and structural similarity recover the same families."""
        for budget in DEFAULT_BUDGETS:
            assert self._f1(budget, "concept_only") == pytest.approx(
                self._f1(budget, "structural_similarity")
            )
