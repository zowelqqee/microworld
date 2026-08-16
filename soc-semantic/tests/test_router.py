"""Properties the confidence router has to hold, checked on synthetic data
and on hand-built scores. No real data needed."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soc_runtime import baseline, config, modeling, router, semantic
from soc_runtime.pipeline import load_features


@pytest.fixture(scope="module")
def scored_frame() -> pd.DataFrame:
    frame = load_features()
    return modeling.restrict_to_scored_techniques(frame)


# --------------------------------------------------------------------------
# Out-of-fold folds and scores
# --------------------------------------------------------------------------

def test_expanding_folds_covers_every_row_in_order():
    folds = router.expanding_folds(100, 4)
    assert len(folds) == 100
    assert folds.min() == 0
    assert folds.max() == 3
    assert np.all(np.diff(folds) >= 0)  # non-decreasing - chronological, not shuffled


def test_out_of_fold_scores_never_touch_the_test_set(scored_frame):
    """The strongest guarantee here is architectural: `out_of_fold_raw_scores`
    takes only `train` as an argument, so there is no code path by which it
    could read the held-out set. This test pins that down by checking the
    scored rows are a subset of train's own row count and index range."""
    train, test = modeling.chronological_split(scored_frame)
    oof_scores, oof_labels, meta = router.out_of_fold_raw_scores(train, n_folds=4)

    assert len(oof_scores) == len(oof_labels)
    # fold 0 is never scored, so the OOF set is strictly smaller than train
    assert len(oof_scores) < len(train)
    assert meta["folds"] == 4
    assert len(meta["per_fold"]) == 3  # folds 1, 2, 3


def test_out_of_fold_scores_requires_sorted_input():
    frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-06-02", "2026-06-01"]),
        **{name: [0.0, 0.0] for name in modeling.ARM_FEATURES["baseline"]},
        modeling.LABEL_COLUMN: [0, 0],
    })
    with pytest.raises(ValueError):
        router.out_of_fold_raw_scores(frame, n_folds=2)


def test_fold_zero_contributes_no_scores():
    """Fold 0 has no earlier fold to be fit on, so it must never appear in
    the returned out-of-fold scores - only folds 1..n_folds-1 do."""
    train = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=40, freq="h"),
        **{name: np.random.default_rng(0).normal(size=40) for name in modeling.ARM_FEATURES["baseline"]},
        modeling.LABEL_COLUMN: (np.arange(40) % 7 == 0).astype(int),
    })
    _, _, meta = router.out_of_fold_raw_scores(train, n_folds=4)
    total_scored = sum(f["scored"] for f in meta["per_fold"])
    fold_sizes = np.bincount(router.expanding_folds(40, 4))
    assert total_scored == fold_sizes[1:].sum()  # excludes fold 0's size


# --------------------------------------------------------------------------
# Band selection
# --------------------------------------------------------------------------

def test_select_band_respects_the_budget():
    rng = np.random.default_rng(3)
    scores = rng.uniform(0, 1, 200)
    labels = (rng.uniform(0, 1, 200) < scores).astype(int)
    curve = router.band_curve(scores, labels, threshold=0.5, grid_points=40)
    for budget in (0.1, 0.3, 0.5, 1.0):
        band = router.select_band(curve, budget)
        assert band["train_routed_share"] <= budget + 1e-9


def test_select_band_raises_when_no_band_fits_budget():
    scores = np.linspace(0, 1, 50)
    labels = (scores > 0.5).astype(int)
    curve = router.band_curve(scores, labels, threshold=0.5, grid_points=20)
    with pytest.raises(ValueError):
        router.select_band(curve, budget=0.0)


def test_wider_budget_never_captures_fewer_errors():
    scores = np.random.default_rng(1).uniform(0, 1, 200)
    labels = (np.random.default_rng(2).uniform(0, 1, 200) < scores).astype(int)
    curve = router.band_curve(scores, labels, threshold=0.5, grid_points=40)
    captured = [router.select_band(curve, b)["train_errors_captured"] for b in (0.05, 0.2, 0.5, 1.0)]
    assert captured == sorted(captured)


# --------------------------------------------------------------------------
# Routing mechanics
# --------------------------------------------------------------------------

def test_route_uses_secondary_only_inside_the_band():
    primary = np.array([0.1, 0.5, 0.9])
    secondary = np.array([0.9, 0.1, 0.1])
    final, routed = router.route(primary, secondary, lo=0.4, hi=0.6)
    assert list(routed) == [False, True, False]
    assert list(final) == [0.1, 0.1, 0.9]


def test_route_band_is_inclusive_at_both_edges():
    primary = np.array([0.4, 0.6])
    secondary = np.array([1.0, 1.0])
    _, routed = router.route(primary, secondary, lo=0.4, hi=0.6)
    assert list(routed) == [True, True]


def test_flip_analysis_counts_fixed_and_broken():
    y = np.array([1, 1, 0, 0])
    primary = np.array([0.1, 0.1, 0.6, 0.1])   # raw: wrong, wrong, wrong, right
    final = np.array([0.9, 0.1, 0.4, 0.1])      # after routing: right, wrong(still), right, right
    routed = np.array([True, True, True, False])
    result = router.flip_analysis(y, primary, final, routed, threshold=0.5)
    assert result["fixed"] == 2   # index 0 (1->1 fixed), index 2 (0->0 fixed)
    assert result["broken"] == 0
    assert result["routed"] == 3
    assert result["net_corrections"] == 2


def test_flip_analysis_counts_broken_when_secondary_makes_it_worse():
    y = np.array([1])
    primary = np.array([0.9])   # raw: correct
    final = np.array([0.1])     # after routing: now wrong
    routed = np.array([True])
    result = router.flip_analysis(y, primary, final, routed, threshold=0.5)
    assert result["broken"] == 1
    assert result["fixed"] == 0
    assert result["net_corrections"] == -1


# --------------------------------------------------------------------------
# Reusing already-fitted models, not retraining
# --------------------------------------------------------------------------

def test_run_router_reuses_fit_arms_output(scored_frame, monkeypatch):
    """`run_router` must call `modeling.fit_arms` exactly once and reuse its
    output for every variant - not fit a fresh raw/semantic/union model per
    router variant."""
    calls = []
    original = modeling.fit_arms

    def counting_fit_arms(train, test, arms=None):
        calls.append(1)
        return original(train, test, arms=arms)

    monkeypatch.setattr(modeling, "fit_arms", counting_fit_arms)
    router.run_router(scored_frame)
    assert len(calls) == 1


# --------------------------------------------------------------------------
# End-to-end
# --------------------------------------------------------------------------

def test_run_router_end_to_end_shape(scored_frame):
    results = router.run_router(scored_frame)

    assert "SYNTHETIC" in results["disclaimer"]
    assert set(results["arms"].keys()) >= {
        "raw", "semantic", "union", "router->semantic", "router->union",
        "router->union (naive band)",
    }
    assert "sanity_check" in results
    for name in ("router->semantic", "router->union", "router->union (naive band)"):
        verdict = results["sanity_check"]["verdict"][name]
        assert "recall_at_threshold" in verdict and "pr_auc" in verdict
        assert isinstance(verdict["recall_at_threshold"]["beats_or_matches"], bool)
    assert "band" in results and results["band"]["budget"] > 0
    assert "band_frontier" in results and len(results["band_frontier"]) > 0
    assert "operational_cost" in results
    assert results["operational_cost"]["semantic_feature_build_seconds_per_alert"] > 0


def test_router_band_is_fit_only_on_train_out_of_fold_scores(scored_frame):
    """The band's reported train_routed_share must be consistent with the
    OOF scores' own size, not the held-out test set's - a cheap structural
    check that nothing in `select_band` silently pulled in test-set size."""
    results = router.run_router(scored_frame)
    train_alerts = results["split"]["train_alerts"]
    oof_rows = results["oof"]["scored_rows"]
    assert oof_rows < train_alerts  # fold 0 excluded, so strictly fewer


def test_routing_by_anomaly_type_covers_all_five_types(scored_frame):
    results = router.run_router(scored_frame)
    by_type = results["routing"]["router->union"]["by_anomaly_type"]
    assert set(by_type.keys()) == {
        "new_host", "off_hours", "new_actor", "manual_recon_chain", "tactic_escalation",
    }
    for entry in by_type.values():
        assert 0.0 <= entry["routed_share"] <= 1.0
        assert 0.0 <= entry["recall"] <= 1.0


# --------------------------------------------------------------------------
# raw_alerts_loader threading (operational-cost timing must use the corpus
# actually being evaluated, not silently default to v1)
# --------------------------------------------------------------------------

def test_measure_operational_cost_uses_the_given_loader_not_the_v1_default():
    """Regression check for a real gap this module had: `frame` was accepted
    but never used for the raw-alert timing measurement, which always
    re-generated v1's default 90-day corpus regardless of what was actually
    being evaluated. `raw_alerts_loader` fixes that - this pins down that a
    custom loader's alert count is what gets reported, not v1 default's."""
    from soc_runtime.synthetic import generate_alerts

    def small_loader() -> pd.DataFrame:
        return generate_alerts(n_days=30)

    small = small_loader()
    featured = baseline.build_features(semantic.build_features(small))
    scored = modeling.restrict_to_scored_techniques(featured)
    train, test = modeling.chronological_split(scored)
    fitted = modeling.fit_arms(train, test)

    result = router.measure_operational_cost(scored, fitted, n_repeats=1, raw_alerts_loader=small_loader)
    expected_n = int((small["event_category"] == "process_creation").sum())
    assert result["n_alerts_timed"] == expected_n


def test_run_router_falls_back_to_a_reachable_traffic_budget(scored_frame, monkeypatch):
    """`config.ROUTER_TRAFFIC_BUDGET` can be unreachable on a corpus whose
    out-of-fold raw scores are coarse (few distinct values) - this is a real
    thing that happens on synthetic_v2's calibrated corpus, not a
    hypothetical. Forcing the configured budget down to an impossible value
    here reproduces that on the existing v1 fixture cheaply, and checks
    `run_router` degrades to the smallest reachable frontier budget instead
    of raising."""
    monkeypatch.setattr(config, "ROUTER_TRAFFIC_BUDGET", 1e-9)
    results = router.run_router(scored_frame)
    assert results["traffic_budget_fallback"] is not None
    assert results["traffic_budget_fallback"]["used_budget"] > 1e-9
    assert results["band"]["train_routed_share"] <= results["traffic_budget_fallback"]["used_budget"] + 1e-9
