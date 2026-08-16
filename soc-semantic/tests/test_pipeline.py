"""End-to-end smoke tests: load -> filter -> feature -> validate, on the
synthetic source only. No network access is exercised."""

from __future__ import annotations

import pytest

from soc_runtime import baseline, config, modeling, pipeline, semantic


def test_load_alerts_synthetic_default():
    frame = pipeline.load_alerts()
    assert len(frame) > 0
    assert set(frame["cluster_node"].unique()).isdisjoint(config.EXCLUDED_CLUSTER_NODES)


def test_load_alerts_rejects_unknown_source(monkeypatch):
    monkeypatch.setattr(config, "DATA_SOURCE", "not-a-real-source")
    with pytest.raises(ValueError, match="unknown config.DATA_SOURCE"):
        pipeline.load_alerts()


def test_load_features_has_both_feature_families():
    frame = pipeline.load_features()
    assert set(semantic.FEATURE_NAMES) <= set(frame.columns)
    assert set(baseline.FEATURE_NAMES) <= set(frame.columns)


def test_load_features_v2_has_both_feature_families():
    """`load_features_v2` is a separate, explicitly opt-in loader for the
    real-aggregate-calibrated corpus - not wired into config.DATA_SOURCE -
    so it needs its own coverage rather than assuming `load_features`'s
    test above exercises the same code path."""
    frame = pipeline.load_features_v2()
    assert set(semantic.FEATURE_NAMES) <= set(frame.columns)
    assert set(baseline.FEATURE_NAMES) <= set(frame.columns)
    assert set(frame["cluster_node"].unique()).isdisjoint(config.EXCLUDED_CLUSTER_NODES)


def test_validation_report_has_expected_shape():
    frame = pipeline.load_features()
    scored = modeling.restrict_to_scored_techniques(frame)
    report = modeling.run_validation(scored)

    assert "disclaimer" in report and "SYNTHETIC" in report["disclaimer"]
    assert set(report["arms"].keys()) == {"baseline", "semantic", "raw_plus_semantic"}
    for arm in report["arms"].values():
        m = arm["metrics"]
        assert 0.0 <= m["precision_at_threshold"] <= 1.0
        assert 0.0 <= m["recall_at_threshold"] <= 1.0


def test_semantic_arm_does_not_regress_recall_below_baseline():
    """Not a hard architectural guarantee - a different synthetic seed could
    in principle flip this - but it should hold with the default generator,
    and a silent regression here is worth failing loudly on."""
    frame = pipeline.load_features()
    scored = modeling.restrict_to_scored_techniques(frame)
    report = modeling.run_validation(scored)
    baseline_recall = report["arms"]["baseline"]["metrics"]["recall_at_threshold"]
    semantic_recall = report["arms"]["semantic"]["metrics"]["recall_at_threshold"]
    assert semantic_recall >= baseline_recall
