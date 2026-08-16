"""The baseline layer is stateless, so its tests only need to check shape
and that the raw fields were read correctly - no causality to verify."""

from __future__ import annotations

from soc_runtime import config
from soc_runtime.baseline import FEATURE_NAMES, build_features
from soc_runtime.semantic import build_features as semantic_build_features
from soc_runtime.synthetic import generate_alerts


def test_adds_declared_columns():
    frame = generate_alerts(n_days=20)
    featured = semantic_build_features(frame)
    featured = build_features(featured)
    assert set(FEATURE_NAMES) <= set(featured.columns)


def test_technique_one_hot_sums_to_one_for_in_scope_rows():
    frame = generate_alerts(n_days=30)
    featured = semantic_build_features(frame)
    featured = build_features(featured)
    in_scope = featured[featured["rule_mitre_technique"].isin(config.IN_SCOPE_TECHNIQUES)]
    technique_cols = [f"base_technique_{t}" for t in config.IN_SCOPE_TECHNIQUES]
    assert (in_scope[technique_cols].sum(axis=1) == 1.0).all()


def test_hour_band_one_hot_sums_to_one():
    frame = generate_alerts(n_days=20)
    featured = semantic_build_features(frame)
    featured = build_features(featured)
    band_cols = [f"base_hour_band_{label}" for _, _, label in config.HOUR_BANDS]
    assert (featured[band_cols].sum(axis=1) == 1.0).all()


def test_no_identity_column_among_base_features():
    assert not any("user" in name.lower() or "host" in name.lower() or "agent" in name.lower()
                   for name in FEATURE_NAMES)
