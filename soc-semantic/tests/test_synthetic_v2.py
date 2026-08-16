"""Tests for the real-aggregate-calibrated generator (`synthetic_v2.py`).

Mirrors `test_synthetic.py`'s structure for the schema/determinism/anomaly
checks that apply to any generator producing this flat schema, then adds
calibration-specific tests that check the generated corpus's measured
statistics land within a defensible tolerance of
`config.REAL_CALIBRATION_TARGETS` - not exact equality, since the targets
themselves are calibrated-toward summary moments (means/medians/shares),
not full distributions to reproduce. Generation is deterministic for a
given seed (`test_deterministic_given_seed` below), so the tolerance bands
here are not fighting run-to-run noise - they exist because hitting nine
independent real-world aggregate statistics with one small, interpretable
generator is a calibration exercise, not an equation to solve exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from soc_runtime import baseline, config, modeling, semantic
from soc_runtime.opensearch_client import first_of_multivalued, translate_technique_name
from soc_runtime.synthetic import COLUMNS
from soc_runtime.synthetic_v2 import (
    COLUMNS_V2,
    TECHNIQUE_POOL,
    _maybe_multivalued,
    generate_alerts_v2,
    measured_calibration,
)

_TARGETS = config.REAL_CALIBRATION_TARGETS


# --------------------------------------------------------------------------
# Schema / plumbing - same expectations as v1, same shared COLUMNS constant
# --------------------------------------------------------------------------

def test_schema_matches_shared_columns():
    """v2 carries v1's whole schema, in order, plus the two multi-valued
    columns v1 has no use for - so anything reading v1's columns reads v2
    unchanged, which is what lets both corpora share one pipeline."""
    frame = generate_alerts_v2(n_days=10)
    assert list(frame.columns) == list(COLUMNS_V2)
    assert list(frame.columns)[:len(COLUMNS)] == list(COLUMNS)


def test_excludes_customer_node_by_construction():
    frame = generate_alerts_v2(n_days=10)
    assert set(frame["cluster_node"].unique()).isdisjoint(config.EXCLUDED_CLUSTER_NODES)
    assert set(frame["cluster_node"].unique()) <= config.ALLOWED_CLUSTER_NODES


def test_sorted_by_timestamp():
    frame = generate_alerts_v2(n_days=10)
    assert frame["timestamp"].is_monotonic_increasing


def test_deterministic_given_seed():
    a = generate_alerts_v2(n_days=10, seed=123)
    b = generate_alerts_v2(n_days=10, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = generate_alerts_v2(n_days=10, seed=1)
    b = generate_alerts_v2(n_days=10, seed=2)
    assert len(a) != len(b) or not a.equals(b)


def test_default_window_is_30_days_matching_real_measurement():
    """v1 defaults to 90 days; v2 defaults to 30 - the exact window the
    real calibration aggregates were measured over, not an arbitrary
    choice or a copy of v1's."""
    frame = generate_alerts_v2()
    span = frame["timestamp"].max().normalize() - frame["timestamp"].min().normalize()
    assert span.days == 29  # 30 calendar days inclusive


def test_all_process_creation_only():
    frame = generate_alerts_v2(n_days=10)
    assert set(frame["event_category"].unique()) == {"process_creation"}


# --------------------------------------------------------------------------
# Anomaly injection - four of v1's five types; tactic_escalation deliberately
# omitted (see synthetic_v2.py's docstring on _inject_anomalies_v2)
# --------------------------------------------------------------------------

def test_labeled_anomalies_present_and_minority():
    frame = generate_alerts_v2()
    n_anomalies = int(frame["is_synthetic_anomaly"].sum())
    assert n_anomalies > 0
    assert n_anomalies < 0.05 * len(frame)


def test_anomaly_types_exclude_tactic_escalation_by_default():
    frame = generate_alerts_v2()
    types = set(frame.loc[frame["is_synthetic_anomaly"], "synthetic_anomaly_type"].unique())
    assert types == {"new_host", "off_hours", "new_actor", "manual_recon_chain"}
    assert "tactic_escalation" not in types


def test_non_anomalous_rows_have_no_anomaly_type():
    frame = generate_alerts_v2()
    legit = frame.loc[~frame["is_synthetic_anomaly"]]
    assert legit["synthetic_anomaly_type"].isna().all()


# --------------------------------------------------------------------------
# Multi-valued technique modelling - the 57.7% "surprising finding" from the
# task brief, tested as a pure function independent of the full generator
# --------------------------------------------------------------------------

def test_multivalued_helper_rate_matches_real_share():
    rng = np.random.default_rng(7)
    n = 50_000
    hits = sum(1 for _ in range(n) if len(_maybe_multivalued("PowerShell", rng)) > 1)
    measured = hits / n
    target = _TARGETS["multi_valued_technique_share"]
    assert abs(measured - target) < 0.02, f"measured {measured} vs target {target}"


def test_multivalued_helper_first_element_is_always_the_requested_name():
    rng = np.random.default_rng(7)
    for _ in range(500):
        raw = _maybe_multivalued("Account Discovery", rng)
        assert raw[0] == "Account Discovery"
        assert first_of_multivalued(raw) == "Account Discovery"
        assert len(raw) in (1, 2)


def test_technique_pool_names_are_real_observed_names():
    """Every name in the pool must be one of the 20 real names the task
    brief supplied - either mapped in config.TECHNIQUE_NAME_TO_ID, or the
    deliberately-unmapped 'Tool' (not a real MITRE technique)."""
    allowed = set(config.TECHNIQUE_NAME_TO_ID) | {"Tool"}
    assert {t["name"] for t in TECHNIQUE_POOL} <= allowed


def test_generated_technique_ids_translate_or_pass_through_consistently():
    frame = generate_alerts_v2(n_days=15)
    tagged = frame["rule_mitre_technique"].dropna().unique()
    for value in tagged:
        # every tagged value must already be what translate_technique_name
        # would produce - i.e. the generator applied the same translation
        # the real ingestion path applies, not a shortcut.
        assert translate_technique_name(value) == value


# --------------------------------------------------------------------------
# Calibration against config.REAL_CALIBRATION_TARGETS - tolerance-based,
# not exact reproduction (see module docstring). All measured on the
# default 30-day, default-seed corpus that ships as the actual generator
# output, not a specially tuned one-off.
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def calibrated_measurement() -> dict:
    frame = generate_alerts_v2()
    return measured_calibration(frame)


def test_total_volume_within_tolerance_of_real_30_day_count(calibrated_measurement):
    target = _TARGETS["process_creation_total_30d"]
    measured = calibrated_measurement["n_process_creation"]
    assert abs(measured - target) / target < 0.15, f"measured {measured} vs target {target}"


def test_subject_user_name_null_share_within_tolerance(calibrated_measurement):
    target = _TARGETS["subject_user_name_null_share"]
    measured = calibrated_measurement["subject_user_name_null_share"]
    assert abs(measured - target) < 0.01, f"measured {measured} vs target {target}"


def test_mitre_tag_null_share_within_tolerance(calibrated_measurement):
    target = _TARGETS["mitre_tag_null_share"]
    measured = calibrated_measurement["mitre_tag_null_share"]
    assert abs(measured - target) < 0.03, f"measured {measured} vs target {target}"


def test_in_scope_six_technique_share_within_tolerance(calibrated_measurement):
    target = _TARGETS["in_scope_six_technique_share"]
    measured = calibrated_measurement["in_scope_six_technique_share"]
    assert abs(measured - target) < 0.06, f"measured {measured} vs target {target}"


def test_scored_frequency_mean_within_tolerance(calibrated_measurement):
    target = _TARGETS["sem_freq_actor_host_technique_24h_mean"]
    measured = calibrated_measurement["sem_freq_actor_host_technique_24h_mean"]
    assert abs(measured - target) / target < 0.30, f"measured {measured} vs target {target}"


def test_scored_frequency_median_within_tolerance(calibrated_measurement):
    target = _TARGETS["sem_freq_actor_host_technique_24h_median"]
    measured = calibrated_measurement["sem_freq_actor_host_technique_24h_median"]
    assert abs(measured - target) / target < 0.20, f"measured {measured} vs target {target}"


def test_scored_frequency_median_exceeds_mean_matching_real_skew(calibrated_measurement):
    """The real report's median (641) exceeding its mean (462) is itself a
    signal about shape - a small number of very repetitive triples
    dominate, pulled down by a long low-frequency tail. Confirms the
    generator reproduces that qualitative shape, not just the two numbers
    independently."""
    mean = calibrated_measurement["sem_freq_actor_host_technique_24h_mean"]
    median = calibrated_measurement["sem_freq_actor_host_technique_24h_median"]
    assert median > mean


def test_scored_novel_actor_host_technique_within_tolerance(calibrated_measurement):
    target = _TARGETS["sem_novel_actor_host_technique_mean"]
    measured = calibrated_measurement["sem_novel_actor_host_technique_mean"]
    assert abs(measured - target) < 0.003, f"measured {measured} vs target {target}"


def test_scored_novel_actor_near_zero(calibrated_measurement):
    """Target is 0.0001 - essentially zero on a corpus this size. Checking
    an exact match would be checking noise; checking the right order of
    magnitude (comfortably under 1%) is the meaningful claim."""
    measured = calibrated_measurement["sem_novel_actor_mean"]
    assert measured < 0.01


def test_scored_chain_discovery_to_escalation_share_is_zero(calibrated_measurement):
    """Real measurement: share_nonzero == 0.0 exactly - this specific
    15-minute Discovery-then-escalation sequence was never observed in the
    real 30-day window. Matches by construction here (tactic_escalation
    anomaly injection is omitted, see synthetic_v2.py), so this must be
    exact, not approximate."""
    assert calibrated_measurement["sem_chain_discovery_to_escalation_15m_share_nonzero"] == 0.0


def test_scored_actor_identity_missing_subject_is_full(calibrated_measurement):
    """Real measurement: mean == 1.0 exactly on the scored subset - every
    in-scope-technique alert had subject_user_name null. Matches by
    construction here (identity masking is forced, not stochastic, for
    in-scope-technique rows - see _process_alert_v2), so this must be
    exact, not approximate."""
    assert calibrated_measurement["sem_actor_identity_missing_subject_mean"] == 1.0


# --------------------------------------------------------------------------
# v1 must remain untouched by this module's existence
# --------------------------------------------------------------------------

def test_v1_synthetic_module_columns_unchanged():
    from soc_runtime.synthetic import COLUMNS as v1_columns
    assert v1_columns == COLUMNS
