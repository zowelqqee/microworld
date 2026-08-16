"""Sanity checks on the synthetic generator. Nothing here needs real data."""

from __future__ import annotations

import subprocess
import sys

import pandas as pd

from soc_runtime import config
from soc_runtime.synthetic import COLUMNS, SERVICE_ACCOUNT, generate_alerts


def test_schema_matches_declared_columns():
    frame = generate_alerts(n_days=20)
    assert list(frame.columns) == list(COLUMNS)


def test_excludes_customer_node_by_construction():
    frame = generate_alerts(n_days=20)
    assert set(frame["cluster_node"].unique()).isdisjoint(config.EXCLUDED_CLUSTER_NODES)
    assert set(frame["cluster_node"].unique()) <= config.ALLOWED_CLUSTER_NODES


def test_sorted_by_timestamp():
    frame = generate_alerts(n_days=20)
    assert frame["timestamp"].is_monotonic_increasing


def test_deterministic_given_seed():
    a = generate_alerts(n_days=20, seed=123)
    b = generate_alerts(n_days=20, seed=123)
    pd.testing.assert_frame_equal(a, b)


def test_default_end_date_is_pinned_not_today():
    """Regression test for a real bug: `generate_alerts()` used to default
    `end` to `pd.Timestamp.now()`, so the 90-day window - and every
    downstream README number - silently shifted by a day each day the
    pipeline ran, because different calendar days land in different
    weekday/weekend positions relative to the fixed seed. The default must
    match an explicit call with `config.SYNTHETIC_END_DATE`, regardless of
    what day this test happens to run on."""
    default_run = generate_alerts(n_days=20)
    pinned_run = generate_alerts(n_days=20, end=config.SYNTHETIC_END_DATE)
    pd.testing.assert_frame_equal(default_run, pinned_run)
    assert str(default_run["timestamp"].max().date()) <= config.SYNTHETIC_END_DATE


def test_deterministic_across_process_boundaries():
    """Regression test for a real bug: `_host_ip`/`_host_node`/the sweep-time
    offset used Python's built-in `hash()`, which is salted per-process for
    strings by default. Two separate `python run_pipeline.py` runs with the
    identical seed produced slightly different data, and the in-process
    `test_deterministic_given_seed` above could not catch it - a process's
    hash salt is fixed for its own lifetime, so two calls in the same test
    always agreed with each other regardless of the bug. This test runs the
    generator in two subprocesses with deliberately different
    `PYTHONHASHSEED` values and checks the *first row* is identical -
    cheap enough to run every time without spawning a subprocess per field.
    """
    script = (
        "from soc_runtime.synthetic import generate_alerts; "
        "df = generate_alerts(n_days=15); "
        "row = df.iloc[0]; "
        "print(row['timestamp'], row['agent_ip'], row['cluster_node'])"
    )
    import os

    def run_with_hashseed(seed: str) -> str:
        env = {**os.environ, "PYTHONHASHSEED": seed}
        result = subprocess.run(
            [sys.executable, "-c", script], cwd=str(config.REPO_ROOT),
            capture_output=True, text=True, env=env, timeout=60,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    assert run_with_hashseed("1") == run_with_hashseed("99999")


def test_different_seed_gives_different_data():
    a = generate_alerts(n_days=20, seed=1)
    b = generate_alerts(n_days=20, seed=2)
    assert len(a) != len(b) or not a.equals(b)


def test_labeled_anomalies_present_and_minority():
    frame = generate_alerts(n_days=90)
    n_anomalies = int(frame["is_synthetic_anomaly"].sum())
    assert n_anomalies > 0
    assert n_anomalies < 0.05 * len(frame), "anomalies should be a small minority, not the norm"


def test_all_declared_anomaly_types_appear():
    frame = generate_alerts(n_days=90)
    types = set(frame.loc[frame["is_synthetic_anomaly"], "synthetic_anomaly_type"].unique())
    assert types == {"new_host", "off_hours", "new_actor", "manual_recon_chain", "tactic_escalation"}


def test_recurring_service_account_is_the_dominant_process_creation_pattern():
    frame = generate_alerts(n_days=90)
    process = frame[frame["event_category"] == "process_creation"]
    share = (process["subject_user_name"] == SERVICE_ACCOUNT[0]).mean()
    assert share > 0.5, "the fixed inventory-agent pattern should dominate alert volume, per the brief"


def test_non_anomalous_rows_have_no_anomaly_type():
    frame = generate_alerts(n_days=30)
    legit = frame.loc[~frame["is_synthetic_anomaly"]]
    assert legit["synthetic_anomaly_type"].isna().all()


def test_volume_in_realistic_range_at_90_days():
    frame = generate_alerts(n_days=90)
    assert 8_000 <= len(frame) <= 60_000
