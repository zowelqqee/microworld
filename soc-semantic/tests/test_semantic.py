"""Properties the semantic layer has to hold, checked on hand-built and
synthetic alerts. None of this needs real data."""

from __future__ import annotations

import pandas as pd
import pytest

from soc_runtime.semantic import (
    FEATURE_NAMES, _epoch_seconds, _parse_domain_user, _resolve_actor, build_features,
)
from soc_runtime.synthetic import generate_alerts

_DEFAULTS = {
    "alert_uid": "u0", "cluster_node": "office-collector", "agent_name": "HOST-A",
    "agent_ip": "10.0.0.1", "rule_id": 1, "rule_level": 3, "rule_description": "test",
    "rule_mitre_tactic": "Discovery", "rule_mitre_technique": "T1033",
    "event_category": "process_creation", "subject_user_name": "user1",
    "subject_domain_name": "CORP", "eventdata_user": None,
    "target_user_name": "user1", "target_domain_name": "CORP",
    "command_line": "whoami.exe /all", "image": "whoami.exe", "parent_image": "cmd.exe",
    "parent_command_line": "cmd.exe /k", "parent_process_guid": "{p}", "process_guid": "{c}",
    "logon_type": None, "authentication_package_name": None,
    "is_synthetic_anomaly": False, "synthetic_anomaly_type": None,
}


def make_rows(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame([{**_DEFAULTS, **row} for row in rows])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    return frame.sort_values("timestamp", kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------
# Causality
# --------------------------------------------------------------------------

def test_features_never_read_the_future():
    """Recomputing on a prefix must reproduce that prefix's rows exactly.
    If any feature looked ahead, truncating the input would change earlier
    rows - the same guard used in the network-intrusion-detection port."""
    base = pd.Timestamp("2026-06-01 08:00:00")
    rows = [
        {"timestamp": base + pd.Timedelta(minutes=7 * i), "agent_name": f"HOST-{i % 4}",
         "subject_user_name": f"user{i % 5}", "rule_mitre_technique": ["T1033", "T1082", "T1087"][i % 3]}
        for i in range(120)
    ]
    frame = make_rows(rows)

    full = build_features(frame)
    prefix = build_features(frame.iloc[:40].reset_index(drop=True))

    common_idx = prefix.index
    pd.testing.assert_frame_equal(
        full.loc[common_idx, list(FEATURE_NAMES)].reset_index(drop=True),
        prefix.loc[common_idx, list(FEATURE_NAMES)].reset_index(drop=True),
    )


def test_requires_sorted_input():
    base = pd.Timestamp("2026-06-01 08:00:00")
    frame = make_rows([
        {"timestamp": base + pd.Timedelta(minutes=10)},
        {"timestamp": base},
    ])
    frame = frame.iloc[::-1].reset_index(drop=True)  # deliberately unsorted
    with pytest.raises(ValueError):
        build_features(frame)


# --------------------------------------------------------------------------
# Real-data compatibility
# --------------------------------------------------------------------------

def test_handles_mitre_technique_ids_outside_the_synthetic_catalogue():
    """Regression test for a real bug: `advance()` used to derive the
    process image via `config.TECHNIQUE_CATALOG[technique]["image"]`, which
    KeyErrors on any MITRE technique ID outside the eight this prototype's
    synthetic generator invented - i.e. on essentially any real alert, since
    a real environment tags dozens of different techniques. The alert's own
    `image` field is used directly now; the technique ID no longer needs a
    catalogue entry at all."""
    base = pd.Timestamp("2026-06-01 08:00:00")
    frame = make_rows([
        {"timestamp": base, "rule_mitre_technique": "T1218.011", "rule_mitre_tactic": "Defense Evasion",
         "image": "C:\\Windows\\System32\\rundll32.exe"},
        {"timestamp": base + pd.Timedelta(minutes=5), "rule_mitre_technique": "T1547.001",
         "rule_mitre_tactic": "Persistence", "image": "C:\\Windows\\System32\\reg.exe"},
    ])
    featured = build_features(frame)  # must not raise KeyError
    assert len(featured) == 2


def test_handles_missing_image_field_without_crashing():
    """Some real rule configurations may not log `data.win.eventdata.image`.
    A null `image` must degrade to a placeholder, not raise or silently
    produce a `None`-keyed process-chain entry that behaves unpredictably."""
    base = pd.Timestamp("2026-06-01 08:00:00")
    frame = make_rows([
        {"timestamp": base, "image": None},
        {"timestamp": base + pd.Timedelta(minutes=5), "image": None},
    ])
    featured = build_features(frame)
    assert featured["sem_process_chain_novel_for_host"].iloc[0] == 1.0
    assert featured["sem_process_chain_novel_for_host"].iloc[1] == 0.0  # same (host, parent, unknown-image) seen before


def test_epoch_seconds_agrees_across_representations_of_the_same_moment():
    """Regression test for the exact TypeError the first real run raised:
    `.astype("int64")` on a `timestamp` column crashes once that column is
    `object` dtype rather than a uniform `datetime64`. `_epoch_seconds` must
    give the identical answer for the same wall-clock moment however it is
    represented: a plain numeric Unix-epoch value (the ticket's original
    hypothesis for what synthetic data looked like), a uniform naive
    `datetime64[ns]` column (what synthetic data actually is), a uniform
    tz-aware column, and - the shape that actually broke on real data - an
    `object`-dtype column mixing tz-aware and tz-naive `Timestamp` instances."""
    moment_naive = pd.Timestamp("2026-08-05 10:00:00")
    moment_utc = pd.Timestamp("2026-08-05 10:00:00", tz="UTC")
    epoch = moment_utc.timestamp()  # the reference value every representation must agree with

    numeric = pd.Series([epoch])
    uniform_naive = pd.Series([moment_naive])
    uniform_aware = pd.Series([moment_utc])
    mixed_object = pd.Series([moment_utc, moment_naive], dtype=object)

    assert uniform_naive.dtype.kind == "M"  # sanity: a real datetime64 column
    assert mixed_object.dtype == object  # sanity: the shape that actually crashed

    assert _epoch_seconds(numeric)[0] == epoch
    assert _epoch_seconds(uniform_naive)[0] == epoch
    assert _epoch_seconds(uniform_aware)[0] == epoch
    assert list(_epoch_seconds(mixed_object)) == [epoch, epoch]


def test_epoch_seconds_handles_the_object_dtype_column_without_raising():
    """The literal crash scenario, reproduced directly: an `object`-dtype
    `timestamp` Series built the way `hits_to_frame` used to produce it
    before its own fix, from real documents with inconsistent UTC-offset
    formatting across sources."""
    mixed = pd.Series(
        [pd.Timestamp("2026-08-05T10:00:00.000Z"), pd.Timestamp("2026-08-05 10:05:00")],
        dtype=object,
    )
    result = _epoch_seconds(mixed)  # must not raise TypeError
    assert result[1] - result[0] == 300.0  # 5 minutes apart


# --------------------------------------------------------------------------
# Actor identity fallback
#
# subject_user_name/subject_domain_name are null on 99.48% of real
# process-creation alerts (10,426 of 10,480 in the first 5-day real sample)
# - not an edge case for an architecture keyed on "who did this". These
# tests check the three-tier fallback (_resolve_actor) and the domain\\user
# parser (_parse_domain_user) in isolation, on fabricated field values -
# there is no live access to confirm the real fill rate of the fallback
# field itself; see the README's "Real data" section.
# --------------------------------------------------------------------------

def test_parse_domain_user_splits_backslash_form():
    assert _parse_domain_user("CORP\\jdoe") == ("jdoe", "CORP")


def test_parse_domain_user_handles_bare_username():
    assert _parse_domain_user("SYSTEM") == ("SYSTEM", None)


def test_resolve_actor_prefers_subject_fields_when_present():
    actor, missing_subject, host_fallback = _resolve_actor("jdoe", "CORP", "OTHERDOMAIN\\other", "HOST-1")
    assert actor == ("jdoe", "CORP")
    assert missing_subject is False
    assert host_fallback is False


def test_resolve_actor_falls_back_to_eventdata_user_when_subject_missing():
    actor, missing_subject, host_fallback = _resolve_actor(None, None, "CORP\\jdoe", "HOST-1")
    assert actor == ("jdoe", "CORP")
    assert missing_subject is True
    assert host_fallback is False


def test_resolve_actor_treats_empty_string_subject_as_missing():
    """Real data can carry an empty string, not just null - both must be
    treated as "not usable", not as a literal empty-string identity."""
    actor, missing_subject, host_fallback = _resolve_actor("", "", "CORP\\jdoe", "HOST-1")
    assert actor == ("jdoe", "CORP")
    assert missing_subject is True


def test_resolve_actor_falls_back_to_host_when_nothing_usable():
    actor, missing_subject, host_fallback = _resolve_actor(None, None, None, "HOST-1")
    assert actor == ("(host:HOST-1)", None)
    assert missing_subject is True
    assert host_fallback is True


def test_resolve_actor_falls_back_to_host_when_eventdata_user_unparseable():
    """An eventdata_user value that parses to an empty username (e.g. a bare
    backslash) must not silently become a usable identity - it degrades all
    the way to the host fallback, same as if the field were absent."""
    actor, missing_subject, host_fallback = _resolve_actor(None, None, "CORP\\", "HOST-1")
    assert actor == ("(host:HOST-1)", None)
    assert host_fallback is True


def test_actor_identity_flags_present_and_correct_through_build_features():
    """End to end: the two sem_* flags on the output must match which tier
    resolved each row's actor."""
    base = pd.Timestamp("2026-06-01 08:00:00")
    frame = make_rows([
        {"timestamp": base, "subject_user_name": "jdoe", "subject_domain_name": "CORP"},
        {"timestamp": base + pd.Timedelta(minutes=5), "subject_user_name": None,
         "subject_domain_name": None, "eventdata_user": "CORP\\jdoe"},
        {"timestamp": base + pd.Timedelta(minutes=10), "subject_user_name": None,
         "subject_domain_name": None, "eventdata_user": None},
    ])
    featured = build_features(frame)
    assert list(featured["sem_actor_identity_missing_subject"]) == [0.0, 1.0, 1.0]
    assert list(featured["sem_actor_identity_host_fallback"]) == [0.0, 0.0, 1.0]


def test_host_fallback_actors_on_the_same_host_share_identity():
    """The documented degradation, checked directly: two alerts from the
    same host with no usable identity collapse onto one pseudo-actor, so
    the second one is not "novel" even though no real identity was ever
    confirmed to repeat."""
    base = pd.Timestamp("2026-06-01 08:00:00")
    frame = make_rows([
        {"timestamp": base, "subject_user_name": None, "subject_domain_name": None,
         "agent_name": "HOST-1"},
        {"timestamp": base + pd.Timedelta(minutes=5), "subject_user_name": None,
         "subject_domain_name": None, "agent_name": "HOST-1"},
    ])
    featured = build_features(frame)
    assert featured["sem_novel_actor"].iloc[0] == 1.0
    assert featured["sem_novel_actor"].iloc[1] == 0.0  # same pseudo-actor "(host:HOST-1)"


# --------------------------------------------------------------------------
# No raw identity leakage
# --------------------------------------------------------------------------

def test_no_feature_carries_raw_identity():
    """Every sem_* column must be numeric - actor names, host names and
    technique IDs are used to key the entity graph but must never themselves
    become a feature value, or a model could just memorise the fixed
    service-account name instead of learning the behavioural pattern."""
    frame = generate_alerts(n_days=20)
    featured = build_features(frame)
    dtypes = featured[list(FEATURE_NAMES)].dtypes
    assert all(dtype.kind in "fiub" for dtype in dtypes), dtypes


# --------------------------------------------------------------------------
# Frequency and novelty
# --------------------------------------------------------------------------

def test_frequency_and_novelty_increment_causally():
    base = pd.Timestamp("2026-06-01 08:00:00")
    rows = [
        {"timestamp": base + pd.Timedelta(hours=i), "agent_name": "HOST-A",
         "subject_user_name": "user1", "rule_mitre_technique": "T1033"}
        for i in range(3)
    ]
    featured = build_features(make_rows(rows))
    assert list(featured["sem_freq_actor_host_technique_24h"]) == [0.0, 1.0, 2.0]
    assert list(featured["sem_novel_actor_host_technique"]) == [1.0, 0.0, 0.0]
    assert list(featured["sem_novel_actor"]) == [1.0, 0.0, 0.0]


def test_frequency_window_expires_after_24h():
    base = pd.Timestamp("2026-06-01 08:00:00")
    rows = [
        {"timestamp": base, "agent_name": "HOST-A", "subject_user_name": "user1", "rule_mitre_technique": "T1033"},
        {"timestamp": base + pd.Timedelta(hours=25), "agent_name": "HOST-A",
         "subject_user_name": "user1", "rule_mitre_technique": "T1033"},
    ]
    featured = build_features(make_rows(rows))
    assert featured["sem_freq_actor_host_technique_24h"].iloc[1] == 0.0
    # but the lifetime "ever seen" novelty flag still remembers it
    assert featured["sem_novel_actor_host_technique"].iloc[1] == 0.0


def test_new_host_for_actor_flagged_even_with_technique_history_elsewhere():
    base = pd.Timestamp("2026-06-01 08:00:00")
    rows = [
        {"timestamp": base, "agent_name": "HOST-A", "subject_user_name": "user1", "rule_mitre_technique": "T1033"},
        {"timestamp": base + pd.Timedelta(hours=1), "agent_name": "HOST-B",
         "subject_user_name": "user1", "rule_mitre_technique": "T1033"},
    ]
    featured = build_features(make_rows(rows))
    assert featured["sem_novel_host_for_actor"].iloc[1] == 1.0
    assert featured["sem_novel_actor"].iloc[1] == 0.0  # same actor, not a new identity


# --------------------------------------------------------------------------
# Escalation chain
# --------------------------------------------------------------------------

def test_escalation_flag_fires_for_discovery_then_credential_access_in_window():
    base = pd.Timestamp("2026-06-01 08:00:00")
    rows = [
        {"timestamp": base, "agent_name": "HOST-A", "subject_user_name": "attacker",
         "rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"},
        {"timestamp": base + pd.Timedelta(minutes=5), "agent_name": "HOST-A", "subject_user_name": "attacker",
         "rule_mitre_technique": "T1003", "rule_mitre_tactic": "Credential Access"},
    ]
    featured = build_features(make_rows(rows))
    assert featured["sem_chain_discovery_to_escalation_15m"].iloc[1] == 1.0
    assert featured["sem_chain_tactic_diversity_15m"].iloc[1] == 2.0


def test_escalation_flag_does_not_fire_outside_window():
    base = pd.Timestamp("2026-06-01 08:00:00")
    rows = [
        {"timestamp": base, "agent_name": "HOST-A", "subject_user_name": "attacker",
         "rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"},
        {"timestamp": base + pd.Timedelta(minutes=20), "agent_name": "HOST-A", "subject_user_name": "attacker",
         "rule_mitre_technique": "T1003", "rule_mitre_tactic": "Credential Access"},
    ]
    featured = build_features(make_rows(rows))
    assert featured["sem_chain_discovery_to_escalation_15m"].iloc[1] == 0.0


def test_escalation_flag_is_directional():
    """Credential Access before Discovery is not the sequence a SOC analyst
    would call an escalation - only Discovery-then-escalation should fire."""
    base = pd.Timestamp("2026-06-01 08:00:00")
    rows = [
        {"timestamp": base, "agent_name": "HOST-A", "subject_user_name": "attacker",
         "rule_mitre_technique": "T1003", "rule_mitre_tactic": "Credential Access"},
        {"timestamp": base + pd.Timedelta(minutes=5), "agent_name": "HOST-A", "subject_user_name": "attacker",
         "rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"},
    ]
    featured = build_features(make_rows(rows))
    assert featured["sem_chain_discovery_to_escalation_15m"].iloc[1] == 0.0


# --------------------------------------------------------------------------
# Aggregate sanity: synthetic anomalies should look more "novel" on average
# than the recurring legitimate pattern. Group means, not exact numbers, so
# this does not pin itself to incidental RNG behaviour.
# --------------------------------------------------------------------------

def test_synthetic_anomalies_score_higher_novelty_on_average_than_recurring_legit():
    from soc_runtime.synthetic import SERVICE_ACCOUNT

    frame = generate_alerts(n_days=90)
    featured = build_features(frame)
    recurring = featured[featured["subject_user_name"] == SERVICE_ACCOUNT[0]]
    anomalies = featured[featured["is_synthetic_anomaly"]]

    assert anomalies["sem_novel_actor_host_technique"].mean() > recurring["sem_novel_actor_host_technique"].mean()
    assert anomalies["sem_actor_time_typicality"].mean() < recurring["sem_actor_time_typicality"].mean()
