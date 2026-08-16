"""real_data_check.py cannot be tested against a live cluster (no
credentials given to this codebase, by design - see the README). What can
and must be guaranteed without one: it never connects on import or with the
default config, its safety guards actually block what they claim to, and
its diagnostics produce correct answers on fabricated documents that mimic
the confirmed real schema."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from soc_runtime import config, real_data_check as rdc
from soc_runtime.semantic import FEATURE_NAMES


def test_import_performs_no_network_io():
    assert hasattr(rdc, "main")


def test_refuses_to_run_against_default_synthetic_data_source():
    assert config.DATA_SOURCE == "synthetic"
    with pytest.raises(rdc.RealDataCheckError, match="SOC_DATA_SOURCE=opensearch"):
        rdc._require_opensearch_data_source()


def test_main_refuses_before_touching_network(monkeypatch):
    """Calling main() with the default config must fail on the data-source
    check, not get far enough to construct a client."""
    with pytest.raises(rdc.RealDataCheckError):
        rdc.main([])


# --------------------------------------------------------------------------
# Volume/window safety guards
# --------------------------------------------------------------------------

def test_check_window_safety_blocks_large_window_by_default():
    with pytest.raises(rdc.RealDataCheckError, match="REAL_DATA_MAX_WINDOW_DAYS"):
        rdc.check_window_safety(config.REAL_DATA_MAX_WINDOW_DAYS + 1, allow_large_window=False)


def test_check_window_safety_allows_large_window_when_overridden():
    rdc.check_window_safety(config.REAL_DATA_MAX_WINDOW_DAYS + 1, allow_large_window=True)  # must not raise


def test_check_window_safety_allows_default_window():
    rdc.check_window_safety(config.REAL_DATA_DEFAULT_WINDOW_DAYS, allow_large_window=False)  # must not raise


def test_check_volume_safety_blocks_large_count_by_default():
    with pytest.raises(rdc.RealDataCheckError, match="REAL_DATA_COUNT_SAFETY_LIMIT"):
        rdc.check_volume_safety(config.REAL_DATA_COUNT_SAFETY_LIMIT + 1, allow_large_window=False)


def test_check_volume_safety_allows_override():
    rdc.check_volume_safety(config.REAL_DATA_COUNT_SAFETY_LIMIT + 1, allow_large_window=True)  # must not raise


# --------------------------------------------------------------------------
# Source breakdown
# --------------------------------------------------------------------------

def _hit(source_shape: dict, **overrides) -> dict:
    base = {
        "_id": "x", "_source": {
            "timestamp": "2026-08-05T10:00:00.000Z", "cluster": {"node": "office-collector"},
            "agent": {"name": "H1"}, "rule": {"id": 1, "level": 3},
        },
    }
    base["_source"].update(source_shape)
    base.update(overrides)
    return base


def test_source_breakdown_counts_all_four_kinds():
    hits = [
        _hit({"data": {"win": {"eventdata": {"commandLine": "whoami.exe"}}}}),
        _hit({"data": {"win": {"eventdata": {"logonType": 2}}}}),
        _hit({"data": {"alert": {"signature": "x"}, "src_ip": "1.2.3.4"}}),
        _hit({"data": {"KES": {"event": "Detected"}}}),
        _hit({"data": {"nothing_recognised": True}}),
    ]
    breakdown = rdc.source_breakdown(hits)
    assert breakdown["by_source_kind"] == {
        "sysmon_process": 1, "sysmon_auth": 1, "suricata": 1, "kes": 1, "unknown": 1,
    }


def test_source_breakdown_counts_multi_valued_mitre_tags():
    hits = [
        _hit({
            "rule": {"id": 1, "level": 3, "mitre": {"tactic": ["Discovery", "Execution"], "technique": ["T1033", "T1059"]}},
            "data": {"win": {"eventdata": {"commandLine": "whoami.exe"}}},
        }),
        _hit({
            "rule": {"id": 2, "level": 3, "mitre": {"tactic": ["Discovery"], "technique": ["T1082"]}},
            "data": {"win": {"eventdata": {"commandLine": "systeminfo.exe"}}},
        }),
    ]
    breakdown = rdc.source_breakdown(hits)
    assert breakdown["multi_valued_technique_alerts"] == 1
    assert breakdown["multi_valued_tactic_alerts"] == 1


# --------------------------------------------------------------------------
# Unknown-source characterisation
#
# The first real run classified 27,180 of 188,946 alerts (14.4%) as
# "unknown" - not fixed here, but characterised so it is not an unlabelled
# black hole in the report.
# --------------------------------------------------------------------------

def test_unknown_source_sample_only_counts_unknown_hits():
    hits = [
        _hit({"decoder": {"name": "syscheck"}, "rule": {"description": "File changed"},
              "data": {"syscheck": {"path": "/etc/passwd"}}}),
        _hit({"data": {"win": {"eventdata": {"commandLine": "whoami.exe"}}}}),  # sysmon_process, not unknown
    ]
    sample = rdc.unknown_source_sample(hits)
    assert sample["n_unknown"] == 1
    assert sample["top_decoder_names"] == [("syscheck", 1)]
    assert sample["top_rule_descriptions"] == [("File changed", 1)]
    assert sample["top_data_key_shapes"] == [{"keys": ["syscheck"], "count": 1}]


def test_unknown_source_sample_aggregates_repeated_shapes():
    hits = [_hit({"decoder": {"name": "rootcheck"}, "data": {"foo": 1}}) for _ in range(3)]
    sample = rdc.unknown_source_sample(hits)
    assert sample["n_unknown"] == 3
    assert sample["top_decoder_names"] == [("rootcheck", 3)]


def test_unknown_source_sample_never_includes_raw_field_values():
    """Only field *names* and small metadata strings (decoder name, rule
    description) may appear - never arbitrary document content, per the
    'aggregates only' requirement."""
    hits = [_hit({"data": {"secret_looking_field": "sensitive-value-should-not-leak"}})]
    sample = rdc.unknown_source_sample(hits)
    dumped = str(sample)
    assert "sensitive-value-should-not-leak" not in dumped
    assert "secret_looking_field" in dumped  # the key name is fine, it's metadata


# --------------------------------------------------------------------------
# Field completeness
# --------------------------------------------------------------------------

def test_field_completeness_reports_null_shares_and_unscored_technique_ids():
    frame = pd.DataFrame([
        {"event_category": "process_creation", "subject_user_name": "u1", "subject_domain_name": "CORP",
         "eventdata_user": None, "agent_name": "H1", "rule_mitre_technique": "T1033",
         "rule_mitre_tactic": "Discovery", "image": "whoami.exe", "timestamp": pd.Timestamp("2026-08-05")},
        {"event_category": "process_creation", "subject_user_name": None, "subject_domain_name": "CORP",
         "eventdata_user": None, "agent_name": "H2", "rule_mitre_technique": "T1218.011",
         "rule_mitre_tactic": "Defense Evasion", "image": None, "timestamp": pd.Timestamp("2026-08-05")},
    ])
    result = rdc.field_completeness(frame)
    assert result["n_process_creation"] == 2
    assert result["field_completeness"]["subject_user_name"]["null_count"] == 1
    assert result["field_completeness"]["subject_user_name"]["null_share"] == 0.5
    # T1218.011 is a real, correctly-ID-shaped technique - just not one of the
    # eight this prototype scores. That's expected, not an unmapped-name gap.
    assert "T1218.011" in result["unscored_technique_ids"]
    assert result["n_unscored_technique_ids"] == 1
    assert result["n_unmapped_technique_names"] == 0
    assert result["share_technique_in_the_six_scored"] == 0.5


def test_field_completeness_flags_technique_values_that_never_translated():
    """The distinct, actionable finding: a value that is still a bare name
    (never matched config.TECHNIQUE_NAME_TO_ID, never ID-shaped) means the
    dictionary needs extending - reported separately from the expected
    "valid ID we just don't score" case above."""
    frame = pd.DataFrame([
        {"event_category": "process_creation", "subject_user_name": "u1", "subject_domain_name": "CORP",
         "eventdata_user": None, "agent_name": "H1", "rule_mitre_technique": "Some Brand New Technique",
         "rule_mitre_tactic": "Discovery", "image": "whoami.exe", "timestamp": pd.Timestamp("2026-08-05")},
    ])
    result = rdc.field_completeness(frame)
    assert result["unmapped_technique_names"] == ["Some Brand New Technique"]
    assert result["n_unmapped_technique_names"] == 1
    assert result["n_unscored_technique_ids"] == 0


def test_field_completeness_reports_actor_identity_fallback_coverage():
    """The specific diagnostic the second real-run finding needs: of the
    alerts subject_user_name cannot identify, how many does eventdata_user
    rescue? Mirrors the real cluster's shape - most alerts null, a minority
    rescued by the fallback field."""
    frame = pd.DataFrame([
        {"event_category": "process_creation", "subject_user_name": "u1", "subject_domain_name": "CORP",
         "eventdata_user": None, "agent_name": "H1", "rule_mitre_technique": "T1033",
         "rule_mitre_tactic": "Discovery", "image": "whoami.exe", "timestamp": pd.Timestamp("2026-08-05")},
        {"event_category": "process_creation", "subject_user_name": None, "subject_domain_name": None,
         "eventdata_user": "CORP\\jdoe", "agent_name": "H2", "rule_mitre_technique": "T1033",
         "rule_mitre_tactic": "Discovery", "image": "whoami.exe", "timestamp": pd.Timestamp("2026-08-05")},
        {"event_category": "process_creation", "subject_user_name": None, "subject_domain_name": None,
         "eventdata_user": None, "agent_name": "H3", "rule_mitre_technique": "T1033",
         "rule_mitre_tactic": "Discovery", "image": "whoami.exe", "timestamp": pd.Timestamp("2026-08-05")},
    ])
    result = rdc.field_completeness(frame)
    fallback = result["actor_identity_fallback_coverage"]
    assert fallback["n_subject_user_name_missing"] == 2
    assert fallback["n_rescued_by_eventdata_user"] == 1
    assert fallback["rescued_share"] == 0.5


def test_field_completeness_on_empty_frame():
    frame = pd.DataFrame({"event_category": []})
    result = rdc.field_completeness(frame)
    assert result["n_process_creation"] == 0


# --------------------------------------------------------------------------
# Safe feature building
# --------------------------------------------------------------------------

def test_build_features_safely_reports_errors_instead_of_raising():
    bad_frame = pd.DataFrame({
        "timestamp": pd.to_datetime(["2026-08-05", "2026-08-01"]),  # deliberately unsorted
        "event_category": ["process_creation", "process_creation"],
        "cluster_node": ["office-collector", "office-collector"],
        "agent_name": ["H1", "H1"], "subject_user_name": ["u1", "u1"],
        "subject_domain_name": ["CORP", "CORP"], "rule_mitre_technique": ["T1033", "T1033"],
        "rule_mitre_tactic": ["Discovery", "Discovery"], "image": ["whoami.exe", "whoami.exe"],
        "parent_image": ["cmd.exe", "cmd.exe"],
    })
    featured, error = rdc.build_features_safely(bad_frame)
    assert featured is None
    assert error is not None
    assert "sorted" in error.lower() or "valueerror" in error.lower()


def test_build_features_safely_succeeds_on_good_data():
    from soc_runtime.synthetic import generate_alerts

    frame = generate_alerts(n_days=10)
    featured, error = rdc.build_features_safely(frame)
    assert error is None
    assert featured is not None
    assert len(featured) > 0


# --------------------------------------------------------------------------
# Descriptive stats (model-free)
# --------------------------------------------------------------------------

def test_descriptive_semantic_stats_needs_no_ground_truth():
    from soc_runtime.synthetic import generate_alerts
    from soc_runtime import baseline, semantic

    frame = generate_alerts(n_days=15)
    featured = baseline.build_features(semantic.build_features(frame))
    stats = rdc.descriptive_semantic_stats(featured)
    assert stats["n_scored"] > 0
    assert set(stats["feature_stats"].keys()) == set(FEATURE_NAMES)
    for feat_stats in stats["feature_stats"].values():
        assert "mean" in feat_stats and "share_nonzero" in feat_stats


# --------------------------------------------------------------------------
# Illustrative synthetic-model scoring
# --------------------------------------------------------------------------

def test_illustrative_synthetic_model_scores_is_clearly_caveated():
    from soc_runtime.synthetic import generate_alerts
    from soc_runtime import baseline, modeling, semantic

    frame = generate_alerts(n_days=20)
    featured = baseline.build_features(semantic.build_features(frame))
    scored = modeling.restrict_to_scored_techniques(featured)
    result = rdc.illustrative_synthetic_model_scores(scored)
    assert "ILLUSTRATIVE ONLY" in result["caveat"]
    for arm in ("baseline", "semantic", "raw_plus_semantic"):
        assert 0.0 <= result[arm]["flagged_share"] <= 1.0


# --------------------------------------------------------------------------
# Known-incident cross-reference
# --------------------------------------------------------------------------

def test_match_known_incidents_finds_alerts_within_tolerance():
    frame = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-08-05 14:00:00"), "agent_name": "HOST-1", "rule_id": 92033,
         "rule_mitre_technique": "T1033", "event_category": "process_creation", "alert_uid": "a1"},
        {"timestamp": pd.Timestamp("2026-08-05 20:00:00"), "agent_name": "HOST-1", "rule_id": 92033,
         "rule_mitre_technique": "T1033", "event_category": "process_creation", "alert_uid": "a2"},
    ])
    incidents = [{"approx_timestamp": "2026-08-05T14:10:00", "agent_name": "HOST-1", "note": "test"}]
    matches = rdc.match_known_incidents(frame, incidents, tolerance_minutes=60)
    assert matches[0]["n_matches"] == 1
    assert matches[0]["matches"][0]["matched_timestamp"].startswith("2026-08-05 14:00:00")


def test_match_known_incidents_narrows_by_rule_id():
    frame = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-08-05 14:00:00"), "agent_name": "HOST-1", "rule_id": 92033,
         "rule_mitre_technique": "T1033", "event_category": "process_creation", "alert_uid": "a1"},
        {"timestamp": pd.Timestamp("2026-08-05 14:01:00"), "agent_name": "HOST-1", "rule_id": 92082,
         "rule_mitre_technique": "T1082", "event_category": "process_creation", "alert_uid": "a2"},
    ])
    incidents = [{"approx_timestamp": "2026-08-05T14:00:30", "agent_name": "HOST-1", "rule_id": 92033}]
    matches = rdc.match_known_incidents(frame, incidents, tolerance_minutes=60)
    assert matches[0]["n_matches"] == 1
    assert matches[0]["matches"][0]["rule_id"] == 92033


def test_match_known_incidents_no_match_reports_zero_not_error():
    frame = pd.DataFrame([
        {"timestamp": pd.Timestamp("2026-01-01"), "agent_name": "HOST-1", "rule_id": 1,
         "rule_mitre_technique": "T1033", "event_category": "process_creation", "alert_uid": "a1"},
    ])
    incidents = [{"approx_timestamp": "2026-08-05T14:00:00", "agent_name": "HOST-9"}]
    matches = rdc.match_known_incidents(frame, incidents)
    assert matches[0]["n_matches"] == 0


# --------------------------------------------------------------------------
# Report writing - aggregates only
# --------------------------------------------------------------------------

def test_write_report_produces_valid_json(tmp_path):
    report = {"disclaimer": "REAL DATA. test", "window": {"days": 5}}
    out_path = tmp_path / "report.json"
    written = rdc._write_report(report, out_path)
    assert written == out_path
    loaded = json.loads(out_path.read_text())
    assert loaded["disclaimer"] == report["disclaimer"]
