"""Multi-valued `rule.mitre.technique` handling.

57.7% of real process-creation alerts carry more than one MITRE technique.
These tests pin down the three things that actually matter about the fix:
the pre-fix mode still reproduces the pre-fix behaviour exactly (so it is a
usable baseline), the state fold no longer drops techniques (the actual
defect), and the aggregation collapses each feature in the direction it was
designed to.

Everything here is hand-built frames - no generator, no network.
"""

from __future__ import annotations

import pandas as pd
import pytest

from soc_runtime import config, modeling, semantic
from soc_runtime.opensearch_client import all_of_multivalued
from soc_runtime.semantic import (
    TACTIC_ALL_COLUMN, TECHNIQUE_ALL_COLUMN, _value_lists, build_features, feature_names,
)

_ALL_MODES = config.MULTI_TECHNIQUE_MODES


def _frame(rows: list[dict]) -> pd.DataFrame:
    """Minimal process-creation frame; `rows` supplies only what varies."""
    base = {
        "cluster_node": "office-collector", "agent_name": "H1", "agent_ip": "10.0.0.1",
        "rule_id": 1, "rule_level": 5, "rule_description": "d",
        "event_category": "process_creation",
        "subject_user_name": "alice", "subject_domain_name": "CORP", "eventdata_user": None,
        "target_user_name": "alice", "target_domain_name": "CORP",
        "command_line": "c", "image": "i.exe", "parent_image": "p.exe",
        "parent_command_line": "pc", "parent_process_guid": "{g}", "process_guid": "{h}",
        "logon_type": None, "authentication_package_name": None,
        "is_synthetic_anomaly": False, "synthetic_anomaly_type": None,
    }
    out = []
    for i, row in enumerate(rows):
        merged = {**base, "alert_uid": f"a{i}", **row}
        merged.setdefault("timestamp", pd.Timestamp("2026-01-01 09:00") + pd.Timedelta(minutes=i))
        out.append(merged)
    return pd.DataFrame(out)


# --------------------------------------------------------------------------
# all_of_multivalued - the adapter-side helper
# --------------------------------------------------------------------------

def test_all_of_multivalued_normalises_every_shape():
    assert all_of_multivalued(["T1033", "T1059"]) == ["T1033", "T1059"]
    assert all_of_multivalued("T1033") == ["T1033"]
    assert all_of_multivalued(None) == []
    assert all_of_multivalued([]) == []


def test_all_of_multivalued_preserves_order_and_drops_duplicates_and_nones():
    assert all_of_multivalued(["T1059", "T1033", "T1059", None]) == ["T1059", "T1033"]


# --------------------------------------------------------------------------
# _value_lists - the invariants the rest of the module relies on
# --------------------------------------------------------------------------

def test_value_lists_falls_back_to_scalar_column_when_all_column_absent():
    frame = _frame([{"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"}])
    assert _value_lists(frame, TECHNIQUE_ALL_COLUMN, "rule_mitre_technique") == [["T1033"]]


def test_value_lists_never_returns_an_empty_list_for_a_null_field():
    """A null technique must stay addressable as `[None]` - it still keys the
    causal state exactly as it did before, rather than vanishing."""
    frame = _frame([{"rule_mitre_technique": None, "rule_mitre_tactic": None,
                     TECHNIQUE_ALL_COLUMN: [], TACTIC_ALL_COLUMN: []}])
    assert _value_lists(frame, TECHNIQUE_ALL_COLUMN, "rule_mitre_technique") == [[None]]


def test_value_lists_puts_the_scalar_column_value_first():
    """The invariant that makes `first_only` an honest baseline: reading
    `list[0]` must be identical to reading the old scalar column, whatever
    order the `_all` column happens to be in."""
    frame = _frame([{"rule_mitre_technique": "T1059", "rule_mitre_tactic": "Execution",
                     TECHNIQUE_ALL_COLUMN: ["T1033", "T1059"], TACTIC_ALL_COLUMN: ["Discovery", "Execution"]}])
    assert _value_lists(frame, TECHNIQUE_ALL_COLUMN, "rule_mitre_technique")[0][0] == "T1059"


# --------------------------------------------------------------------------
# Mode plumbing
# --------------------------------------------------------------------------

def test_feature_names_only_the_multi_modes_carry_the_multiplicity_feature():
    assert feature_names("first_only") == semantic.FEATURE_NAMES
    for mode in ("primary_plus_count", "aggregate"):
        assert feature_names(mode) == semantic.FEATURE_NAMES + (semantic.MULTI_TECHNIQUE_FEATURE[0],)


def test_feature_names_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="unknown multi-technique mode"):
        feature_names("take-the-last-one")


def test_default_mode_is_first_only_so_published_v1_numbers_reproduce():
    """Not a style preference: `primary_plus_count` adds a column that is
    constant on single-technique corpora, and even a constant column shifts
    a fitted LogisticRegression (v1 PR-AUC 0.975616 -> 0.975429). The
    default has to stay the mode every published number was measured under."""
    assert config.MULTI_TECHNIQUE_MODE == "first_only"


def test_real_data_path_defaults_to_the_corrected_mode():
    """The global default exists for v1 reproducibility; the real-data path
    has no such constraint and should not inherit a known-wrong fold."""
    assert config.REAL_DATA_MULTI_TECHNIQUE_MODE == "primary_plus_count"


def test_arm_features_tracks_the_mode():
    assert modeling.arm_features("first_only") == modeling.ARM_FEATURES
    union = modeling.arm_features("aggregate")["raw_plus_semantic"]
    assert semantic.MULTI_TECHNIQUE_FEATURE[0] in union


# --------------------------------------------------------------------------
# Single-technique corpora must be unaffected by the mode
# --------------------------------------------------------------------------

def test_modes_agree_exactly_when_no_alert_carries_multiple_techniques():
    """The control: with nothing to fix, the fix must change nothing. Any
    drift here would mean the mode plumbing itself alters results."""
    frame = _frame([
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"},
        {"rule_mitre_technique": "T1059", "rule_mitre_tactic": "Execution"},
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"},
    ])
    reference = build_features(frame, mode="first_only")
    for mode in ("primary_plus_count", "aggregate"):
        other = build_features(frame, mode=mode)
        pd.testing.assert_frame_equal(
            reference[list(semantic.FEATURE_NAMES)], other[list(semantic.FEATURE_NAMES)],
        )


def test_first_only_ignores_the_all_columns_entirely():
    """`first_only` must be bit-identical whether or not the corpus carries
    the `_all` columns - otherwise it is not the pre-fix behaviour and every
    before/after comparison built on it measures two changes at once."""
    without = _frame([
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"},
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"},
    ])
    with_all = _frame([
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033", "T1059"], TACTIC_ALL_COLUMN: ["Discovery", "Execution"]},
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033", "T1059"], TACTIC_ALL_COLUMN: ["Discovery", "Execution"]},
    ])
    a = build_features(without, mode="first_only")[list(semantic.FEATURE_NAMES)]
    b = build_features(with_all, mode="first_only")[list(semantic.FEATURE_NAMES)]
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


# --------------------------------------------------------------------------
# The actual defect: the causal state used to drop secondary techniques
# --------------------------------------------------------------------------

def test_secondary_technique_is_folded_into_state_so_a_later_alert_is_not_falsely_novel():
    """The core regression this whole change exists for. Alert 1 carries
    T1033 *and* T1059; alert 2 carries T1059 alone. Under `first_only` the
    state never learned T1059, so alert 2 is scored as a never-seen triple -
    which is simply wrong: the same actor ran it on the same host a minute
    earlier."""
    frame = _frame([
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033", "T1059"], TACTIC_ALL_COLUMN: ["Discovery", "Execution"]},
        {"rule_mitre_technique": "T1059", "rule_mitre_tactic": "Execution",
         TECHNIQUE_ALL_COLUMN: ["T1059"], TACTIC_ALL_COLUMN: ["Execution"]},
    ])
    stale = build_features(frame, mode="first_only")
    assert stale["sem_novel_actor_host_technique"].tolist() == [1.0, 1.0]

    for mode in ("primary_plus_count", "aggregate"):
        fixed = build_features(frame, mode=mode)
        assert fixed["sem_novel_actor_host_technique"].tolist() == [1.0, 0.0], mode
        assert fixed["sem_freq_actor_host_technique_24h"].tolist()[1] == 1.0, mode


def test_secondary_tactic_is_folded_so_tactic_diversity_can_see_it():
    """`sem_chain_tactic_diversity_15m` exists to count distinct tactics.
    Under `first_only` an alert tagged Discovery *and* Execution contributes
    one of them, so the feature cannot observe the very thing it measures."""
    frame = _frame([
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033", "T1059"], TACTIC_ALL_COLUMN: ["Discovery", "Execution"]},
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033"], TACTIC_ALL_COLUMN: ["Discovery"]},
    ])
    assert build_features(frame, mode="first_only")["sem_chain_tactic_diversity_15m"].tolist() == [1.0, 1.0]
    for mode in ("primary_plus_count", "aggregate"):
        seen = build_features(frame, mode=mode)["sem_chain_tactic_diversity_15m"].tolist()
        assert seen[1] == 2.0, mode


# --------------------------------------------------------------------------
# The multiplicity feature (Option A's own contribution)
# --------------------------------------------------------------------------

def test_multiplicity_feature_counts_the_whole_alert_not_the_read_set():
    frame = _frame([
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033", "T1059", "T1087"], TACTIC_ALL_COLUMN: ["Discovery", "Execution"]},
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033"], TACTIC_ALL_COLUMN: ["Discovery"]},
    ])
    name = semantic.MULTI_TECHNIQUE_FEATURE[0]
    # Even in primary_plus_count, where secondaries move nothing else.
    assert build_features(frame, mode="primary_plus_count")[name].tolist() == [3.0, 1.0]
    assert build_features(frame, mode="aggregate")[name].tolist() == [3.0, 1.0]


def test_multiplicity_feature_absent_in_first_only():
    frame = _frame([{"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery"}])
    assert semantic.MULTI_TECHNIQUE_FEATURE[0] not in build_features(frame, mode="first_only").columns


# --------------------------------------------------------------------------
# Aggregation direction (Option C's defining choice)
# --------------------------------------------------------------------------

def test_aggregate_collapses_frequency_toward_the_least_established_technique():
    """`min`, not `max`. Three prior T1033 alerts and no prior T1059; the
    fourth alert carries both. Reporting the *most*-established technique's
    count would hide the unexplained one behind the noisy one - exactly the
    information loss the change exists to remove."""
    rows = [
        {"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
         TECHNIQUE_ALL_COLUMN: ["T1033"], TACTIC_ALL_COLUMN: ["Discovery"]}
        for _ in range(3)
    ]
    rows.append({"rule_mitre_technique": "T1033", "rule_mitre_tactic": "Discovery",
                 TECHNIQUE_ALL_COLUMN: ["T1033", "T1059"], TACTIC_ALL_COLUMN: ["Discovery", "Execution"]})
    frame = _frame(rows)

    primary = build_features(frame, mode="primary_plus_count")
    aggregated = build_features(frame, mode="aggregate")
    # Reading the primary alone sees three prior T1033s...
    assert primary["sem_freq_actor_host_technique_24h"].tolist()[3] == 3.0
    # ...aggregating surfaces the never-before-seen T1059 instead.
    assert aggregated["sem_freq_actor_host_technique_24h"].tolist()[3] == 0.0
    assert aggregated["sem_novel_actor_host_technique"].tolist()[3] == 1.0
    assert primary["sem_novel_actor_host_technique"].tolist()[3] == 0.0


def test_aggregation_map_covers_only_technique_keyed_features():
    """Guard against a future feature being added to `_AGGREGATION` that is
    not actually per-technique - it would be silently collapsed over a
    dimension it does not vary on."""
    assert set(semantic._AGGREGATION) == {
        "sem_freq_actor_host_technique_24h",
        "sem_freq_actor_technique_7d",
        "sem_novel_actor_host_technique",
    }
    assert set(semantic._AGGREGATION) <= set(semantic.FEATURE_NAMES)


def test_discovery_to_escalation_needs_a_prior_alert_not_a_simultaneous_tag():
    """An alert tagged Discovery and Credential Access at once is not a
    Discovery-*then*-escalation sequence. Counting it as one would redefine
    the feature rather than let it see more."""
    frame = _frame([
        {"rule_mitre_technique": "T1003", "rule_mitre_tactic": "Credential Access",
         TECHNIQUE_ALL_COLUMN: ["T1003", "T1033"],
         TACTIC_ALL_COLUMN: ["Credential Access", "Discovery"]},
    ])
    for mode in _ALL_MODES:
        built = build_features(frame, mode=mode)
        assert built["sem_chain_discovery_to_escalation_15m"].tolist() == [0.0], mode
