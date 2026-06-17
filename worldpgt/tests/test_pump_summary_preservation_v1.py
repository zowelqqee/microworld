"""Knowledge Pump Summary Preservation v1 — deterministic unit tests.

Verifies that merge_qa_preservation() correctly preserves compact Pump Fact QA
fields across plan-only pump runs, using the canonical QA artifact as the
authoritative source, with honest freshness marking.

No network. No overlay writes. No memory modifications.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.knowledge_pump.pump_summary_qa_preserver import (
    _QA_COMPACT_TARGET_FIELDS,
    merge_qa_preservation,
)

# ---------------------------------------------------------------------------
# Minimal QA artifact fixture (mirrors pump_fact_qa_summary.json shape)
# ---------------------------------------------------------------------------

_SAMPLE_QA_ARTIFACT = {
    "pump_fact_qa_enabled": True,
    "pump_fact_qa_fact_count": 196,
    "pump_fact_qa_relation_fact_count": 38,
    "pump_fact_qa_definition_fact_count": 158,
    "pump_fact_qa_positive_prompt_count": 388,
    "pump_fact_qa_total_prompt_count": 423,
    "pump_fact_qa_positive_answer_count": 388,
    "pump_fact_qa_positive_audit_count": 0,
    "pump_fact_qa_positive_planner_gap_count": 0,
    "pump_fact_qa_positive_wrong_count": 0,
    "pump_fact_qa_positive_unsupported_count": 0,
    "pump_fact_qa_adversarial_fail_count": 0,
    "pump_fact_qa_current_safety_fail_count": 0,
    "pump_fact_qa_answer_precision": 1.0,
    "pump_fact_qa_supported_answer_rate": 1.0,
    "pump_fact_qa_planner_gap_rate": 0.0,
    "pump_fact_qa_all_critical_passed": True,
    # v1.8 short-name aliases
    "fact_count": 196,
    "wrong_answer_count": 0,
    "unsupported_answer_count": 0,
    "adversarial_fail_count": 0,
    "current_safety_fail_count": 0,
    "answer_precision": 1.0,
    "supported_answer_rate": 1.0,
    "planner_gap_rate": 0.0,
    "all_critical_passed": True,
}

_REQUIRED_COMPACT_FIELDS = [
    "pump_fact_qa_fact_count",
    "pump_fact_qa_relation_fact_count",
    "pump_fact_qa_definition_fact_count",
    "pump_fact_qa_positive_answer_count",
    "pump_fact_qa_positive_planner_gap_count",
    "pump_fact_qa_wrong_count",
    "pump_fact_qa_unsupported_answer_count",
    "pump_fact_qa_all_critical_passed",
    "pump_fact_qa_status",
    "pump_fact_qa_stale",
    "pump_fact_qa_preserved",
]


def _write_qa_artifact(tmp_path: Path, data: dict) -> None:
    qa_dir = tmp_path / "pump_fact_qa_v1"
    qa_dir.mkdir(parents=True, exist_ok=True)
    (qa_dir / "pump_fact_qa_summary.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def _fresh_summary() -> dict:
    """A freshly built pump summary with no QA fields."""
    return {
        "plan_only": True,
        "pump_answerable_fact_delta_count": 196,
        "pump_fact_qa_fact_count_before_v2": 13,
        "pump_fact_qa_fact_count_after_v2": 196,
    }


# ===========================================================================
# Category 1: QA artifact exists, fact count matches
# ===========================================================================


def test_current_artifact_sets_status(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_status"] == "current_from_qa_artifact"


def test_current_artifact_stale_is_false(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_stale"] is False


def test_current_artifact_preserved_is_true(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_preserved"] is True


# ===========================================================================
# Category 2: Preserved summary includes all required compact fields
# ===========================================================================


@pytest.mark.parametrize("field", _REQUIRED_COMPACT_FIELDS)
def test_required_field_present_after_preservation(tmp_path, field):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert field in summary, f"Field {field!r} missing from preserved summary"


def test_fact_count_preserved_correctly(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_fact_count"] == 196


def test_relation_fact_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_relation_fact_count"] == 38


def test_definition_fact_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_definition_fact_count"] == 158


def test_positive_answer_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_positive_answer_count"] == 388


def test_positive_planner_gap_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_positive_planner_gap_count"] == 0


def test_wrong_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_wrong_count"] == 0


def test_unsupported_answer_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_unsupported_answer_count"] == 0


def test_all_critical_passed_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_all_critical_passed"] is True


def test_supported_answer_rate_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_supported_answer_rate"] == 1.0


def test_adversarial_fail_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_adversarial_fail_count"] == 0


def test_current_safety_fail_count_preserved(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_current_safety_fail_count"] == 0


# ===========================================================================
# Category 3: Matching fact count → current_from_qa_artifact, stale=False
# ===========================================================================


def test_matching_fact_count_not_stale(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_stale"] is False
    assert summary["pump_fact_qa_status"] == "current_from_qa_artifact"


def test_matching_fact_count_no_expected_field(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert "pump_fact_qa_expected_fact_count" not in summary


# ===========================================================================
# Category 4: Mismatched fact count → stale_fact_count_mismatch, stale=True
# ===========================================================================


def test_mismatch_sets_stale_status(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=210)
    assert summary["pump_fact_qa_status"] == "stale_fact_count_mismatch"


def test_mismatch_stale_is_true(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=210)
    assert summary["pump_fact_qa_stale"] is True


def test_mismatch_preserved_is_true(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=210)
    assert summary["pump_fact_qa_preserved"] is True


def test_mismatch_includes_expected_fact_count(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=210)
    assert summary["pump_fact_qa_expected_fact_count"] == 210


def test_mismatch_includes_artifact_fact_count(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=210)
    assert summary["pump_fact_qa_artifact_fact_count"] == 196


def test_mismatch_still_preserves_qa_fields(tmp_path):
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=210)
    assert summary.get("pump_fact_qa_fact_count") == 196
    assert summary.get("pump_fact_qa_all_critical_passed") is True


# ===========================================================================
# Category 5: No QA artifact, old summary has QA keys → preserved_from_previous
# ===========================================================================


def _write_old_pump_summary(tmp_path: Path, qa_fields: dict) -> None:
    base = {"plan_only": True, "pump_answerable_fact_delta_count": 196}
    base.update(qa_fields)
    (tmp_path / "pump_summary.json").write_text(
        json.dumps(base, indent=2), encoding="utf-8"
    )


def test_no_artifact_old_summary_preserved_status(tmp_path):
    _write_old_pump_summary(
        tmp_path,
        {"pump_fact_qa_fact_count": 150, "pump_fact_qa_all_critical_passed": True},
    )
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_status"] == "preserved_from_previous_summary"


def test_no_artifact_old_summary_stale_is_true(tmp_path):
    _write_old_pump_summary(
        tmp_path,
        {"pump_fact_qa_fact_count": 150, "pump_fact_qa_all_critical_passed": True},
    )
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_stale"] is True


def test_no_artifact_old_summary_preserved_is_true(tmp_path):
    _write_old_pump_summary(
        tmp_path,
        {"pump_fact_qa_fact_count": 150, "pump_fact_qa_all_critical_passed": True},
    )
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_preserved"] is True


def test_no_artifact_old_qa_fields_carried_over(tmp_path):
    _write_old_pump_summary(
        tmp_path,
        {"pump_fact_qa_fact_count": 150, "pump_fact_qa_all_critical_passed": False},
    )
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_fact_count"] == 150
    assert summary["pump_fact_qa_all_critical_passed"] is False


# ===========================================================================
# Category 6: No QA artifact, no old QA keys → missing
# ===========================================================================


def test_no_artifact_no_old_summary_status_missing(tmp_path):
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_status"] == "missing"


def test_no_artifact_no_old_summary_enabled_false(tmp_path):
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_enabled"] is False


def test_no_artifact_no_old_summary_preserved_false(tmp_path):
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_preserved"] is False


def test_no_artifact_old_summary_without_qa_keys_still_missing(tmp_path):
    # Old pump_summary.json exists but has no QA fields → falls through to missing.
    (tmp_path / "pump_summary.json").write_text(
        json.dumps({"plan_only": True, "pump_answerable_fact_delta_count": 196}),
        encoding="utf-8",
    )
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_status"] == "missing"


# ===========================================================================
# Category 7: v1.8 short-name aliases are resolved correctly
# ===========================================================================


def test_v18_alias_fact_count_resolved(tmp_path):
    # QA artifact only has the v1.8 short alias, not the pump_fact_qa_* form.
    qa = {"fact_count": 180, "all_critical_passed": True}
    _write_qa_artifact(tmp_path, qa)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=180)
    assert summary["pump_fact_qa_fact_count"] == 180


def test_v18_alias_wrong_answer_count_resolved(tmp_path):
    qa = {
        "pump_fact_qa_fact_count": 196,
        "wrong_answer_count": 0,
        "all_critical_passed": True,
    }
    _write_qa_artifact(tmp_path, qa)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_wrong_count"] == 0
    assert summary["pump_fact_qa_wrong_answer_count"] == 0


def test_v18_alias_supported_answer_rate_resolved(tmp_path):
    qa = {
        "pump_fact_qa_fact_count": 196,
        "supported_answer_rate": 0.99,
        "all_critical_passed": True,
    }
    _write_qa_artifact(tmp_path, qa)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary["pump_fact_qa_supported_answer_rate"] == 0.99


# ===========================================================================
# Category 8: Integration — plan-only run preserves QA fields
# ===========================================================================

_REPO = Path(__file__).resolve().parent.parent.parent
_PUMP_OUT = _REPO / "worldpgt" / "experiments" / "knowledge_pump_v1"
_QA_ARTIFACT = _PUMP_OUT / "pump_fact_qa_v1" / "pump_fact_qa_summary.json"
_PUMP_SUMMARY = _PUMP_OUT / "pump_summary.json"


@pytest.mark.skipif(
    not _QA_ARTIFACT.exists(),
    reason="Real QA artifact not present — run run_pump_fact_qa_v1.py first",
)
def test_real_pump_summary_has_qa_fact_count():
    data = json.loads(_PUMP_SUMMARY.read_text(encoding="utf-8"))
    assert data.get("pump_fact_qa_fact_count") is not None, (
        "pump_fact_qa_fact_count missing from pump_summary.json — "
        "run run_knowledge_pump_v1.py after applying this fix"
    )


@pytest.mark.skipif(
    not _QA_ARTIFACT.exists(),
    reason="Real QA artifact not present — run run_pump_fact_qa_v1.py first",
)
def test_real_pump_summary_qa_status_present():
    data = json.loads(_PUMP_SUMMARY.read_text(encoding="utf-8"))
    assert "pump_fact_qa_status" in data


@pytest.mark.skipif(
    not _QA_ARTIFACT.exists(),
    reason="Real QA artifact not present — run run_pump_fact_qa_v1.py first",
)
def test_real_pump_summary_not_stale_when_counts_match():
    data = json.loads(_PUMP_SUMMARY.read_text(encoding="utf-8"))
    if data.get("pump_fact_qa_status") == "current_from_qa_artifact":
        assert data.get("pump_fact_qa_stale") is False


# ===========================================================================
# Category 9: Safety contract
# ===========================================================================

_PROTECTED = [
    _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json",
    _REPO / "worldpgt" / "experiments" / "accepted_wiki_memory_overlay_v1.json",
    _REPO / "worldpgt" / "experiments" / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _REPO / "worldpgt" / "experiments" / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
]


def test_preserver_does_not_touch_protected_files(tmp_path):
    import hashlib

    def _sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    before = {p: _sha(p) for p in _PROTECTED if p.exists()}
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    for p, h in before.items():
        assert _sha(p) == h, f"Protected file modified: {p}"


def test_preserver_does_not_write_network_calls(tmp_path):
    # merge_qa_preservation is pure file I/O within out_dir; it never sets
    # network_calls = True or modifies any network-gated field.
    _write_qa_artifact(tmp_path, _SAMPLE_QA_ARTIFACT)
    summary = _fresh_summary()
    merge_qa_preservation(summary, tmp_path, current_fact_count=196)
    assert summary.get("network_calls") is not True
