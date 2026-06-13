"""Tests for overlay_gap_impact_analyzer.py (Task 2 — gap impact analysis v1).

Unit tests use synthetic minimal data so they run without any file I/O.
Integration tests run the full analysis once (via lru_cache) against the
committed overlay artifacts and assert on its summary values.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile
from functools import lru_cache

import pytest

from worldpgt.experiments.overlay_gap_impact_analyzer import (
    BLOCKER_TO_EFFECT,
    BLOCKER_TO_REQUIRED_INPUT,
    TARGETABLE_BLOCKERS,
    analyze_accepted_auto_usage,
    analyze_unchanged_audits,
    build_acquisition_plan,
    build_analysis_csv_rows,
    choose_recommended_step,
    classify_row_changes,
    explain_changed_rows,
    extract_overlay_cue_matches,
    run_analysis,
)

# ---------------------------------------------------------------------------
# Paths to committed artifacts
# ---------------------------------------------------------------------------

_EXPERIMENTS = os.path.join(
    os.path.dirname(__file__), "..", "experiments"
)


def _exp(name: str) -> str:
    return os.path.join(_EXPERIMENTS, name)


# ---------------------------------------------------------------------------
# Integration fixture — run once per test session
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _run_full_analysis_cached() -> dict:
    """Run the full analysis against committed artifacts; cache the result."""
    def _csv(path: str) -> list[dict]:
        with open(path, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    def _json(path: str) -> dict:
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    return run_analysis(
        baseline_rows=_csv(_exp("microworld_continuation_v1_2_outputs.csv")),
        overlay_rows=_csv(_exp("knowledge_overlay_v1_outputs.csv")),
        overlay_summary=_json(_exp("knowledge_overlay_v1_summary.json")),
        audit_reasons_data=_json(_exp("microworld_continuation_v1_2_audit_reasons.json")),
        curriculum_data=_json(_exp("microworld_continuation_v1_2_coverage_gap_curriculum.json")),
        auto_review_data=_json(_exp("knowledge_ingestion_v1_auto_review.json")),
    )


# ---------------------------------------------------------------------------
# Unit tests — extract_overlay_cue_matches
# ---------------------------------------------------------------------------

def test_extract_overlay_cue_matches_empty_hits():
    rows = [{"id": "r1", "memory_hits": "term=bat | cue=ball -> sports_equipment"}]
    assert extract_overlay_cue_matches(rows) == {}


def test_extract_overlay_cue_matches_single_cue():
    rows = [{
        "id": "r1",
        "memory_hits": "positive_cue=pitcher -> sports_equipment | overlay_cue=pitcher -> bat:sports_equipment",
    }]
    result = extract_overlay_cue_matches(rows)
    assert "r1" in result
    assert result["r1"] == [("pitcher", "bat", "sports_equipment")]


def test_extract_overlay_cue_matches_multiple_cues():
    rows = [{
        "id": "r2",
        "memory_hits": (
            "overlay_cue=wing -> crane:bird | "
            "overlay_cue=construction site -> crane:machine"
        ),
    }]
    result = extract_overlay_cue_matches(rows)
    assert len(result["r2"]) == 2
    assert ("wing", "crane", "bird") in result["r2"]
    assert ("construction site", "crane", "machine") in result["r2"]


# ---------------------------------------------------------------------------
# Unit tests — classify_row_changes
# ---------------------------------------------------------------------------

def _make_row(rid, decision, sense="", cont="", term="", memory_hits=""):
    return {"id": rid, "decision": decision, "selected_sense": sense,
            "continuation": cont, "ambiguous_term": term, "memory_hits": memory_hits}


def test_classify_row_changes_audit_to_continue():
    baseline = [_make_row("r1", "audit")]
    overlay = [_make_row("r1", "continue", sense="sports_equipment",
                         memory_hits="overlay_cue=pitcher -> bat:sports_equipment")]
    overlay[0]["memory_hits"] = "overlay_cue=pitcher -> bat:sports_equipment"
    result = classify_row_changes(baseline, overlay)
    assert result[0]["change_type"] == "audit_to_continue"
    assert result[0]["changed"] is True


def test_classify_row_changes_continue_to_audit():
    baseline = [_make_row("r1", "continue", sense="machine")]
    overlay = [_make_row("r1", "audit")]
    result = classify_row_changes(baseline, overlay)
    assert result[0]["change_type"] == "continue_to_audit"
    assert result[0]["changed"] is True


def test_classify_row_changes_unchanged_audit():
    baseline = [_make_row("r1", "audit")]
    overlay = [_make_row("r1", "audit")]
    result = classify_row_changes(baseline, overlay)
    assert result[0]["change_type"] == "unchanged_audit"
    assert result[0]["changed"] is False


def test_classify_row_changes_unchanged_continue():
    cont = "The bat flew silently."
    baseline = [_make_row("r1", "continue", sense="animal", cont=cont)]
    overlay = [_make_row("r1", "continue", sense="animal", cont=cont)]
    result = classify_row_changes(baseline, overlay)
    assert result[0]["change_type"] == "unchanged_continue"
    assert result[0]["changed"] is False


# ---------------------------------------------------------------------------
# Unit tests — explain_changed_rows
# ---------------------------------------------------------------------------

def _make_overlay_row_explained(rid, b_dec, o_dec, cue_marker, reasons, term="bat",
                                 o_sense="", exp_sense=""):
    return {
        "id": rid, "decision": o_dec, "ambiguous_term": term,
        "selected_sense": o_sense, "expected_sense": exp_sense,
        "reasons": reasons,
        "memory_hits": f"positive_cue={cue_marker} | overlay_cue={cue_marker}",
        "continuation": "",
    }


def test_explain_changed_rows_audit_to_continue():
    baseline = [{"id": "r1", "decision": "audit", "selected_sense": "", "continuation": ""}]
    overlay = [_make_overlay_row_explained(
        "r1", "audit", "continue",
        cue_marker="pitcher -> bat:sports_equipment",
        reasons="top_sense=sports_equipment | top_score=1 | second_score=0 | margin=1",
        o_sense="sports_equipment", exp_sense="sports_equipment",
    )]
    explanations = explain_changed_rows(baseline, overlay, {})
    assert len(explanations) == 1
    exp = explanations[0]
    assert exp["change_type"] == "audit_to_continue"
    assert exp["mechanism"] == "new_cue_broke_cue_deficit"
    assert exp["correct_direction"] is True
    assert exp["overlay_cue"] == "pitcher"


def test_explain_changed_rows_continue_to_audit():
    baseline = [{"id": "r1", "decision": "continue", "selected_sense": "machine",
                 "continuation": "The crane lifted the beam."}]
    overlay = [_make_overlay_row_explained(
        "r1", "continue", "audit",
        cue_marker="wing -> crane:bird",
        reasons="top_sense=machine | top_score=2 | second_score=1 | margin=1 | conflict_detected",
        term="crane", o_sense="", exp_sense="bird",
    )]
    explanations = explain_changed_rows(baseline, overlay, {})
    assert len(explanations) == 1
    exp = explanations[0]
    assert exp["change_type"] == "continue_to_audit"
    assert exp["mechanism"] == "new_cue_introduced_conflict"
    # baseline was selecting machine but expected bird → auditing is correct
    assert exp["correct_direction"] is True


# ---------------------------------------------------------------------------
# Unit tests — analyze_unchanged_audits
# ---------------------------------------------------------------------------

def _make_audit_row(rid, term="bat", reasons="", hits=""):
    return {"id": rid, "decision": "audit", "ambiguous_term": term,
            "reasons": reasons, "memory_hits": hits, "continuation": ""}


def test_analyze_unchanged_audits_newly_audited_is_low_margin_conflict():
    baseline = [{"id": "r1", "decision": "continue", "selected_sense": "machine",
                 "continuation": "", "reasons": ""}]
    overlay = [_make_audit_row("r1", term="crane",
                               reasons="top_score=2 | second_score=1 | margin=1 | conflict_detected",
                               hits="overlay_cue=wing -> crane:bird")]
    items = analyze_unchanged_audits(overlay, baseline, {}, {})
    assert items[0]["is_newly_audited"] is True
    assert items[0]["blocker_type"] == "low_margin_conflict"
    assert items[0]["targetable"] is False
    assert items[0]["why_overlay_insufficient"] == "newly_audited_by_overlay"


def test_analyze_unchanged_audits_blocker_from_curriculum():
    baseline = [{"id": "r1", "decision": "audit", "continuation": ""}]
    overlay = [_make_audit_row("r1", term="spring")]
    curriculum_by_id = {"r1": {"task_id": "task-01", "gap_type": "missing_phrase_candidate"}}
    items = analyze_unchanged_audits(overlay, baseline, {}, curriculum_by_id)
    assert items[0]["blocker_type"] == "missing_phrase_candidate"
    assert items[0]["targetable"] is True
    assert items[0]["curriculum_task_id"] == "task-01"


def test_analyze_unchanged_audits_blocker_from_audit_reason():
    baseline = [{"id": "r1", "decision": "audit", "continuation": ""}]
    overlay = [_make_audit_row("r1", term="bat")]
    ar_by_id = {"r1": {"row_id": "r1", "primary_audit_reason": "sense_tie"}}
    items = analyze_unchanged_audits(overlay, baseline, ar_by_id, {})
    assert items[0]["blocker_type"] == "missing_positive_cue"
    assert items[0]["targetable"] is True


def test_analyze_unchanged_audits_why_overlay_insufficient_conflict_at_min_margin():
    baseline = [{"id": "r1", "decision": "audit", "continuation": ""}]
    overlay = [_make_audit_row(
        "r1", term="crane",
        reasons="top_score=3 | second_score=2 | margin=1 | conflict_detected",
        hits="overlay_cue=construction site -> crane:machine",
    )]
    ar_by_id = {"r1": {"row_id": "r1", "primary_audit_reason": "sense_tie"}}
    items = analyze_unchanged_audits(overlay, baseline, ar_by_id, {})
    assert items[0]["overlay_influenced"] is True
    assert items[0]["why_overlay_insufficient"] == "conflict_persists_at_min_margin"


# ---------------------------------------------------------------------------
# Unit tests — analyze_accepted_auto_usage
# ---------------------------------------------------------------------------

def _minimal_auto_review(term, sense, item_type, value):
    return {
        "proposal_reviews": [{
            "term": term,
            "sense": sense,
            "items": [{"decision": "accepted_auto", "item_type": item_type, "value": value}],
        }]
    }


def test_analyze_accepted_auto_usage_duplicate_builtin():
    # "baseball" is already in bat:sports_equipment builtin
    data = _minimal_auto_review("bat", "sports_equipment", "positive_cue", "baseball")
    result = analyze_accepted_auto_usage(data, [], [])
    assert result[0]["usage_category"] == "duplicate_builtin_memory"


def test_analyze_accepted_auto_usage_used_changed_decision():
    data = _minimal_auto_review("bat", "sports_equipment", "positive_cue", "pitcher")
    overlay_rows = [{
        "id": "v1-044", "decision": "continue", "ambiguous_term": "bat",
        "memory_hits": "overlay_cue=pitcher -> bat:sports_equipment",
    }]
    baseline_rows = [{"id": "v1-044", "decision": "audit"}]
    result = analyze_accepted_auto_usage(data, overlay_rows, baseline_rows)
    assert result[0]["usage_category"] == "used_changed_decision"
    assert "v1-044" in result[0]["matched_row_ids"]


def test_analyze_accepted_auto_usage_not_observed():
    # "glacier" won't appear in any prompt
    data = _minimal_auto_review("bat", "sports_equipment", "positive_cue", "glacier")
    result = analyze_accepted_auto_usage(data, [], [])
    assert result[0]["usage_category"] == "not_observed_in_prompts"


def test_analyze_accepted_auto_usage_unknown_type():
    data = _minimal_auto_review("bat", "sports_equipment", "typical_action", "swing at pitches")
    result = analyze_accepted_auto_usage(data, [], [])
    assert result[0]["usage_category"] == "unknown"


# ---------------------------------------------------------------------------
# Unit test — build_acquisition_plan
# ---------------------------------------------------------------------------

def test_build_acquisition_plan_targetable_only():
    items = [
        {"row_id": "r1", "term": "bat", "blocker_type": "missing_phrase_candidate",
         "targetable": True, "curriculum_task_id": "t-01"},
        {"row_id": "r2", "term": "bat", "blocker_type": "true_unsafe_keep_audit",
         "targetable": False, "curriculum_task_id": ""},
        {"row_id": "r3", "term": "crane", "blocker_type": "missing_positive_cue",
         "targetable": True, "curriculum_task_id": ""},
    ]
    plan = build_acquisition_plan(items, {})
    row_ids = [p["row_id"] for p in plan]
    assert "r1" in row_ids
    assert "r3" in row_ids
    assert "r2" not in row_ids


# ---------------------------------------------------------------------------
# Unit test — choose_recommended_step
# ---------------------------------------------------------------------------

def test_choose_recommended_step_phrase_candidates_dominant():
    plan = (
        [{"blocker_type": "missing_phrase_candidate"}] * 10
        + [{"blocker_type": "missing_positive_cue"}] * 8
    )
    assert choose_recommended_step(plan) == "expand_phrase_candidates"


def test_choose_recommended_step_no_items():
    assert choose_recommended_step([]) == "no_targetable_gaps"


# ---------------------------------------------------------------------------
# Integration tests — full run against committed artifacts
# ---------------------------------------------------------------------------

def test_full_run_newly_continued_count():
    analysis = _run_full_analysis_cached()
    assert analysis["summary"]["newly_continued_count"] == 1


def test_full_run_net_coverage_zero():
    analysis = _run_full_analysis_cached()
    assert analysis["summary"]["net_trusted_coverage_change"] == 0


def test_full_run_targetable_audit_count():
    analysis = _run_full_analysis_cached()
    assert analysis["summary"]["targetable_audit_count"] == 38


def test_full_run_keep_audit_count():
    analysis = _run_full_analysis_cached()
    assert analysis["summary"]["keep_audit_count"] == 24


def test_full_run_recommended_step_is_expand_phrase_candidates():
    analysis = _run_full_analysis_cached()
    assert analysis["summary"]["recommended_next_step"] == "expand_phrase_candidates"


def test_full_run_accepted_auto_usage_total():
    analysis = _run_full_analysis_cached()
    assert analysis["summary"]["accepted_auto_usage_summary"]["total"] == 163


def test_full_run_v1_044_resolved_by_overlay():
    analysis = _run_full_analysis_cached()
    changed = {e["row_id"]: e for e in analysis["changed_rows"]}
    assert "v1-044" in changed
    assert changed["v1-044"]["change_type"] == "audit_to_continue"
    assert changed["v1-044"]["mechanism"] == "new_cue_broke_cue_deficit"


def test_full_run_v1_110_newly_audited():
    analysis = _run_full_analysis_cached()
    changed = {e["row_id"]: e for e in analysis["changed_rows"]}
    assert "v1-110" in changed
    assert changed["v1-110"]["change_type"] == "continue_to_audit"
    assert changed["v1-110"]["mechanism"] == "new_cue_introduced_conflict"


def test_full_run_acquisition_plan_has_no_keep_audit_rows():
    analysis = _run_full_analysis_cached()
    for item in analysis["targeted_acquisition_plan"]:
        assert item["blocker_type"] in TARGETABLE_BLOCKERS, (
            f"{item['row_id']} has non-targetable blocker {item['blocker_type']} in plan"
        )


def test_full_run_risk_regression_count_zero():
    analysis = _run_full_analysis_cached()
    assert analysis["summary"]["risk_regression_count"] == 0


def test_full_run_analysis_csv_rows_count():
    # 62 overlay-audit rows + 1 audit_to_continue = 63 rows
    analysis = _run_full_analysis_cached()
    assert len(analysis["analysis_csv_rows"]) == 63


def test_full_run_duplicate_builtin_count():
    analysis = _run_full_analysis_cached()
    usage = analysis["summary"]["accepted_auto_usage_summary"]
    # 45 of the 104 positive_cues were already in builtin
    assert usage.get("duplicate_builtin_memory", 0) == 45


def test_full_run_not_observed_positive_cues_majority():
    # Most new cues don't appear in the 120 prompts
    analysis = _run_full_analysis_cached()
    usage = analysis["summary"]["accepted_auto_usage_summary"]
    not_observed = usage.get("not_observed_in_prompts", 0)
    used = usage.get("used_changed_decision", 0) + usage.get("used_no_decision_change", 0)
    assert not_observed > used, (
        f"Expected most new cues to go unobserved; got not_observed={not_observed}, used={used}"
    )
