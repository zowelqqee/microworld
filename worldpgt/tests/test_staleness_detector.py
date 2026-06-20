"""Tests for snapshot freshness/staleness detection."""

from __future__ import annotations

from pathlib import Path

from worldpgt.knowledge.staleness_detector import detect_stale_candidates

_REPO = Path(__file__).resolve().parent.parent.parent
_PUMP_OVERLAY = _REPO / "worldpgt" / "experiments" / "knowledge_pump_v1" / "pump_dry_run_overlay.json"


def test_detects_snapshot_age_and_ratio_from_month_as_of():
    overlay = [
        {
            "overlay_type": "overlay_source_fact",
            "subject": "Elon Musk",
            "predicate": "estimated_net_worth",
            "object": "US$1.1 trillion",
            "as_of": "2026-06",
            "stability": "volatile",
            "temporal_class": "snapshot",
        }
    ]
    stale = detect_stale_candidates(overlay, "2026-06-19")
    assert len(stale) == 1
    assert stale[0].age_days == 18
    assert stale[0].freshness_window == 30
    assert stale[0].staleness_ratio == 0.6
    assert stale[0].is_stale is False


def test_historical_facts_are_not_returned():
    overlay = [
        {
            "overlay_type": "overlay_relation",
            "subject": "SpaceX",
            "predicate": "founded_by",
            "object": "Elon Musk",
            "as_of": "2020-01-01",
            "stability": "stable",
            "temporal_class": "historical",
        }
    ]
    assert detect_stale_candidates(overlay, "2026-06-19") == []


def test_stale_candidates_sorted_by_ratio_descending():
    overlay = [
        {
            "overlay_type": "overlay_source_fact",
            "subject": "Recent",
            "predicate": "ranking",
            "object": "rank 2",
            "as_of": "2026-06-01",
            "temporal_class": "snapshot",
        },
        {
            "overlay_type": "overlay_source_fact",
            "subject": "Old",
            "predicate": "estimated_net_worth",
            "object": "US$1B",
            "as_of": "2026-04-01",
            "temporal_class": "snapshot",
        },
    ]
    stale = detect_stale_candidates(overlay, "2026-06-19")
    assert [s.subject for s in stale] == ["Old", "Recent"]
    assert stale[0].staleness_ratio > stale[1].staleness_ratio


def test_real_elon_musk_net_worth_ratio_today():
    if not _PUMP_OVERLAY.exists():
        return
    stale = detect_stale_candidates(_PUMP_OVERLAY, "2026-06-19")
    musk = next(
        s
        for s in stale
        if s.subject == "Elon Musk" and s.predicate == "estimated_net_worth"
    )
    assert musk.as_of == "2026-06"
    assert musk.age_days == 18
    assert musk.freshness_window == 30
    assert musk.staleness_ratio == 0.6
