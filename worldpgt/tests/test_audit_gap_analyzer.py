"""Tests for audit_gap_analyzer — gap type classification, aggregation, filtering."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from worldpgt.knowledge_pump.audit_gap_analyzer import _gap_type, analyze_gaps


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_log(path: Path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(e) for e in entries) + "\n",
        encoding="utf-8",
    )


def _entry(
    entity="SpaceX",
    support_kind="missing_knowledge",
    reason="no fact found",
    source="assistant_surface",
    days_ago: int = 0,
    temporal_mismatch: bool = False,
) -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return {
        "timestamp": ts,
        "question": f"What is {entity}?",
        "entity": entity,
        "support_kind": support_kind,
        "reason": reason,
        "source": source,
        "temporal_mismatch": temporal_mismatch,
    }


# ---------------------------------------------------------------------------
# Gap type classification
# ---------------------------------------------------------------------------


class TestGapTypeClassification:
    def test_missing_knowledge_is_missing_facts(self):
        assert _gap_type({"support_kind": "missing_knowledge"}) == "missing_facts"

    def test_unsupported_is_unknown_entity(self):
        assert _gap_type({"support_kind": "unsupported"}) == "unknown_entity"

    def test_audit_blocked_context_is_policy(self):
        assert _gap_type({"support_kind": "audit_blocked_context"}) == "policy_blocked"

    def test_multihop_volatile_hop_is_policy(self):
        assert _gap_type({"support_kind": "multihop_audit", "reason": "volatile_hop in chain"}) == "policy_blocked"

    def test_multihop_high_risk_hop_is_policy(self):
        assert _gap_type({"support_kind": "multihop_audit", "reason": "high_risk_hop"}) == "policy_blocked"

    def test_multihop_transitive_founder_leak_is_policy(self):
        assert _gap_type({"support_kind": "multihop_audit", "reason": "transitive_founder_leak"}) == "policy_blocked"

    def test_multihop_transitive_owner_leak_is_policy(self):
        assert _gap_type({"support_kind": "multihop_audit", "reason": "transitive_owner_leak"}) == "policy_blocked"

    def test_multihop_missing_hop_is_missing_facts(self):
        assert _gap_type({"support_kind": "multihop_audit", "reason": "missing hop in relation chain"}) == "missing_facts"

    def test_multihop_unknown_pattern_is_unknown_entity(self):
        assert _gap_type({"support_kind": "multihop_audit", "reason": "no recognized multi-hop question pattern matched"}) == "unknown_entity"

    def test_fallback_reason_blocked_keyword(self):
        assert _gap_type({"support_kind": "other", "reason": "blocked by policy"}) == "policy_blocked"

    def test_fallback_reason_missing_keyword(self):
        assert _gap_type({"support_kind": "other", "reason": "no fact available, missing data"}) == "missing_facts"

    def test_fallback_unknown(self):
        assert _gap_type({"support_kind": "other", "reason": "something else entirely"}) == "unknown_entity"

    def test_current_live_data_in_reason_is_policy(self):
        assert _gap_type({"support_kind": "missing_knowledge", "reason": "this asks for current/live data"}) == "policy_blocked"

    def test_missing_relation_is_missing_facts(self):
        assert _gap_type({"support_kind": "missing_knowledge", "reason": "no stable relation for this subject is present in the overlay"}) == "missing_facts"

    def test_no_stable_definition_is_unknown_entity(self):
        assert _gap_type({"support_kind": "missing_knowledge", "reason": "no stable definition for this entity is present in the overlay"}) == "unknown_entity"

    def test_private_sensitive_reason_is_policy(self):
        assert _gap_type({"support_kind": "missing_knowledge", "reason": "private/sensitive personal information"}) == "policy_blocked"

    def test_temporal_mismatch_is_stale_snapshot(self):
        assert _gap_type({"support_kind": "missing_knowledge", "temporal_mismatch": True}) == "stale_snapshot"

    def test_snapshot_requires_as_of_reason_is_stale_snapshot(self):
        assert _gap_type({"support_kind": "multihop_audit", "reason": "snapshot_missing_as_of:owned_by"}) == "stale_snapshot"


# ---------------------------------------------------------------------------
# analyze_gaps
# ---------------------------------------------------------------------------


class TestAnalyzeGaps:
    def test_empty_file(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log.write_text("")
        report = analyze_gaps(log, period_days=30)
        assert report.total_audit_events == 0
        assert report.acquisition_candidates == []
        assert report.policy_blocked == []

    def test_missing_file(self, tmp_path):
        report = analyze_gaps(tmp_path / "nonexistent.jsonl", period_days=30)
        assert report.total_audit_events == 0

    def test_acquisition_candidates_do_not_include_policy_blocked(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [
            _entry("SpaceX", "missing_knowledge"),
            _entry("Elon Musk", "audit_blocked_context"),
        ])
        report = analyze_gaps(log, period_days=30)
        candidate_entities = {e.entity for e in report.acquisition_candidates}
        blocked_entities = {e.entity for e in report.policy_blocked}
        assert "SpaceX" in candidate_entities
        assert "Elon Musk" not in candidate_entities
        assert "Elon Musk" in blocked_entities
        assert "SpaceX" not in blocked_entities

    def test_stale_candidates_are_separate(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        overlay = tmp_path / "overlay.json"
        overlay.write_text(json.dumps([
            {
                "overlay_type": "overlay_source_fact",
                "subject": "Acme",
                "predicate": "estimated_net_worth",
                "object": "US$1B",
                "as_of": "2026-05-01",
                "temporal_class": "snapshot",
                "stability": "volatile",
            }
        ]), encoding="utf-8")
        _write_log(log, [
            _entry("SpaceX", "missing_knowledge"),
            _entry("Acme", "multihop_audit", reason="snapshot_missing_as_of:owned_by", temporal_mismatch=True),
        ])
        report = analyze_gaps(log, period_days=30, overlay_path=overlay, current_date="2026-06-19")
        assert {e.entity for e in report.acquisition_candidates} == {"SpaceX"}
        assert [e.subject for e in report.stale_candidates] == ["Acme"]
        assert report.stale_candidates[0].predicate == "estimated_net_worth"
        assert report.stale_candidates[0].staleness_ratio > 1

    def test_count_aggregation_same_entity(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [_entry("SpaceX", "missing_knowledge") for _ in range(5)])
        report = analyze_gaps(log, period_days=30)
        assert len(report.acquisition_candidates) == 1
        assert report.acquisition_candidates[0].entity == "SpaceX"
        assert report.acquisition_candidates[0].count == 5

    def test_sorted_by_count_descending(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        entries = (
            [_entry("Rare", "missing_knowledge")]
            + [_entry("SpaceX", "missing_knowledge") for _ in range(3)]
        )
        _write_log(log, entries)
        report = analyze_gaps(log, period_days=30)
        assert report.acquisition_candidates[0].entity == "SpaceX"
        assert report.acquisition_candidates[0].count == 3

    def test_period_filtering_excludes_old_entries(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [
            _entry("RecentEntity", days_ago=1),
            _entry("OldEntity", days_ago=60),
        ])
        report = analyze_gaps(log, period_days=30)
        entities = {e.entity for e in report.acquisition_candidates}
        assert "RecentEntity" in entities
        assert "OldEntity" not in entities

    def test_period_zero_includes_all(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [
            _entry("RecentEntity", days_ago=1),
            _entry("OldEntity", days_ago=200),
        ])
        report = analyze_gaps(log, period_days=0)
        entities = {e.entity for e in report.acquisition_candidates}
        assert "RecentEntity" in entities
        assert "OldEntity" in entities

    def test_top_reasons_captured(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [_entry("SpaceX", reason="rocket fact missing")])
        report = analyze_gaps(log, period_days=30)
        assert "rocket fact missing" in report.acquisition_candidates[0].top_reasons

    def test_unknown_entity_placeholder_used_when_no_entity(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        entry = _entry("SpaceX")
        entry["entity"] = ""  # no entity extracted
        _write_log(log, [entry])
        report = analyze_gaps(log, period_days=30)
        entities = {e.entity for e in report.acquisition_candidates + report.policy_blocked}
        assert "[unknown entity]" in entities

    def test_to_dict_is_json_serializable(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [_entry("SpaceX")])
        report = analyze_gaps(log, period_days=30)
        json.dumps(report.to_dict())  # must not raise
        assert "stale_candidates" in report.to_dict()

    def test_gap_type_in_report_entry(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [_entry("SpaceX", "missing_knowledge")])
        report = analyze_gaps(log, period_days=30)
        assert report.acquisition_candidates[0].gap_type == "missing_facts"

    def test_unknown_entity_kind_in_report(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [_entry("Starbase", "unsupported")])
        report = analyze_gaps(log, period_days=30)
        assert report.acquisition_candidates[0].gap_type == "unknown_entity"

    def test_eligibility_filter_keeps_source_backed_entity(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [
            _entry("Starlink", "missing_knowledge"),
            _entry("Zaphod Beeblebrox", "missing_knowledge", reason="no stable definition"),
        ])
        report = analyze_gaps(log, period_days=30, require_acquisition_eligibility=True)
        entities = {entry.entity for entry in report.acquisition_candidates}

        assert "Starlink" in entities
        assert "Zaphod Beeblebrox" not in entities

    def test_eligibility_filter_excludes_synthetic_source_even_if_known(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        entry = _entry("SpaceX", "missing_knowledge", source="synthetic_fixture")
        _write_log(log, [entry])
        report = analyze_gaps(log, period_days=30, require_acquisition_eligibility=True)

        assert report.acquisition_candidates == []

    def test_synthetic_edge_cases_remain_available_without_production_filter(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        _write_log(log, [
            _entry("Zaphod Beeblebrox", "missing_knowledge", reason="no stable definition"),
            _entry("the Galactic Empire", "missing_knowledge", reason="no stable relation"),
        ])
        report = analyze_gaps(log, period_days=30)

        assert {entry.entity for entry in report.acquisition_candidates} == {
            "Zaphod Beeblebrox",
            "the Galactic Empire",
        }
