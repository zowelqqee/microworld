"""Tests for audit-driven pump stale recheck signal wiring."""

from __future__ import annotations

from worldpgt.experiments.run_audit_driven_pump_v1 import _stale_fact_signals
from worldpgt.experiments.run_knowledge_pump_v1 import _load_gap_frontier_signals
from worldpgt.knowledge.staleness_detector import StaleCandidate
from worldpgt.knowledge_pump.audit_types import GapReport


def test_stale_fact_signals_keep_highest_ratio_per_subject():
    report = GapReport(
        generated_at="2026-06-19T00:00:00+00:00",
        period_days=30,
        total_audit_events=0,
        acquisition_candidates=[],
        policy_blocked=[],
        stale_candidates=[
            StaleCandidate("Elon Musk", "ranking", "rank 1", "2026-06", 18, 90, 0.2),
            StaleCandidate("Elon Musk", "estimated_net_worth", "US$1.1T", "2026-06", 18, 30, 0.6),
        ],
    )
    signals = _stale_fact_signals(report)
    assert list(signals) == ["elon musk"]
    assert signals["elon musk"]["predicate"] == "estimated_net_worth"
    assert signals["elon musk"]["staleness_ratio"] == 0.6


def test_pump_runner_loads_cleaned_gap_report_signals(tmp_path):
    (tmp_path / "gap_report.json").write_text(
        """
        {
          "acquisition_candidates": [
            {"entity": "Starlink", "gap_type": "missing_facts", "count": 3},
            {"entity": "[unknown entity]", "gap_type": "missing_facts", "count": 9}
          ],
          "stale_candidates": [
            {"subject": "Elon Musk", "predicate": "ranking", "staleness_ratio": 0.2},
            {"subject": "Elon Musk", "predicate": "estimated_net_worth", "staleness_ratio": 0.6}
          ]
        }
        """,
        encoding="utf-8",
    )

    gap_entities, stale_signals = _load_gap_frontier_signals(tmp_path)

    assert gap_entities == {"starlink"}
    assert list(stale_signals) == ["elon musk"]
    assert stale_signals["elon musk"]["predicate"] == "estimated_net_worth"
