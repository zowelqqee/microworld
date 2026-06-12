"""Tests for core.memory_benchmark."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import memory_benchmark as memory


def test_memory_helper_returns_null_when_psutil_unavailable(monkeypatch):
    monkeypatch.setattr(memory, "_PSUTIL_LOADED", True)
    monkeypatch.setattr(memory, "_PSUTIL", None)
    monkeypatch.setattr(memory, "_PSUTIL_IMPORT_ERROR", ImportError("missing"))
    assert memory.get_rss_bytes() is None
    status = memory.memory_status()
    assert status["available"] is False
    assert "psutil unavailable" in status["skipped_reason"]
    snapshot = memory.snapshot_memory("x")
    assert snapshot["rss_bytes"] is None
    assert snapshot["available"] is False


def test_memory_sampler_can_start_and_stop_without_hanging():
    sampler = memory.MemorySampler(interval_ms=1)
    sampler.start()
    peak = sampler.stop()
    if memory.memory_status()["available"]:
        assert isinstance(peak, int)
        assert peak >= 0
    else:
        assert peak is None


def test_memory_tracker_reports_schema_and_optional_ints():
    with memory.MemoryTracker("phase", interval_ms=1) as tracker:
        payload = [0] * 10
        assert len(payload) == 10
    result = tracker.to_dict()
    assert result["label"] == "phase"
    assert "rss_start_bytes" in result
    assert "rss_end_bytes" in result
    assert "peak_rss_bytes" in result
    if result["available"]:
        assert isinstance(result["rss_start_bytes"], int)
        assert isinstance(result["rss_end_bytes"], int)
        assert isinstance(result["peak_rss_bytes"], int)
        assert result["rss_start_bytes"] >= 0
        assert result["rss_end_bytes"] >= 0
        assert result["peak_rss_bytes"] >= 0


def test_phase_memory_metrics_and_mb_summary_are_schema_stable():
    phase = memory.phase_memory_metrics(
        "training",
        {
            "rss_start_bytes": 1024 * 1024,
            "rss_end_bytes": 2 * 1024 * 1024,
            "rss_delta_bytes": 1024 * 1024,
            "peak_rss_bytes": 3 * 1024 * 1024,
        },
    )
    assert phase["rss_before_training_bytes"] == 1024 * 1024
    assert phase["training_peak_rss_bytes"] == 3 * 1024 * 1024
    summary = memory.memory_mb_summary(phase)
    assert summary["rss_before_training_mb"] == 1
    assert summary["training_peak_rss_mb"] == 3
