"""Lightweight RSS tracking helpers for benchmark scripts.

psutil is optional.  When it is not installed, helpers return null RSS values
with an explicit unavailable reason so benchmark JSON stays schema-stable.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

_PSUTIL: Any = None
_PSUTIL_LOADED = False
_PSUTIL_IMPORT_ERROR: Exception | None = None


def _load_psutil():
    global _PSUTIL, _PSUTIL_LOADED, _PSUTIL_IMPORT_ERROR
    if _PSUTIL_LOADED:
        return _PSUTIL
    _PSUTIL_LOADED = True
    try:
        import psutil  # type: ignore

        _PSUTIL = psutil
    except Exception as exc:  # pragma: no cover - depends on local environment
        _PSUTIL = None
        _PSUTIL_IMPORT_ERROR = exc
    return _PSUTIL


def memory_status(enabled: bool = True) -> dict:
    if not enabled:
        return {"available": False, "skipped_reason": "memory tracking disabled"}
    psutil = _load_psutil()
    if psutil is None:
        reason = "psutil unavailable"
        if _PSUTIL_IMPORT_ERROR is not None:
            reason = f"psutil unavailable: {_PSUTIL_IMPORT_ERROR}"
        return {"available": False, "skipped_reason": reason}
    return {"available": True, "skipped_reason": None}


def get_rss_bytes() -> int | None:
    psutil = _load_psutil()
    if psutil is None:
        return None
    try:
        return int(psutil.Process(os.getpid()).memory_info().rss)
    except Exception:  # pragma: no cover - defensive for platform/process oddities
        return None


def bytes_to_mb(value: int | None) -> float | None:
    if value is None:
        return None
    return value / (1024 * 1024)


def snapshot_memory(label: str, *, enabled: bool = True) -> dict:
    status = memory_status(enabled)
    rss = get_rss_bytes() if status["available"] else None
    return {
        "label": label,
        "available": status["available"],
        "memory_metrics_available": status["available"],
        "skipped_reason": status["skipped_reason"],
        "rss_bytes": rss,
        "rss_mb": bytes_to_mb(rss),
    }


class MemorySampler:
    """Sample RSS in a background thread and retain the highest observed value."""

    def __init__(self, interval_ms: int = 10, *, enabled: bool = True) -> None:
        self.interval_sec = max(interval_ms, 1) / 1000
        self.enabled = enabled
        self.available = memory_status(enabled)["available"]
        self.peak_rss_bytes: int | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample_once(self) -> None:
        rss = get_rss_bytes()
        if rss is not None:
            if self.peak_rss_bytes is None or rss > self.peak_rss_bytes:
                self.peak_rss_bytes = rss

    def start(self) -> "MemorySampler":
        if not self.available:
            return self
        self._sample_once()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval_sec):
            self._sample_once()

    def stop(self) -> int | None:
        if not self.available:
            return self.peak_rss_bytes
        self._sample_once()
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(self.interval_sec * 5, 0.1))
        return self.peak_rss_bytes


class MemoryTracker:
    """Context manager recording RSS start/end/delta and sampled peak RSS."""

    def __init__(self, label: str, *, enabled: bool = True, interval_ms: int = 10) -> None:
        self.label = label
        self.enabled = enabled
        self.interval_ms = interval_ms
        self.status = memory_status(enabled)
        self.sampler = MemorySampler(interval_ms, enabled=enabled)
        self.rss_start_bytes: int | None = None
        self.rss_end_bytes: int | None = None
        self.peak_rss_bytes: int | None = None
        self.duration_sec: float | None = None
        self._start_time: float | None = None

    def __enter__(self) -> "MemoryTracker":
        if self.status["available"]:
            self.rss_start_bytes = get_rss_bytes()
            self.sampler.start()
        self._start_time = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._start_time is not None:
            self.duration_sec = time.perf_counter() - self._start_time
        if self.status["available"]:
            self.peak_rss_bytes = self.sampler.stop()
            self.rss_end_bytes = get_rss_bytes()

    def to_dict(self) -> dict:
        delta = None
        if self.rss_start_bytes is not None and self.rss_end_bytes is not None:
            delta = self.rss_end_bytes - self.rss_start_bytes
        return {
            "label": self.label,
            "available": self.status["available"],
            "skipped_reason": self.status["skipped_reason"],
            "rss_start_bytes": self.rss_start_bytes,
            "rss_end_bytes": self.rss_end_bytes,
            "rss_delta_bytes": delta,
            "peak_rss_bytes": self.peak_rss_bytes,
            "duration_sec": self.duration_sec,
        }


def phase_memory_metrics(phase: str, tracker_result: dict | None) -> dict:
    if tracker_result is None:
        start = end = delta = peak = None
    else:
        start = tracker_result.get("rss_start_bytes")
        end = tracker_result.get("rss_end_bytes")
        delta = tracker_result.get("rss_delta_bytes")
        peak = tracker_result.get("peak_rss_bytes")
    return {
        f"rss_before_{phase}_bytes": start,
        f"rss_after_{phase}_bytes": end,
        f"{phase}_rss_delta_bytes": delta,
        f"{phase}_peak_rss_bytes": peak,
    }


def memory_mb_summary(memory: dict) -> dict:
    return {
        key[:-6] + "_mb": bytes_to_mb(value)
        for key, value in memory.items()
        if key.endswith("_bytes") and isinstance(value, int)
    }


def empty_memory_metrics(phases: list[str], *, enabled: bool = True) -> dict:
    status = memory_status(enabled)
    memory = {
        "available": status["available"],
        "memory_metrics_available": status["available"],
        "skipped_reason": status["skipped_reason"],
    }
    for phase in phases:
        memory.update(phase_memory_metrics(phase, None))
    memory["memory_mb"] = {}
    return memory
