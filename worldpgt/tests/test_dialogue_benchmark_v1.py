"""Pytest gate over the deterministic dialogue-context benchmark.

The benchmark itself (worldpgt/benchmarks/dialogue_benchmark.py) enforces:
false-resolution rate = 0, double-run trace determinism, and
replay(records) == live state, per session. This test makes it part of the
suite and adds a latency budget check on the resolver path.
"""

from __future__ import annotations

from worldpgt.benchmarks.dialogue_benchmark import DEFAULT_FIXTURE_PATH, run_benchmark


def test_dialogue_benchmark_all_sessions_pass():
    report = run_benchmark(DEFAULT_FIXTURE_PATH)
    assert report.passed, "\n" + report.summary()


def test_dialogue_resolver_latency_budget():
    report = run_benchmark(DEFAULT_FIXTURE_PATH)
    assert report.resolver_calls > 0
    mean_us = report.resolver_total_ns / report.resolver_calls / 1000
    # Architecture budget: < 1ms added latency. The benchmark path (grammar
    # scan + scoring, no I/O) must stay an order of magnitude under it.
    assert mean_us < 1000, f"resolver mean {mean_us:.1f}µs exceeds 1ms budget"
