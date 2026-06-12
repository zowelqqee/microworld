import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.efficiency_benchmark import (
    compute_compression_ratio,
    estimate_tokens,
    run_efficiency_benchmark,
)


def test_token_estimate_deterministic():
    assert estimate_tokens(400) == pytest.approx(100.0)
    assert estimate_tokens(401) == pytest.approx(100.25)
    assert estimate_tokens(401) == estimate_tokens(401)


def test_compression_ratio_computed():
    assert compute_compression_ratio(1000.0, 250.0) == pytest.approx(4.0)
    assert compute_compression_ratio(1000.0, 0.0) == pytest.approx(0.0)


def test_benchmark_runs():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run(
        [sys.executable, "examples/efficiency_benchmark.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Microworld Efficiency Benchmark" in result.stdout
    assert "audit_context_tokens" in result.stdout
    assert "trust_profile_tokens" in result.stdout
    assert "compression_ratio" in result.stdout
    assert "Microworld stores feedback as compact trust state" in result.stdout


def test_learned_accepted_lte_baseline_at_threshold():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    result = run_efficiency_benchmark(
        data_path=os.path.join(root, "data", "conceptnet_sample.csv"),
        threshold=0.4,
        repeated_queries=100,
    )

    assert result.learned_accepted <= result.baseline_accepted
    assert result.suppressed == result.baseline_accepted - result.learned_accepted
