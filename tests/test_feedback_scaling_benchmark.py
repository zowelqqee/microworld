import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.feedback_scaling_benchmark import (
    generate_synthetic_audit_rows,
    run_feedback_scaling_benchmark,
)
from examples.efficiency_benchmark import estimate_tokens


def test_synthetic_generation_deterministic():
    first = generate_synthetic_audit_rows(10)
    second = generate_synthetic_audit_rows(10)

    assert first == second
    assert first[0]["relation_type"] == "made_of"
    assert first[0]["manual_label"] == "plausible"
    assert "drift=none" in first[0]["reason"]


def test_token_estimates_computed():
    results = run_feedback_scaling_benchmark((100,))
    result = results[0]

    assert result.raw_audit_tokens == pytest.approx(estimate_tokens(result.raw_audit_bytes))
    assert result.trust_profile_tokens == pytest.approx(
        estimate_tokens(result.trust_profile_bytes)
    )
    assert result.compression_ratio > 0


def test_compression_ratio_increases_with_larger_row_counts():
    small, large = run_feedback_scaling_benchmark((100, 1_000))

    assert large.raw_audit_tokens > small.raw_audit_tokens
    assert large.compression_ratio > small.compression_ratio


def test_benchmark_runs():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run(
        [sys.executable, "examples/feedback_scaling_benchmark.py", "--scales", "100,500"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Feedback Compression Scaling Benchmark" in result.stdout
    assert "rows |" in result.stdout
    assert "audit_tokens" in result.stdout
    assert "trust_tokens" in result.stdout
    assert "Feedback rows grow linearly" in result.stdout
