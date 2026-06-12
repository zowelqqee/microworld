import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.feedback_scaling_benchmark import ScalingResult
from examples.feedback_scaling_plot import build_svg


def _result(rows: int, audit_tokens: float, trust_tokens: float) -> ScalingResult:
    return ScalingResult(
        rows=rows,
        raw_audit_bytes=int(audit_tokens * 4),
        raw_audit_tokens=audit_tokens,
        trust_profile_bytes=int(trust_tokens * 4),
        trust_profile_tokens=trust_tokens,
        compression_ratio=audit_tokens / trust_tokens,
        learn_time_ms=1.0,
    )


def test_build_svg_contains_two_curves_and_labels():
    svg = build_svg(
        [
            _result(100, 5_000.0, 300.0),
            _result(1_000, 50_000.0, 310.0),
        ]
    )

    assert svg.count("<polyline") == 2
    assert "raw audit context tokens" in svg
    assert "learned trust profile tokens" in svg
    assert "feedback rows" in svg
    assert "estimated tokens" in svg


def test_plot_script_writes_svg(tmp_path):
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    output = tmp_path / "feedback_scaling_curves.svg"
    result = subprocess.run(
        [
            sys.executable,
            "examples/feedback_scaling_plot.py",
            "--scales",
            "100,500",
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Wrote feedback scaling plot" in result.stdout
    assert output.exists()
    assert "<svg" in output.read_text(encoding="utf-8")
