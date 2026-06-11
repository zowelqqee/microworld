import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pattern_prediction import PatternPrediction
from examples.application_demo import format_evidence_chain, format_prediction_block


def test_application_demo_runs_and_prints_four_sections():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run(
        [sys.executable, "examples/application_demo.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "1. Strong Predictions" in result.stdout
    assert "2. Weak But Useful Predictions" in result.stdout
    assert "3. Rejected / Risky Predictions" in result.stdout
    assert "4. Summary" in result.stdout


def test_format_prediction_block_includes_chain_and_interpretation():
    pred = PatternPrediction(
        source="blood",
        relation_type="made_of",
        target="iron",
        confidence=0.42,
        reason="transitive pattern: made_of -> made_of",
        evidence=["haemoglobin"],
        drift_type="atomic_component",
        drift_penalty=0.65,
    )

    assert format_evidence_chain(pred) == (
        "blood --made_of--> haemoglobin --made_of--> iron"
    )
    block = format_prediction_block(
        pred,
        "useful but indirect",
        status="risky",
        include_drift=True,
    )
    assert "blood --made_of--> iron  [risky]" in block
    assert "confidence    : 0.420" in block
    assert "drift         : atomic_component (penalty=0.65)" in block
    assert "interpretation: useful but indirect" in block
