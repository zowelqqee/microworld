import os
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory_pipeline import prediction_key
from examples.full_pipeline_demo import (
    apply_synthetic_audit,
    build_demo_pipeline,
)


SECTION_HEADERS = [
    "SECTION 1 - Input observations",
    "SECTION 2 - World before sleep",
    "SECTION 3 - Sleep / consolidation",
    "SECTION 4 - Predictions",
    "SECTION 5 - Human audit",
    "SECTION 6 - Trust learning",
    "SECTION 7 - Re-run predictions",
    "SECTION 8 - Why this matters",
]


def test_full_pipeline_demo_runs_and_prints_sections():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run(
        [sys.executable, "examples/full_pipeline_demo.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    for header in SECTION_HEADERS:
        assert header in result.stdout
    assert "Microworld changes future reasoning behavior" in result.stdout


def test_trust_changes_after_audit():
    pipeline = build_demo_pipeline()
    pipeline.sleep()
    predictions = pipeline.predict()
    before = dict(pipeline.relation_trust)

    apply_synthetic_audit(pipeline, predictions)
    update = pipeline.learn_from_audit()

    assert update.learned
    assert update.after["uses"] != before["uses"]
    assert update.after["uses"] < before["uses"]


def test_prediction_confidence_changes_after_learning():
    pipeline = build_demo_pipeline()
    pipeline.sleep()
    before_predictions = pipeline.predict()
    before_by_key = {prediction_key(pred): pred for pred in before_predictions}

    apply_synthetic_audit(pipeline, before_predictions)
    pipeline.learn_from_audit()
    after_predictions = pipeline.predict()
    after_by_key = {prediction_key(pred): pred for pred in after_predictions}

    changed = [
        key
        for key, before in before_by_key.items()
        if key in after_by_key
        and abs(before.confidence - after_by_key[key].confidence) > 1e-9
    ]
    assert changed
