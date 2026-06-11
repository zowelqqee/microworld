import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pattern_prediction import PatternPrediction
from examples.trust_transfer_experiment import (
    compute_threshold_metrics,
    split_relations,
)


def _row(index: int) -> dict[str, str]:
    return {
        "source": f"source_{index}",
        "relation_type": "made_of",
        "target": f"target_{index}",
    }


def _pred(relation: str, confidence: float) -> PatternPrediction:
    return PatternPrediction(
        source=f"{relation}_source",
        relation_type=relation,
        target=f"{relation}_target",
        confidence=confidence,
        reason="test",
        evidence=["middle"],
    )


def _triple(row: dict[str, str]) -> tuple[str, str, str]:
    return row["source"], row["relation_type"], row["target"]


def test_split_reproducibility():
    rows = [_row(i) for i in range(10)]

    train_a, test_a = split_relations(rows, train_fraction=0.7)
    train_b, test_b = split_relations(rows, train_fraction=0.7)

    assert train_a == train_b
    assert test_a == test_b
    assert len(train_a) == 7
    assert len(test_a) == 3
    assert train_a[0]["source"] == "source_0"
    assert test_a[0]["source"] == "source_7"


def test_train_test_non_overlap():
    rows = [_row(i) for i in range(12)]

    train, test = split_relations(rows, train_fraction=0.7)
    train_triples = {_triple(row) for row in train}
    test_triples = {_triple(row) for row in test}

    assert train_triples.isdisjoint(test_triples)


def test_experiment_runs():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    result = subprocess.run(
        [sys.executable, "examples/trust_transfer_experiment.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Trust Transfer Experiment" in result.stdout
    assert "TRAIN" in result.stdout
    assert "TEST" in result.stdout
    assert "Per Relation - TRAIN" in result.stdout
    assert "Per Relation - TEST" in result.stdout


def test_threshold_metrics_computed():
    predictions = [
        _pred("made_of", 0.5),
        _pred("made_of", 0.3),
        _pred("part_of", 0.4),
        _pred("is_a", 0.39),
    ]

    metrics = compute_threshold_metrics(predictions, threshold=0.4)

    assert metrics.total_predictions == 4
    assert metrics.accepted_predictions == 2
    assert metrics.average_confidence == pytest.approx((0.5 + 0.3 + 0.4 + 0.39) / 4)
    assert metrics.per_relation["made_of"].total_predictions == 2
    assert metrics.per_relation["made_of"].accepted_predictions == 1
    assert metrics.per_relation["part_of"].accepted_predictions == 1
    assert metrics.per_relation["is_a"].accepted_predictions == 0
