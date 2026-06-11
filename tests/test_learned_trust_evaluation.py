import os
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pattern_prediction import PatternBasedPredictor, PatternPrediction
from core.relation_trust import DEFAULT_RELATION_TRUST
from core.relations import Relation
from examples.learned_trust_evaluation_demo import (
    compare_predictions,
    merge_learned_relation_trust,
)


def _pred(source: str, relation: str, target: str, confidence: float) -> PatternPrediction:
    return PatternPrediction(
        source=source,
        relation_type=relation,
        target=target,
        confidence=confidence,
        reason="test",
        evidence=["mid"],
    )


def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def test_demo_runs():
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
    trust_path = os.path.join(root, "data", "trust_profile.json")
    if not os.path.exists(trust_path):
        subprocess.run(
            [sys.executable, "examples/trust_learning_demo.py"],
            cwd=root,
            check=True,
            text=True,
            capture_output=True,
        )

    result = subprocess.run(
        [sys.executable, "examples/learned_trust_evaluation_demo.py"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Learned Trust Evaluation" in result.stdout
    assert "Relation Trust Deltas" in result.stdout
    assert "baseline predictions above threshold" in result.stdout
    assert "learned predictions above threshold" in result.stdout
    assert "Top Suppressed Below Threshold" in result.stdout
    assert "Top Newly Promoted Above Threshold" in result.stdout


def test_comparison_detects_suppressed_and_promoted():
    baseline = [
        _pred("a", "made_of", "c", 0.8),
        _pred("x", "part_of", "z", 0.4),
    ]
    learned = [
        _pred("a", "made_of", "c", 0.5),
        _pred("x", "part_of", "z", 0.6),
    ]

    comparison = compare_predictions(baseline, learned)

    assert len(comparison.suppressed) == 1
    assert comparison.suppressed[0][0].source == "a"
    assert comparison.suppressed[0][1] == pytest.approx(-0.3)
    assert len(comparison.promoted) == 1
    assert comparison.promoted[0][0].source == "x"
    assert comparison.promoted[0][1] == pytest.approx(0.2)


def test_learned_trust_changes_confidence_when_trust_differs():
    rels = _rels(
        ("a", "made_of", "b"),
        ("b", "made_of", "c"),
    )

    high = PatternBasedPredictor(rels).predict_from_bigrams(
        min_count=1,
        min_confidence=0.0,
        hub_penalty=False,
        relation_trust={"made_of": 1.0},
    )
    low = PatternBasedPredictor(rels).predict_from_bigrams(
        min_count=1,
        min_confidence=0.0,
        hub_penalty=False,
        relation_trust={"made_of": 0.25},
    )

    assert high
    assert low
    assert low[0].confidence < high[0].confidence


def test_identical_trust_gives_no_deltas():
    baseline = [
        _pred("a", "made_of", "c", 0.8),
        _pred("x", "part_of", "z", 0.4),
    ]
    learned = [
        _pred("a", "made_of", "c", 0.8),
        _pred("x", "part_of", "z", 0.4),
    ]

    comparison = compare_predictions(baseline, learned)

    assert comparison.suppressed == []
    assert comparison.promoted == []


def test_disabled_relation_not_promoted_by_missing_learned_trust():
    merged = merge_learned_relation_trust(
        DEFAULT_RELATION_TRUST,
        {"made_of": 0.62},
    )

    assert merged["at_location"] == pytest.approx(
        DEFAULT_RELATION_TRUST["at_location"]
    )


def test_missing_known_relation_keeps_baseline_trust():
    merged = merge_learned_relation_trust(
        DEFAULT_RELATION_TRUST,
        {"made_of": 0.62},
    )

    assert merged["part_of"] == pytest.approx(DEFAULT_RELATION_TRUST["part_of"])


def test_threshold_changes_prediction_count():
    baseline = [
        _pred("a", "made_of", "c", 0.6),
        _pred("x", "part_of", "z", 0.4),
    ]
    learned = [
        _pred("a", "made_of", "c", 0.4),
        _pred("x", "part_of", "z", 0.4),
    ]

    comparison = compare_predictions(baseline, learned, min_confidence=0.5)

    assert comparison.baseline_above_threshold_count == 1
    assert comparison.learned_above_threshold_count == 0


def test_suppressed_below_threshold_is_reported():
    baseline = [_pred("a", "made_of", "c", 0.6)]
    learned = [_pred("a", "made_of", "c", 0.4)]

    comparison = compare_predictions(baseline, learned, min_confidence=0.5)

    assert len(comparison.suppressed_below_threshold) == 1
    pred, delta = comparison.suppressed_below_threshold[0]
    assert pred.source == "a"
    assert delta == pytest.approx(-0.2)
