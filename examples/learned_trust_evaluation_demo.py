"""
Evaluate how audit-learned relation trust changes prediction behavior.
"""
import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import build_world_from_relations, load_relations_csv
from core.pattern_prediction import PatternBasedPredictor, PatternPrediction
from core.relation_trust import DEFAULT_RELATION_TRUST
from core.trust_learning import TrustProfile

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_PATH = os.path.join(_ROOT, "data", "conceptnet_sample.csv")
TRUST_PATH = os.path.join(_ROOT, "data", "trust_profile.json")


@dataclass
class PredictionComparison:
    baseline_count: int
    learned_count: int
    baseline_above_threshold_count: int
    learned_above_threshold_count: int
    baseline_avg_by_relation: dict[str, float]
    learned_avg_by_relation: dict[str, float]
    suppressed: list[tuple[PatternPrediction, float]]
    promoted: list[tuple[PatternPrediction, float]]
    suppressed_below_threshold: list[tuple[PatternPrediction, float]]
    newly_promoted_above_threshold: list[tuple[PatternPrediction, float]]


def prediction_key(prediction: PatternPrediction) -> tuple[str, str, str]:
    return (prediction.source, prediction.relation_type, prediction.target)


def average_confidence_by_relation(
    predictions: list[PatternPrediction],
) -> dict[str, float]:
    totals: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for prediction in predictions:
        totals[prediction.relation_type] += prediction.confidence
        counts[prediction.relation_type] += 1
    return {
        relation: totals[relation] / counts[relation]
        for relation in sorted(counts)
    }


def merge_learned_relation_trust(
    baseline_trust: dict[str, float],
    learned_trust: dict[str, float],
) -> dict[str, float]:
    """Apply learned trust as overrides while preserving audited defaults."""
    merged = dict(baseline_trust)
    merged.update(learned_trust)
    return merged


def predictions_above_threshold(
    predictions: list[PatternPrediction],
    min_confidence: float,
) -> list[PatternPrediction]:
    return [pred for pred in predictions if pred.confidence >= min_confidence]


def compare_predictions(
    baseline: list[PatternPrediction],
    learned: list[PatternPrediction],
    min_confidence: float = 0.5,
    epsilon: float = 1e-9,
) -> PredictionComparison:
    baseline_map = {prediction_key(pred): pred for pred in baseline}
    learned_map = {prediction_key(pred): pred for pred in learned}
    suppressed: list[tuple[PatternPrediction, float]] = []
    promoted: list[tuple[PatternPrediction, float]] = []
    suppressed_below_threshold: list[tuple[PatternPrediction, float]] = []
    newly_promoted_above_threshold: list[tuple[PatternPrediction, float]] = []

    for key, base_pred in baseline_map.items():
        learned_pred = learned_map.get(key)
        learned_conf = learned_pred.confidence if learned_pred is not None else 0.0
        delta = learned_conf - base_pred.confidence
        if delta < -epsilon:
            suppressed.append((base_pred, delta))
        if base_pred.confidence >= min_confidence and learned_conf < min_confidence:
            suppressed_below_threshold.append((base_pred, delta))

    for key, learned_pred in learned_map.items():
        base_pred = baseline_map.get(key)
        base_conf = base_pred.confidence if base_pred is not None else 0.0
        delta = learned_pred.confidence - base_conf
        if delta > epsilon:
            promoted.append((learned_pred, delta))
        if base_conf < min_confidence and learned_pred.confidence >= min_confidence:
            newly_promoted_above_threshold.append((learned_pred, delta))

    suppressed.sort(key=lambda item: (item[1], item[0].relation_type, item[0].source))
    promoted.sort(key=lambda item: (-item[1], item[0].relation_type, item[0].source))
    suppressed_below_threshold.sort(
        key=lambda item: (item[1], item[0].relation_type, item[0].source)
    )
    newly_promoted_above_threshold.sort(
        key=lambda item: (-item[1], item[0].relation_type, item[0].source)
    )
    return PredictionComparison(
        baseline_count=len(baseline),
        learned_count=len(learned),
        baseline_above_threshold_count=len(
            predictions_above_threshold(baseline, min_confidence)
        ),
        learned_above_threshold_count=len(
            predictions_above_threshold(learned, min_confidence)
        ),
        baseline_avg_by_relation=average_confidence_by_relation(baseline),
        learned_avg_by_relation=average_confidence_by_relation(learned),
        suppressed=suppressed,
        promoted=promoted,
        suppressed_below_threshold=suppressed_below_threshold,
        newly_promoted_above_threshold=newly_promoted_above_threshold,
    )


def run_predictions(
    relations,
    relation_trust: dict[str, float],
) -> list[PatternPrediction]:
    return PatternBasedPredictor(relations).predict_from_bigrams(
        min_count=5,
        min_confidence=0.0,
        hub_penalty=True,
        relation_trust=relation_trust,
        use_relation_drift=True,
    )


def _print_trust_delta_table(
    baseline_trust: dict[str, float],
    learned_trust: dict[str, float],
) -> None:
    print("\nRelation Trust Deltas")
    print("---------------------")
    print(f"{'relation':16s} {'baseline_trust':>14} {'learned_trust':>13} {'delta':>8}")
    relations = sorted(set(baseline_trust) | set(learned_trust))
    for relation in relations:
        base = baseline_trust.get(relation, 0.5)
        learned = learned_trust.get(relation, 0.5)
        print(f"{relation:16s} {base:>14.3f} {learned:>13.3f} {learned - base:>8.3f}")


def _print_avg_table(comparison: PredictionComparison) -> None:
    print("\nAverage Confidence By Relation")
    print("------------------------------")
    print(f"{'relation':16s} {'baseline':>10} {'learned':>10} {'delta':>8}")
    relations = sorted(
        set(comparison.baseline_avg_by_relation) | set(comparison.learned_avg_by_relation)
    )
    for relation in relations:
        base = comparison.baseline_avg_by_relation.get(relation, 0.0)
        learned = comparison.learned_avg_by_relation.get(relation, 0.0)
        print(f"{relation:16s} {base:>10.3f} {learned:>10.3f} {learned - base:>8.3f}")


def _print_predictions(title: str, predictions: list[PatternPrediction], limit: int = 20) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for pred in predictions[:limit]:
        via = ", ".join(pred.evidence[:3])
        print(
            f"  {pred.source} --{pred.relation_type}--> {pred.target} "
            f"conf={pred.confidence:.3f} via={via}"
        )


def _print_changed(title: str, changes: list[tuple[PatternPrediction, float]], limit: int = 20) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not changes:
        print("  (none)")
        return
    for pred, delta in changes[:limit]:
        via = ", ".join(pred.evidence[:3])
        print(
            f"  {pred.source} --{pred.relation_type}--> {pred.target} "
            f"baseline_conf={pred.confidence:.3f} delta={delta:+.3f} via={via}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate audit-learned relation trust against defaults."
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for reported prediction counts (default: 0.5).",
    )
    args = parser.parse_args()

    if not os.path.exists(DATA_PATH):
        print(f"Input not found: {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    if not os.path.exists(TRUST_PATH):
        print(f"Trust profile not found: {TRUST_PATH}", file=sys.stderr)
        print("Run: python3 examples/trust_learning_demo.py", file=sys.stderr)
        sys.exit(1)

    rows = load_relations_csv(DATA_PATH)
    world = build_world_from_relations(rows)
    profile = TrustProfile.from_json(TRUST_PATH)
    learned_relation_trust = merge_learned_relation_trust(
        DEFAULT_RELATION_TRUST,
        profile.relation_trust,
    )

    baseline = run_predictions(world.get_relations(), DEFAULT_RELATION_TRUST)
    learned = run_predictions(world.get_relations(), learned_relation_trust)
    comparison = compare_predictions(
        baseline,
        learned,
        min_confidence=args.min_confidence,
    )
    baseline_above = predictions_above_threshold(baseline, args.min_confidence)
    learned_above = predictions_above_threshold(learned, args.min_confidence)

    print("Learned Trust Evaluation")
    print("========================")
    print(f"relations loaded                      : {len(world.get_relations())}")
    print(f"min confidence threshold              : {args.min_confidence:.3f}")
    print(f"baseline predictions total            : {comparison.baseline_count}")
    print(f"learned predictions total             : {comparison.learned_count}")
    print(
        "baseline predictions above threshold : "
        f"{comparison.baseline_above_threshold_count}"
    )
    print(
        "learned predictions above threshold  : "
        f"{comparison.learned_above_threshold_count}"
    )
    print(
        "suppressed below threshold           : "
        f"{len(comparison.suppressed_below_threshold)}"
    )
    print(
        "newly promoted above threshold       : "
        f"{len(comparison.newly_promoted_above_threshold)}"
    )

    _print_trust_delta_table(DEFAULT_RELATION_TRUST, learned_relation_trust)
    _print_avg_table(comparison)
    _print_predictions("Top 20 Predictions - Baseline", baseline_above)
    _print_predictions("Top 20 Predictions - Learned", learned_above)
    _print_changed("Top Suppressed Predictions", comparison.suppressed)
    _print_changed("Top Promoted Predictions", comparison.promoted)
    _print_changed(
        "Top Suppressed Below Threshold",
        comparison.suppressed_below_threshold,
    )
    _print_changed(
        "Top Newly Promoted Above Threshold",
        comparison.newly_promoted_above_threshold,
    )


if __name__ == "__main__":
    main()
