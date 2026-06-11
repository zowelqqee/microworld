"""
Evaluate whether audit-learned trust transfers to unseen relation data.

The experiment keeps the graph split simple and reproducible:
first 70% of the ConceptNet sample is train, last 30% is test.  Trust is not
learned from either split; it is learned only from existing human audit CSVs.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import build_world_from_relations, load_relations_csv
from core.pattern_prediction import PatternBasedPredictor, PatternPrediction
from core.relation_trust import DEFAULT_RELATION_TRUST
from core.trust_learning import TrustProfile, learn_trust_from_audits

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_PATH = os.path.join(_ROOT, "data", "conceptnet_sample.csv")
AUDIT_PATHS = [
    os.path.join(_ROOT, "data", "audit_all.csv"),
    os.path.join(_ROOT, "data", "audit_mixed_out.csv"),
    os.path.join(_ROOT, "data", "audit_drift_aware.csv"),
]
RELATIONS_OF_INTEREST = ("made_of", "part_of", "is_a")


@dataclass
class RelationMetrics:
    total_predictions: int
    accepted_predictions: int
    average_confidence: float


@dataclass
class ThresholdMetrics:
    total_predictions: int
    accepted_predictions: int
    average_confidence: float
    per_relation: dict[str, RelationMetrics]


@dataclass
class SplitComparison:
    baseline: ThresholdMetrics
    learned: ThresholdMetrics

    @property
    def accepted_delta(self) -> int:
        return self.learned.accepted_predictions - self.baseline.accepted_predictions

    @property
    def average_confidence_delta(self) -> float:
        return self.learned.average_confidence - self.baseline.average_confidence


def split_relations(
    rows: list[dict[str, str]],
    train_fraction: float = 0.7,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Return deterministic first-N train and remaining test rows."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = int(len(rows) * train_fraction)
    return rows[:split_at], rows[split_at:]


def merge_learned_relation_trust(
    baseline_trust: dict[str, float],
    learned_trust: dict[str, float],
) -> dict[str, float]:
    """Use learned audit values as explicit overrides, preserving defaults."""
    merged = dict(baseline_trust)
    merged.update(learned_trust)
    return merged


def compute_threshold_metrics(
    predictions: list[PatternPrediction],
    threshold: float,
    relations_of_interest: tuple[str, ...] = RELATIONS_OF_INTEREST,
) -> ThresholdMetrics:
    accepted = [pred for pred in predictions if pred.confidence >= threshold]
    per_relation_predictions: dict[str, list[PatternPrediction]] = defaultdict(list)
    for pred in predictions:
        per_relation_predictions[pred.relation_type].append(pred)

    relation_names = sorted(set(relations_of_interest) | set(per_relation_predictions))
    per_relation = {
        relation: _relation_metrics(per_relation_predictions.get(relation, []), threshold)
        for relation in relation_names
    }
    return ThresholdMetrics(
        total_predictions=len(predictions),
        accepted_predictions=len(accepted),
        average_confidence=_average_confidence(predictions),
        per_relation=per_relation,
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


def compare_trust_on_rows(
    rows: list[dict[str, str]],
    baseline_trust: dict[str, float],
    learned_trust: dict[str, float],
    threshold: float,
) -> SplitComparison:
    world = build_world_from_relations(rows)
    relations = world.get_relations()
    baseline_predictions = run_predictions(relations, baseline_trust)
    learned_predictions = run_predictions(relations, learned_trust)
    return SplitComparison(
        baseline=compute_threshold_metrics(baseline_predictions, threshold),
        learned=compute_threshold_metrics(learned_predictions, threshold),
    )


def load_learned_profile(audit_paths: list[str]) -> tuple[TrustProfile, list[str]]:
    existing = [path for path in audit_paths if os.path.exists(path)]
    return learn_trust_from_audits(existing), existing


def run_transfer_experiment(
    data_path: str = DATA_PATH,
    audit_paths: list[str] | None = None,
    threshold: float = 0.4,
    train_fraction: float = 0.7,
) -> tuple[SplitComparison, SplitComparison, TrustProfile, list[str]]:
    rows = load_relations_csv(data_path)
    train_rows, test_rows = split_relations(rows, train_fraction=train_fraction)
    profile, existing_audits = load_learned_profile(audit_paths or AUDIT_PATHS)
    learned_trust = merge_learned_relation_trust(
        DEFAULT_RELATION_TRUST,
        profile.relation_trust,
    )
    train = compare_trust_on_rows(
        train_rows,
        DEFAULT_RELATION_TRUST,
        learned_trust,
        threshold,
    )
    test = compare_trust_on_rows(
        test_rows,
        DEFAULT_RELATION_TRUST,
        learned_trust,
        threshold,
    )
    return train, test, profile, existing_audits


def _relation_metrics(
    predictions: list[PatternPrediction],
    threshold: float,
) -> RelationMetrics:
    return RelationMetrics(
        total_predictions=len(predictions),
        accepted_predictions=sum(1 for pred in predictions if pred.confidence >= threshold),
        average_confidence=_average_confidence(predictions),
    )


def _average_confidence(predictions: list[PatternPrediction]) -> float:
    if not predictions:
        return 0.0
    return sum(pred.confidence for pred in predictions) / len(predictions)


def _print_split(title: str, comparison: SplitComparison) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    print(f"baseline total     : {comparison.baseline.total_predictions}")
    print(f"learned total      : {comparison.learned.total_predictions}")
    print(f"baseline accepted  : {comparison.baseline.accepted_predictions}")
    print(f"learned accepted   : {comparison.learned.accepted_predictions}")
    print(f"delta              : {comparison.accepted_delta:+d}")
    print(f"baseline avg conf  : {comparison.baseline.average_confidence:.3f}")
    print(f"learned avg conf   : {comparison.learned.average_confidence:.3f}")
    print(f"avg conf delta     : {comparison.average_confidence_delta:+.3f}")


def _print_relation_table(title: str, comparison: SplitComparison) -> None:
    print(f"\nPer Relation - {title}")
    print("----------------" + "-" * len(title))
    print(
        f"{'relation':10s} {'base_acc':>9} {'learn_acc':>9} "
        f"{'delta':>7} {'base_avg':>9} {'learn_avg':>9}"
    )
    for relation in RELATIONS_OF_INTEREST:
        base = comparison.baseline.per_relation[relation]
        learned = comparison.learned.per_relation[relation]
        print(
            f"{relation:10s} {base.accepted_predictions:>9d} "
            f"{learned.accepted_predictions:>9d} "
            f"{learned.accepted_predictions - base.accepted_predictions:>+7d} "
            f"{base.average_confidence:>9.3f} {learned.average_confidence:>9.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate learned trust transfer on train/test graph splits."
    )
    parser.add_argument("--input", default=DATA_PATH)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = load_relations_csv(args.input)
    train_rows, test_rows = split_relations(rows, args.train_fraction)
    train, test, profile, existing_audits = run_transfer_experiment(
        data_path=args.input,
        threshold=args.threshold,
        train_fraction=args.train_fraction,
    )

    print("Trust Transfer Experiment")
    print("=========================")
    print(f"threshold        : {args.threshold:.2f}")
    print(f"relations total  : {len(rows)}")
    print(f"train relations  : {len(train_rows)}")
    print(f"test relations   : {len(test_rows)}")
    print(f"audit files used : {len(existing_audits)}")
    print(f"audit rows used  : {profile.counts.get('used_rows', 0)}")

    _print_split("TRAIN", train)
    _print_split("TEST", test)
    _print_relation_table("TRAIN", train)
    _print_relation_table("TEST", test)


if __name__ == "__main__":
    main()
