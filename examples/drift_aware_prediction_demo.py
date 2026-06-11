"""
Demo: drift-aware scoring for made_of transitive predictions.

This is not a weights+biases model.  It is an explicit semantic-level penalty
for made_of chains whose target shifts from direct material toward raw,
atomic, or abstract components.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.pattern_prediction import PatternBasedPredictor
from core.relations import Relation


def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def _prediction_map(preds):
    return {
        (p.source, p.relation_type, p.target): p
        for p in preds
    }


def main() -> None:
    relations = _rels(
        # direct material: no drift penalty
        ("song", "made_of", "music"),
        ("music", "made_of", "sounds"),
        # raw material: slight penalty
        ("book", "made_of", "paper"),
        ("paper", "made_of", "wood"),
        # atomic component: stronger penalty
        ("blood", "made_of", "haemoglobin"),
        ("haemoglobin", "made_of", "iron"),
        # abstract component: penalty
        ("community", "made_of", "culture"),
        ("culture", "made_of", "ideals"),
    )
    predictor = PatternBasedPredictor(relations)
    raw = _prediction_map(
        predictor.predict_from_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
        )
    )
    drift = _prediction_map(
        predictor.predict_from_bigrams(
            min_count=1,
            min_confidence=0.0,
            hub_penalty=False,
            use_relation_drift=True,
        )
    )

    rows = [
        ("song", "made_of", "sounds"),
        ("book", "made_of", "wood"),
        ("blood", "made_of", "iron"),
        ("community", "made_of", "ideals"),
    ]

    print("\nDrift-aware made_of scoring")
    print(f"{'prediction':38s} {'raw':>7} {'drift':>7}  annotation")
    print("-" * 72)
    for key in rows:
        raw_pred = raw[key]
        drift_pred = drift[key]
        label = f"{key[0]} --{key[1]}--> {key[2]}"
        annotation = (
            "no drift"
            if drift_pred.drift_type is None
            else f"drift={drift_pred.drift_type}, penalty={drift_pred.drift_penalty:.2f}"
        )
        print(
            f"{label:38s} {raw_pred.confidence:>7.3f} "
            f"{drift_pred.confidence:>7.3f}  {annotation}"
        )

    print("\nExample reason:")
    print(drift[("blood", "made_of", "iron")].reason)


if __name__ == "__main__":
    main()
