"""
Application-level demo: Microworld as an explainable graph reasoning engine.

The output is intentionally curated for readability, but examples are looked up
from live ConceptNet-sample predictions instead of being printed as static text.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import build_world_from_relations, load_relations_csv
from core.node_quality import node_quality
from core.pattern_prediction import PatternBasedPredictor, PatternPrediction
from core.reasoning_relations import DEFAULT_DISABLED_RELATIONS, DEFAULT_REASONING_RELATIONS
from core.relation_trust import DEFAULT_RELATION_TRUST

_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))

STRONG_EXAMPLES = [
    (("song", "made_of", "sounds"), "A song is made of music, and music is made of sounds."),
    (("adrenal_cortex", "part_of", "endocrine_system"), "Specific anatomy rolls up to a larger body system."),
    (("al_ain", "part_of", "united_arab_emirates"), "City-to-emirate-to-country composition is a useful geographic rollup."),
    (("alaska_peninsula", "part_of", "united_states"), "Region-to-state-to-country composition stays at the same semantic level."),
    (("talbe", "made_of", "wood"), "Useful despite typo in source; table/tree/wood chain is recoverable."),
]

WEAK_EXAMPLES = [
    (("book", "made_of", "cellulose"), "useful but indirect: paper composition exposes underlying material."),
    (("blood", "made_of", "iron"), "useful but indirect: relation drifts toward contains_element."),
    (("dna", "made_of", "hydrogen"), "useful but indirect: molecular composition drifts to atomic component."),
    (("toast", "made_of", "wheat"), "useful but indirect: food product points back to raw ingredient."),
    (("community", "made_of", "ideals"), "useful but indirect: social entity drifts into abstract component."),
]

RISKY_EXAMPLES = [
    (("internet", "made_of", "sister_naked"), "dataset noise: low-quality node should be rejected by node quality."),
    (("troll", "made_of", "epic_fail"), "dataset noise: meme/junk node should be rejected by node quality."),
    (("aconcagua", "part_of", "bolivia"), "overgeneralized geography: Andes spans countries; peak is not part_of each one."),
    (("arm_bone", "part_of", "armchair"), "sense ambiguity: arm as body part vs furniture component."),
]


def section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))


def prediction_key(prediction: PatternPrediction) -> tuple[str, str, str]:
    return (prediction.source, prediction.relation_type, prediction.target)


def format_evidence_chain(prediction: PatternPrediction) -> str:
    """Format the first evidence path as A -r-> B -r-> C."""
    if not prediction.evidence:
        return f"{prediction.source} --{prediction.relation_type}--> {prediction.target}"
    via = prediction.evidence[0]
    return (
        f"{prediction.source} --{prediction.relation_type}--> {via} "
        f"--{prediction.relation_type}--> {prediction.target}"
    )


def format_prediction_block(
    prediction: PatternPrediction,
    interpretation: str,
    status: str | None = None,
    include_drift: bool = False,
) -> str:
    """Return a compact, readable block for one prediction."""
    lines = []
    label = f"{prediction.source} --{prediction.relation_type}--> {prediction.target}"
    if status:
        label = f"{label}  [{status}]"
    lines.append(label)
    lines.append(f"  confidence    : {prediction.confidence:.3f}")
    lines.append(f"  evidence chain: {format_evidence_chain(prediction)}")
    if include_drift:
        drift = prediction.drift_type or "none"
        lines.append(
            f"  drift         : {drift} (penalty={prediction.drift_penalty:.2f})"
        )
    lines.append(f"  reason        : {prediction.reason}")
    lines.append(f"  interpretation: {interpretation}")
    return "\n".join(lines)


def _prediction_map(preds: list[PatternPrediction]) -> dict[tuple[str, str, str], PatternPrediction]:
    return {prediction_key(pred): pred for pred in preds}


def _print_curated(
    title: str,
    examples: list[tuple[tuple[str, str, str], str]],
    predictions: dict[tuple[str, str, str], PatternPrediction],
    include_drift: bool = False,
    status: str | None = None,
) -> None:
    section(title)
    shown = 0
    for key, interpretation in examples:
        pred = predictions.get(key)
        if pred is None:
            print(f"{key[0]} --{key[1]}--> {key[2]}  [not found in current sample]")
            continue
        print(format_prediction_block(pred, interpretation, status=status, include_drift=include_drift))
        print()
        shown += 1
    if shown == 0:
        print("No curated examples found in the current sample.")


def _find_disabled_at_location(
    disabled_predictions: dict[tuple[str, str, str], PatternPrediction],
) -> PatternPrediction | None:
    preferred = ("admiralty_island", "at_location", "caribbean_sea")
    if preferred in disabled_predictions:
        return disabled_predictions[preferred]
    return next(
        (pred for key, pred in disabled_predictions.items() if key[1] == "at_location"),
        None,
    )


def _print_risky(
    default_predictions: dict[tuple[str, str, str], PatternPrediction],
    diagnostic_predictions: dict[tuple[str, str, str], PatternPrediction],
    disabled_predictions: dict[tuple[str, str, str], PatternPrediction],
) -> None:
    section("3. Rejected / Risky Predictions")

    disabled = _find_disabled_at_location(disabled_predictions)
    if disabled is not None and prediction_key(disabled) not in default_predictions:
        print(format_prediction_block(
            disabled,
            "disabled by default: at_location is too noisy for reasoning.",
            status="disabled relation",
        ))
        print()

    for key, interpretation in RISKY_EXAMPLES:
        pred = diagnostic_predictions.get(key)
        if pred is None:
            print(f"{key[0]} --{key[1]}--> {key[2]}  [not found in diagnostics]")
            continue
        status = "risky"
        if min(node_quality(pred.source), node_quality(pred.target), *(node_quality(v) for v in pred.evidence)) < 0.3:
            status = "low-quality node"
        print(format_prediction_block(pred, interpretation, status=status, include_drift=True))
        print()


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Sample not found: {DATA_PATH}")
        return

    rows = load_relations_csv(DATA_PATH)
    world = build_world_from_relations(rows)
    predictor = PatternBasedPredictor(world.get_relations())

    common_args = dict(
        min_count=1,
        min_confidence=0.0,
        hub_penalty=True,
        relation_trust=DEFAULT_RELATION_TRUST,
        use_relation_drift=True,
    )
    default_predictions = _prediction_map(
        predictor.predict_from_bigrams(
            **common_args,
            use_node_quality=True,
            min_node_quality=0.3,
        )
    )
    diagnostic_predictions = _prediction_map(
        predictor.predict_from_bigrams(
            **common_args,
            use_node_quality=True,
            min_node_quality=0.0,
        )
    )
    disabled_predictions = _prediction_map(
        predictor.predict_from_bigrams(
            **common_args,
            use_node_quality=False,
            include_disabled_relations=True,
        )
    )

    print("Microworld Application Demo")
    print("===========================")
    print(f"Loaded {len(world.get_relations())} relations from {os.path.relpath(DATA_PATH)}")

    _print_curated("1. Strong Predictions", STRONG_EXAMPLES, default_predictions)
    _print_curated(
        "2. Weak But Useful Predictions",
        WEAK_EXAMPLES,
        diagnostic_predictions,
        include_drift=True,
    )
    _print_risky(default_predictions, diagnostic_predictions, disabled_predictions)

    section("4. Summary")
    print(f"enabled relations : {', '.join(sorted(DEFAULT_REASONING_RELATIONS))}")
    print(f"disabled relations: {', '.join(sorted(DEFAULT_DISABLED_RELATIONS))}")
    print("reasoning layers  : hub penalty, relation trust, relation drift")
    print("node quality      : enabled for normal predictions; relaxed only for diagnostics")
    print("audit highlights  :")
    print("  ConceptNet human audit: 104 reviewed, 78.8% useful")
    print("  made_of 86.2%")
    print("  part_of 76.7%")
    print("  is_a 75.6%")
    print("  mixed reasoning: 76.7% useful on 30 reviewed")


if __name__ == "__main__":
    main()
