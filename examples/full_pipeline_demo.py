"""
Full Microworld pipeline demo: explicit memory, reasoning, audit, and update.

This uses a small curated research/project-memory graph so the complete loop is
readable in a terminal:

INPUT -> OBSERVE -> WORLD GRAPH -> SLEEP / CONSOLIDATION -> PREDICT
-> AUDIT -> TRUST UPDATE -> RE-PREDICT -> OUTPUT
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.memory_pipeline import (
    MicroworldPipeline,
    accepted_predictions,
    compare_prediction_runs,
    prediction_key,
)
from core.pattern_prediction import PatternPrediction

DEMO_FACTS: list[tuple[str, str, str, str]] = [
    ("microworld", "uses", "pattern_prediction", "input: microworld uses pattern_prediction"),
    ("pattern_prediction", "uses", "relation_trust", "input: pattern_prediction uses relation_trust"),
    ("relation_trust", "uses", "human_audit", "input: relation_trust uses human_audit"),
    ("human_audit", "uses", "audit_labels", "input: human_audit uses audit_labels"),
    ("microworld", "uses", "mixed_reasoning", "input: microworld uses mixed_reasoning"),
    ("mixed_reasoning", "uses", "transitive_reasoning", "input: mixed_reasoning uses transitive_reasoning"),
    ("mixed_reasoning", "uses", "relation_trust", "input: mixed_reasoning uses relation_trust"),
    ("mixed_reasoning", "extends", "transitive_reasoning", "input: mixed_reasoning extends transitive_reasoning"),
    ("transitive_reasoning", "extends", "pattern_prediction", "input: transitive_reasoning extends pattern_prediction"),
    ("microworld", "is_a", "reasoning_engine", "input: microworld is_a reasoning_engine"),
    ("reasoning_engine", "capable_of", "explain_predictions", "input: reasoning_engine capable_of explain_predictions"),
    ("pattern_prediction", "is_a", "reasoning_layer", "input: pattern_prediction is_a reasoning_layer"),
    ("mixed_reasoning", "is_a", "reasoning_layer", "input: mixed_reasoning is_a reasoning_layer"),
    ("reasoning_layer", "has_property", "inspectable", "input: reasoning_layer has_property inspectable"),
    ("relation_trust", "learned_from", "human_audit", "input: relation_trust learned_from human_audit"),
    ("human_audit", "evaluates", "predictions", "input: human_audit evaluates predictions"),
    ("predictions", "affect", "reasoning", "input: predictions affect reasoning"),
    ("audit_feedback", "affect", "relation_trust", "input: audit_feedback affect relation_trust"),
    ("relation_drift", "lowers", "confidence", "input: relation_drift lowers confidence"),
    ("node_quality", "rejects", "sister_naked", "input: node_quality rejects noisy node sister_naked"),
    ("sister_naked", "rejects", "noisy_nodes", "input: noisy node would imply more noise"),
    ("project_report", "made_of", "paper", "input: project_report made_of paper"),
    ("paper", "made_of", "wood", "input: paper made_of wood"),
]

SYNTHETIC_AUDIT_LABELS: dict[tuple[str, str, str], str] = {
    ("microworld", "uses", "relation_trust"): "correct",
    ("pattern_prediction", "uses", "human_audit"): "plausible",
    ("relation_trust", "uses", "audit_labels"): "wrong",
    ("microworld", "capable_of", "explain_predictions"): "correct",
    ("mixed_reasoning", "extends", "pattern_prediction"): "plausible",
    ("project_report", "made_of", "wood"): "plausible",
}


def build_demo_pipeline() -> MicroworldPipeline:
    pipeline = MicroworldPipeline(prediction_min_count=1, prediction_threshold=0.4)
    for source, relation_type, target, evidence in DEMO_FACTS:
        pipeline.observe(source, relation_type, target, evidence=evidence)
    return pipeline


def apply_synthetic_audit(
    pipeline: MicroworldPipeline,
    predictions: list[PatternPrediction],
) -> None:
    by_key = {prediction_key(prediction): prediction for prediction in predictions}
    for key, label in SYNTHETIC_AUDIT_LABELS.items():
        prediction = by_key.get(key)
        if prediction is not None:
            pipeline.audit(prediction, label)


def _print_observations() -> None:
    print("SECTION 1 - Input observations")
    print("------------------------------")
    for source, relation_type, target, _evidence in DEMO_FACTS:
        print(f"  {source} --{relation_type}--> {target}")


def _print_world_before_sleep(pipeline: MicroworldPipeline) -> None:
    print("\nSECTION 2 - World before sleep")
    print("------------------------------")
    print(f"objects   : {len(pipeline.world.get_objects())}")
    print(f"relations : {len(pipeline.world.get_relations())}")


def _print_sleep_report(pipeline: MicroworldPipeline) -> None:
    report = pipeline.sleep()
    print("\nSECTION 3 - Sleep / consolidation")
    print("---------------------------------")
    print(f"discovered concepts              : {len(report.concepts)}")
    for concept in report.concepts[:4]:
        members = ", ".join(concept.members[:4])
        pattern = "; ".join(concept.common_relations)
        print(f"  {concept.id} members=[{members}] pattern={pattern}")
    print(f"discovered patterns              : {len(report.patterns)}")
    for pattern in report.patterns[:5]:
        chain = " -> ".join(pattern.relations)
        print(f"  {chain} count={pattern.count}")
    print(f"structural similarities          : {len(report.structural_similarities)}")
    for left, right, score in report.structural_similarities[:3]:
        print(f"  {left} ~ {right} score={score:.3f}")
    print("relation trust before learning   :")
    for relation in ("uses", "extends", "capable_of", "has_property", "made_of"):
        print(f"  {relation:12s} {report.relation_trust.get(relation, 0.0):.3f}")
    print("node quality rejects             :")
    if not report.low_quality_nodes:
        print("  (none)")
    for node, score in report.low_quality_nodes.items():
        print(f"  {node:12s} quality={score:.3f}")


def _print_predictions(title: str, predictions: list[PatternPrediction], limit: int = 5) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for prediction in predictions[:limit]:
        evidence = ", ".join(prediction.evidence) if prediction.evidence else "none"
        drift = f" drift={prediction.drift_type}" if prediction.drift_type else ""
        print(
            f"  {prediction.source} --{prediction.relation_type}--> {prediction.target} "
            f"conf={prediction.confidence:.3f}{drift}"
        )
        print(f"    evidence: {evidence}")
        print(f"    reason  : {prediction.reason}")


def _print_audit(pipeline: MicroworldPipeline, predictions: list[PatternPrediction]) -> None:
    apply_synthetic_audit(pipeline, predictions)
    print("\nSECTION 5 - Human audit")
    print("-----------------------")
    for record in pipeline.audit_records:
        prediction = record.prediction
        print(
            f"  {prediction.source} --{prediction.relation_type}--> {prediction.target} "
            f"label={record.label} score={record.score:.1f}"
        )


def _print_trust_update(pipeline: MicroworldPipeline):
    update = pipeline.learn_from_audit()
    print("\nSECTION 6 - Trust learning")
    print("--------------------------")
    print("relation        before   after   audit_n")
    for relation in sorted(update.learned):
        before = update.before.get(relation, 0.0)
        after = update.after.get(relation, 0.0)
        count = update.counts.get(relation, 0)
        print(f"{relation:12s} {before:>7.3f} {after:>7.3f} {count:>7d}")
    return update


def _print_rerun(
    before: list[PatternPrediction],
    after: list[PatternPrediction],
    threshold: float,
) -> None:
    survived, suppressed, changed = compare_prediction_runs(before, after, threshold)
    print("\nSECTION 7 - Re-run predictions")
    print("------------------------------")
    print(f"threshold              : {threshold:.2f}")
    print(f"survived above threshold: {len(survived)}")
    print(f"suppressed             : {len(suppressed)}")
    print(f"changed confidence     : {len(changed)}")

    print("\n  survived")
    for prediction in survived[:5]:
        print(f"    {prediction.source} --{prediction.relation_type}--> {prediction.target} conf={prediction.confidence:.3f}")

    print("\n  suppressed")
    if not suppressed:
        print("    (none)")
    for prediction in suppressed[:5]:
        print(f"    {prediction.source} --{prediction.relation_type}--> {prediction.target} old_conf={prediction.confidence:.3f}")

    print("\n  changed confidence")
    for prediction, delta in changed[:5]:
        print(
            f"    {prediction.source} --{prediction.relation_type}--> {prediction.target} "
            f"new_conf={prediction.confidence:.3f} delta={delta:+.3f}"
        )


def main() -> None:
    pipeline = build_demo_pipeline()
    _print_observations()
    _print_world_before_sleep(pipeline)
    _print_sleep_report(pipeline)

    predictions_before = pipeline.predict()
    print("\nSECTION 4 - Predictions")
    print("-----------------------")
    print(f"total predictions                 : {len(predictions_before)}")
    print(
        f"accepted at threshold {pipeline.prediction_threshold:.2f}      : "
        f"{len(accepted_predictions(predictions_before, pipeline.prediction_threshold))}"
    )
    _print_predictions("Top predictions before audit", predictions_before, limit=5)

    _print_audit(pipeline, predictions_before)
    _print_trust_update(pipeline)

    predictions_after = pipeline.predict()
    _print_rerun(predictions_before, predictions_after, pipeline.prediction_threshold)

    print("\nSECTION 8 - Why this matters")
    print("----------------------------")
    print(
        "Microworld changes future reasoning behavior from explicit audit feedback "
        "without neural retraining."
    )


if __name__ == "__main__":
    main()
