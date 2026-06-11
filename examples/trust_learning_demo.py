"""
Demo: learn trust priors from human audit labels.

This is deliberately non-neural: no backprop, no learned weights.  It maps
manual labels to scores and averages them in interpretable buckets.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import build_world_from_relations, load_relations_csv
from core.pattern_prediction import PatternBasedPredictor
from core.trust_learning import TrustProfile, learn_trust_from_audits

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
AUDIT_PATHS = [
    os.path.join(_ROOT, "data", "audit_all.csv"),
    os.path.join(_ROOT, "data", "audit_mixed_out.csv"),
    os.path.join(_ROOT, "data", "audit_drift_aware.csv"),
]
DATA_PATH = os.path.join(_ROOT, "data", "conceptnet_sample.csv")
OUT_PATH = os.path.join(_ROOT, "data", "trust_profile.json")


def _print_table(title: str, values: dict[str, float], counts: dict[str, int], limit: int = 12) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    if not values:
        print("  (none learned)")
        return
    for key, score in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:limit]:
        print(f"  {key:28s} trust={score:.3f}  n={counts.get(key, 0)}")


def _print_profile(profile: TrustProfile) -> None:
    print("Audit-driven trust profile")
    print("==========================")
    print(f"rows seen : {profile.counts.get('rows', 0)}")
    print(f"rows used : {profile.counts.get('used_rows', 0)}")
    print(f"missing files: {profile.counts.get('missing_files', 0)}")
    _print_table(
        "Relation Trust",
        profile.relation_trust,
        profile.counts.get("relation_trust", {}),
    )
    _print_table(
        "Rule Trust",
        profile.rule_trust,
        profile.counts.get("rule_trust", {}),
    )
    _print_table(
        "Drift Trust",
        profile.drift_trust,
        profile.counts.get("drift_trust", {}),
    )
    _print_table(
        "Evidence Node Trust",
        profile.evidence_trust,
        profile.counts.get("evidence_trust", {}),
    )


def _rerun_predictions(profile: TrustProfile) -> None:
    if not os.path.exists(DATA_PATH):
        print(f"\nSample not found: {DATA_PATH}")
        return
    rows = load_relations_csv(DATA_PATH)
    world = build_world_from_relations(rows)
    preds = PatternBasedPredictor(world.get_relations()).predict_from_bigrams(
        min_count=5,
        min_confidence=0.0,
        max_intermediate_degree=20,
        relation_trust=profile.relation_trust,
        use_relation_drift=True,
    )
    print("\nTop predictions using learned relation_trust")
    print("--------------------------------------------")
    for pred in preds[:10]:
        via = ", ".join(pred.evidence[:3])
        print(
            f"  {pred.source} --{pred.relation_type}--> {pred.target} "
            f"conf={pred.confidence:.3f} via={via}"
        )


def main() -> None:
    existing = [path for path in AUDIT_PATHS if os.path.exists(path)]
    if not existing:
        print("No audit files found.")
        return

    print("Loading audit files:")
    for path in existing:
        print(f"  {os.path.relpath(path, _ROOT)}")

    profile = learn_trust_from_audits(existing)
    _print_profile(profile)
    profile.to_json(OUT_PATH)
    print(f"\nSaved trust profile: {os.path.relpath(OUT_PATH, _ROOT)}")
    _rerun_predictions(profile)


if __name__ == "__main__":
    main()
