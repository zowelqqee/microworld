"""
Demo: mixed-pattern link prediction on the ConceptNet sample.

Workflow:
  1. Load data/conceptnet_sample.csv
  2. Show the manually allowed mixed bigram rules
  3. Predict mixed-pattern links
  4. Print top predictions and representative explanations
"""
from collections import defaultdict
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.pattern_prediction import DEFAULT_MIXED_BIGRAM_RULES, PatternBasedPredictor
from core.reasoning_relations import DEFAULT_DISABLED_RELATIONS, DEFAULT_REASONING_RELATIONS

_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))


def section(title: str) -> None:
    print(f"\n{'=' * 72}")
    print(f"  {title}")
    print("=" * 72)


def _print_pred_table(preds, label: str, n: int = 30) -> None:
    print(f"\n  {label}  (top {min(n, len(preds))} of {len(preds)} total)\n")
    w_col = max(
        (len(p.source) + len(p.target) + len(p.relation_type) + 6)
        for p in preds[:n]
    ) if preds else 44
    w_col = max(w_col, 44)
    print(f"  {'Prediction':{w_col}s}  {'Conf':>6}  Via")
    print("  " + "-" * (w_col + 26))
    for p in preds[:n]:
        pred_str = f"{p.source} --{p.relation_type}--> {p.target}"
        via = ", ".join(p.evidence[:3]) + (" ..." if len(p.evidence) > 3 else "")
        print(f"  {pred_str:{w_col}s}  {p.confidence:>6.3f}  {via}")


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Sample not found: {DATA_PATH}")
        print("Generate it with:  python scripts/build_conceptnet_sample.py")
        return

    rows = load_relations_csv(DATA_PATH)
    w = build_world_from_relations(rows)
    rels = w.get_relations()
    predictor = PatternBasedPredictor(rels)

    print(f"\n  Loaded {len(rels)} relations, {len(w.get_objects())} objects")

    section("Reasoning Relation Policy")
    print(f"  Enabled core relations : {', '.join(sorted(DEFAULT_REASONING_RELATIONS))}")
    print(f"  Disabled by default    : {', '.join(sorted(DEFAULT_DISABLED_RELATIONS))}")

    section("Allowed Mixed Rules")
    for (r1, r2), out_rel in sorted(DEFAULT_MIXED_BIGRAM_RULES.items()):
        print(f"  {r1:14s} + {r2:14s} => {out_rel}")

    preds = predictor.predict_from_mixed_bigrams(min_count=5)

    section("Mixed Predictions")
    print(f"  Number of mixed predictions: {len(preds)}")
    _print_pred_table(preds, "Top mixed predictions", n=30)

    section("Top Predictions By Output Relation")
    by_relation = defaultdict(list)
    for p in preds:
        by_relation[p.relation_type].append(p)

    for rel_type in sorted(by_relation):
        rel_preds = sorted(
            by_relation[rel_type],
            key=lambda p: (-p.confidence, p.source, p.target),
        )
        print(f"\n  {rel_type} ({len(rel_preds)} predictions)")
        for p in rel_preds[:5]:
            print(
                f"    {p.source} --{p.relation_type}--> {p.target} "
                f"conf={p.confidence:.3f} via={', '.join(p.evidence[:3])}"
            )

    section("Examples With Evidence")
    for p in preds[:8]:
        print()
        print("  " + predictor.explain_prediction(p).replace("\n", "\n  "))


if __name__ == "__main__":
    main()
