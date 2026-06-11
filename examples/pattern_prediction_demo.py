"""
Demo: pattern-based link prediction on ConceptNet sample.

Workflow:
  1. Load data/conceptnet_sample.csv
  2. Discover frequent transitive relation bigrams
  3. Show raw vs. hub-filtered predictions side-by-side
  4. Evaluate by hiding known transitive edges and measuring recovery

Does NOT touch the lifecycle PredictionEngine.
"""
import sys
import os
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.patterns import PatternDiscoveryEngine
from core.pattern_prediction import (
    PatternBasedPredictor,
    evaluate_pattern_prediction_recovery,
)

_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))


def section(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print("═" * 72)


def _print_pred_table(preds, label: str, n: int = 20) -> None:
    print(f"\n  {label}  (top {min(n, len(preds))} of {len(preds)} total)\n")
    w_col = max(
        (len(p.source) + len(p.target) + len(p.relation_type) + 6)
        for p in preds[:n]
    ) if preds else 40
    w_col = max(w_col, 40)
    print(f"  {'Prediction':{w_col}s}  {'Conf':>6}  Via")
    print("  " + "─" * (w_col + 26))
    for p in preds[:n]:
        pred_str = f"{p.source} --{p.relation_type}--> {p.target}"
        via = ", ".join(p.evidence[:3]) + (" …" if len(p.evidence) > 3 else "")
        print(f"  {pred_str:{w_col}s}  {p.confidence:>6.3f}  {via}")


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Sample not found: {DATA_PATH}")
        print("Generate it with:  python scripts/build_conceptnet_sample.py")
        return

    # ── load ──────────────────────────────────────────────────────────────────
    rows = load_relations_csv(DATA_PATH)
    w = build_world_from_relations(rows)
    rels = w.get_relations()
    print(f"\n  Loaded {len(rels)} relations, {len(w.get_objects())} objects")

    # ── top transitive patterns ───────────────────────────────────────────────
    section("Transitive bigram patterns  (r -> r,  min_count=5)")
    engine = PatternDiscoveryEngine(rels)
    bigrams = engine.discover_relation_bigrams(min_count=5)
    transitive = [p for p in bigrams if p.relations[0] == p.relations[1]]

    print(f"  {'Relation':20s}  {'Count':>6}  {'Support':>8}  {'Conf (base)':>12}")
    print("  " + "─" * 54)
    for p in transitive:
        rel = p.relations[0]
        conf = min(0.95, 0.5 + 0.05 * math.log(p.count + 1))
        print(f"  {rel:20s}  {p.count:>6d}  {p.support:>8.4f}  {conf:>12.3f}")

    # ── hub degree stats ──────────────────────────────────────────────────────
    section("Intermediate-node degree distribution  (top 15 hubs)")
    predictor = PatternBasedPredictor(rels)
    # collect all distinct intermediates that appear in 2-hop same-relation chains
    intermediates: dict[str, int] = {}
    for r1 in rels:
        for r2_type, _ in predictor._outgoing[r1.target]:
            if r2_type == r1.relation_type:
                node = r1.target
                intermediates[node] = predictor.get_total_degree(node)

    top_hubs = sorted(intermediates.items(), key=lambda kv: -kv[1])[:15]
    print(f"  {'Node':25s}  {'Total degree':>12}  {'Chain intermediary count':>24}")
    print("  " + "─" * 66)
    for node, deg in top_hubs:
        chain_cnt = predictor.get_chain_intermediate_count(node)
        print(f"  {node:25s}  {deg:>12d}  {chain_cnt:>24d}")

    # ── raw predictions ───────────────────────────────────────────────────────
    section("Raw predictions  (no hub penalty,  min_count=5)")
    raw_preds = predictor.predict_from_bigrams(min_count=5, hub_penalty=False)
    _print_pred_table(raw_preds, "Raw (unpenalised)")

    # ── hub-penalised predictions ─────────────────────────────────────────────
    section("Hub-penalised predictions  (hub_penalty=True,  min_count=5)")
    hub_preds = predictor.predict_from_bigrams(min_count=5, hub_penalty=True)
    _print_pred_table(hub_preds, "Hub-penalised")

    # ── degree-filtered predictions ───────────────────────────────────────────
    section("Degree-filtered predictions  (max_intermediate_degree=20,  min_count=5)")
    filtered_preds = predictor.predict_from_bigrams(
        min_count=5, hub_penalty=True, max_intermediate_degree=20
    )
    _print_pred_table(filtered_preds, "Degree-filtered (degree <= 20)")

    # ── side-by-side comparison ───────────────────────────────────────────────
    section("Side-by-side: raw vs. hub-penalised  (first 15)")
    raw_keys   = {(p.source, p.relation_type, p.target) for p in raw_preds}
    hub_keys   = {(p.source, p.relation_type, p.target) for p in hub_preds}
    raw_only   = raw_keys - hub_keys
    print(f"  Raw total        : {len(raw_preds)}")
    print(f"  Hub-penalised    : {len(hub_preds)}")
    print(f"  Dropped by penalty filter (conf < 0.5 after penalty): {len(raw_only)}")

    if raw_only:
        raw_conf = {(p.source, p.relation_type, p.target): p for p in raw_preds}
        print(f"\n  Examples of suppressed high-hub predictions:")
        w_col = 48
        print(f"  {'Prediction':{w_col}s}  {'Raw conf':>8}  Via")
        print("  " + "─" * (w_col + 20))
        for key in sorted(raw_only)[:10]:
            p = raw_conf[key]
            pred_str = f"{p.source} --{p.relation_type}--> {p.target}"
            via = ", ".join(p.evidence[:2])
            print(f"  {pred_str:{w_col}s}  {p.confidence:>8.3f}  {via}")

    # ── examples with full explanation ────────────────────────────────────────
    section("Detailed explanations — first 5 hub-penalised predictions")
    for p in hub_preds[:5]:
        print()
        print("  " + predictor.explain_prediction(p).replace("\n", "\n  "))

    # ── evaluation: hide-and-recover ──────────────────────────────────────────
    section("Evaluation: hide-and-recover  (max_hidden=100)")
    result = evaluate_pattern_prediction_recovery(w, max_hidden=100, min_count=5)
    print(f"  Hidden transitive edges  : {result.hidden_count}")
    print(f"  Total predictions        : {result.prediction_count}")
    print(f"  True positives           : {result.true_positives}")
    print(f"  False positives          : {result.false_positives}")
    print(f"  False negatives          : {result.false_negatives}")
    print(f"  Precision                : {result.precision:.3f}")
    print(f"  Recall                   : {result.recall:.3f}")
    print(f"  F1                       : {result.f1:.3f}")
    print()

    if result.hidden_count == 0:
        print("  ℹ  No existing transitive closures found in this sample.")
        print("     All pattern predictions are novel inferences.")
    elif result.recall > 0:
        print(f"  ✓ Predictor recovered {result.recall:.0%} of hidden transitive edges.")


if __name__ == "__main__":
    main()
