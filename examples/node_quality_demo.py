"""
Demo: node quality filtering on pattern predictions.

Constructs a synthetic graph that contains deliberately bad intermediate nodes
(sister_naked, epic_fail) alongside clean ones (steel, iron) and shows
before/after predictions when use_node_quality=True.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.relations import Relation
from core.pattern_prediction import PatternBasedPredictor
from core.node_quality import node_quality, DEFAULT_BAD_TOKENS


def _rels(*triples) -> list[Relation]:
    return [Relation(s, r, t) for s, r, t in triples]


def section(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print("═" * 64)


def _print_preds(preds, label: str) -> None:
    print(f"\n  {label}  ({len(preds)} predictions)")
    if not preds:
        print("    (none)")
        return
    w = max(len(f"{p.source} --{p.relation_type}--> {p.target}") for p in preds)
    w = max(w, 40)
    print(f"  {'Prediction':{w}s}  {'Conf':>6}  Reason snippet")
    print("  " + "─" * (w + 30))
    for p in preds:
        pred_str = f"{p.source} --{p.relation_type}--> {p.target}"
        snippet  = p.reason.split("(", 1)[-1].rstrip(")")[:36]
        print(f"  {pred_str:{w}s}  {p.confidence:>6.3f}  {snippet}")


def main() -> None:
    # ── graph ─────────────────────────────────────────────────────────────────
    # Good chains (high-quality nodes)
    good_chains = _rels(
        *[(f"tool{i}", "made_of", "steel") for i in range(8)],
        ("steel",  "made_of", "iron"),
        *[(f"part{i}", "made_of", "iron") for i in range(8)],
        ("iron",   "made_of", "metal"),
    )
    # Bad chains (noisy intermediate nodes)
    bad_chains = _rels(
        *[(f"thing{i}", "made_of", "sister_naked") for i in range(8)],
        ("sister_naked", "made_of", "material"),
        *[(f"item{i}", "made_of", "epic_fail") for i in range(8)],
        ("epic_fail",    "made_of", "stuff"),
        # non-English compound node
        *[(f"obj{i}", "made_of", "tu_hermana_en_bolas") for i in range(8)],
        ("tu_hermana_en_bolas", "made_of", "whatever"),
    )
    all_rels = good_chains + bad_chains
    predictor = PatternBasedPredictor(all_rels)

    # ── node quality scores ───────────────────────────────────────────────────
    section("Node quality scores")
    sample_nodes = [
        "steel", "iron", "metal",
        "sister_naked", "epic_fail", "tu_hermana_en_bolas",
        "naked", "bolas", "fail",
        "a_very_long_name_that_exceeds_the_threshold_limit",
        "tool0",
    ]
    print(f"\n  {'Node':<45}  {'Quality':>7}  Low?")
    print("  " + "─" * 60)
    for node in sample_nodes:
        q    = node_quality(node)
        low  = "yes" if q < 0.3 else "no"
        print(f"  {node:<45}  {q:>7.3f}  {low}")

    # ── raw predictions ───────────────────────────────────────────────────────
    section("Raw predictions  (use_node_quality=False)")
    raw = predictor.predict_from_bigrams(min_count=1, hub_penalty=False,
                                          use_node_quality=False)
    _print_preds(raw, "Raw (no node quality filter)")

    # ── quality-filtered ──────────────────────────────────────────────────────
    section("Quality-filtered predictions  (use_node_quality=True, min=0.3)")
    filtered = predictor.predict_from_bigrams(min_count=1, hub_penalty=False,
                                               use_node_quality=True,
                                               min_node_quality=0.3)
    _print_preds(filtered, "After node quality filter")

    # ── comparison ───────────────────────────────────────────────────────────
    section("Summary")
    raw_keys      = {(p.source, p.relation_type, p.target) for p in raw}
    filtered_keys = {(p.source, p.relation_type, p.target) for p in filtered}
    suppressed    = raw_keys - filtered_keys
    print(f"\n  Raw predictions      : {len(raw_keys)}")
    print(f"  After quality filter : {len(filtered_keys)}")
    print(f"  Suppressed           : {len(suppressed)}")
    if suppressed:
        print("\n  Examples suppressed:")
        for src, rel, tgt in sorted(suppressed)[:8]:
            print(f"    {src} --{rel}--> {tgt}")
    print()


if __name__ == "__main__":
    main()
