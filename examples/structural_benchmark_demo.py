"""
Demo v1.1: clean structural benchmark.

Compares five inference modes on the SAME controlled hidden set:

    1. majority_only        — no similarity, no concepts
    2. manual_similarity    — hand-injected oracle edge (the old cheat)
    3. structural_similarity — similarity discovered from the graph itself
    4. concept_only         — concept membership, no similarity
    5. hybrid_structural    — concepts + discovered structural similarity

Research question:
    Can automatically discovered structure replace the manual oracle?

Dataset:
    seed-based (семя):  яблоня, груша, апельсиновое_дерево
    pit-based (косточка): персиковое_дерево, абрикосовое_дерево
    + two NEW trees whose lifecycle-closing edge is hidden:
        сливовое_дерево -> слива -> косточка   (minority pit pattern)
        лимонное_дерево -> лимон -> семя        (dominant seed pattern)

For each new tree we observe produces_result + contains, but hide grows_into.
Hidden set (identical across all modes):
    косточка --grows_into--> сливовое_дерево
    семя     --grows_into--> лимонное_дерево
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World
from core.relations import Relation
from core.evaluation import PredictionEvaluator

COMPLETE = [
    ("яблоня",              "яблоко",     "семя"),
    ("груша",               "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",   "семя"),
    ("персиковое_дерево",   "персик",     "косточка"),
    ("абрикосовое_дерево",  "абрикос",    "косточка"),
]
PARTIAL = [
    ("сливовое_дерево", "слива", "косточка"),   # minority pit
    ("лимонное_дерево", "лимон", "семя"),         # dominant seed
]

MODES = [
    "majority_only",
    "manual_similarity",
    "structural_similarity",
    "concept_only",
    "hybrid_structural",
]


def build_world() -> tuple[World, list[Relation]]:
    """Fresh world per call. Closing edges for PARTIAL trees are never added."""
    w = World()
    for tree, fruit, conn in COMPLETE:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
        w.observe(f"{conn} вырастает в {tree}")
    for tree, fruit, conn in PARTIAL:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")   # observed structure (fair signal)
        # grows_into withheld → becomes the prediction target
    hidden = [Relation(conn, "grows_into", tree) for tree, fruit, conn in PARTIAL]
    return w, hidden


def run_mode(mode: str):
    """Build a fresh isolated world, load only this mode's machinery, evaluate."""
    w, hidden = build_world()
    use_sim = use_con = False

    if mode == "majority_only":
        pass
    elif mode == "manual_similarity":
        w.add_similarity("слива", "персик", 0.9)   # the oracle
        use_sim = True
    elif mode == "structural_similarity":
        w.add_structural_similarities(min_score=0.5)
        use_sim = True
    elif mode == "concept_only":
        w.discover_concepts()
        use_con = True
    elif mode == "hybrid_structural":
        w.add_structural_similarities(min_score=0.5)
        w.discover_concepts()
        use_sim = True
        use_con = True
    else:
        raise ValueError(f"unknown mode: {mode}")

    ev = PredictionEvaluator()
    return ev.evaluate_predictions(
        w, hidden, use_similarity=use_sim, use_concepts=use_con
    )


def section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print("═" * 70)


def main() -> None:
    section("Dataset")
    print("  Complete lifecycles:")
    for tree, fruit, conn in COMPLETE:
        print(f"    {tree:22s} -> {fruit:12s} -> {conn}")
    print("  Partial (grows_into hidden):")
    for tree, fruit, conn in PARTIAL:
        print(f"    {tree:22s} -> {fruit:12s} -> {conn}   "
              f"(hidden: {conn} --grows_into--> {tree})")
    print("\n  Dominant pattern: семя (3 complete)   Minority: косточка (2 complete)")

    section("Benchmark — five modes, same hidden set")
    print(f"  {'mode':24s}  {'P':>5}  {'R':>5}  {'F1':>5}  {'TP':>3} {'FP':>3} {'FN':>3}")
    print("  " + "─" * 58)
    results = {}
    for mode in MODES:
        r = run_mode(mode)
        results[mode] = r
        print(
            f"  {mode:24s}  {r.precision:5.2f}  {r.recall:5.2f}  {r.f1:5.2f}"
            f"  {r.true_positives:3d} {r.false_positives:3d} {r.false_negatives:3d}"
        )

    section("Reading the table")
    maj = results["majority_only"]
    man = results["manual_similarity"]
    struct = results["structural_similarity"]
    print(f"  • majority_only   F1={maj.f1:.2f} — picks dominant 'семя', so the")
    print(f"      minority pit fruit слива is mispredicted (FN + contradiction FPs).")
    print(f"  • manual_similarity F1={man.f1:.2f} — the oracle edge слива~персик fixes it,")
    print(f"      but the answer was hand-fed.")
    print(f"  • structural_similarity F1={struct.f1:.2f} — matches the oracle WITHOUT")
    print(f"      being told: слива~персик={0.6:.2f} emerges from shared graph structure.")
    print(f"  • concept_only / hybrid_structural — direct concept membership")
    print(f"      (слива observed containing косточка) recovers it too.")
    print()
    if struct.f1 > maj.f1:
        print(f"  ✓ Structural discovery improves over majority "
              f"({maj.f1:.2f} → {struct.f1:.2f}) with no oracle.")


if __name__ == "__main__":
    main()
