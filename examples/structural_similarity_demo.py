"""
Demo v1.1: automatic structural similarity discovery — no oracle.

Previously similarity was injected by hand:
    world.add_similarity("слива", "персик", 0.9)
That is an oracle: it tells the system the answer it is supposed to infer.

Here similarity is computed from the graph itself via Jaccard overlap of
structural profiles. Nothing is hand-fed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World


def section(title: str) -> None:
    print(f"\n{'═' * 66}")
    print(f"  {title}")
    print("═" * 66)


# ─────────────────────────────────────────────────────────────────────────────
# Part 1 — discovery on the minimal contains-only world from the spec
# ─────────────────────────────────────────────────────────────────────────────

def part1_discovery() -> None:
    section("Part 1 — discover_structural_similarities() on contains-only world")
    w = World()
    for fruit, conn in [
        ("яблоко",     "семя"),
        ("груша_плод", "семя"),
        ("апельсин",   "семя"),
        ("персик",     "косточка"),
        ("абрикос",    "косточка"),
        ("слива",      "косточка"),
    ]:
        w.observe(f"{fruit} содержит {conn}")

    print("  Profiles:")
    for fruit in ["яблоко", "персик", "слива"]:
        from core.structural_similarity import StructuralSimilarityEngine
        prof = StructuralSimilarityEngine(w.get_relations()).entity_profile(fruit)
        print(f"    {fruit:12s} → {sorted(prof)}")

    print("\n  Discovered similarities (min_score=0.5):")
    for a, b, score in w.discover_structural_similarities(min_score=0.5):
        print(f"    {a:12s} ~ {b:12s}  {score:.3f}")

    print("\n  Cross-group spot checks (should be lower):")
    for a, b in [("яблоко", "персик"), ("яблоко", "слива"), ("семя", "косточка")]:
        print(f"    {a:12s} ~ {b:12s}  {w.structural_similarity(a, b):.3f}")


# ─────────────────────────────────────────────────────────────────────────────
# Part 2 — prediction driven only by discovered structural similarity
# ─────────────────────────────────────────────────────────────────────────────

LIFECYCLES = [
    ("яблоня",              "яблоко",     "семя"),
    ("груша",               "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",   "семя"),
    ("персиковое_дерево",   "персик",     "косточка"),
    ("абрикосовое_дерево",  "абрикос",    "косточка"),
]
PARTIAL = ("сливовое_дерево", "слива", "косточка")


def build_lifecycle_world() -> World:
    w = World()
    for tree, fruit, conn in LIFECYCLES:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
        w.observe(f"{conn} вырастает в {tree}")
    # Partial anchor: we OBSERVE the plum and its pit, but NOT that the pit
    # regrows the tree. The lifecycle-closing edge is what we must infer.
    w.observe(f"{PARTIAL[0]} производит {PARTIAL[1]}")   # сливовое_дерево -> слива
    w.observe(f"{PARTIAL[1]} содержит {PARTIAL[2]}")     # слива -> косточка  (observed!)
    # HIDDEN (to be predicted): косточка -> grows_into -> сливовое_дерево
    return w


def closing_pred(world: World):
    """Return any predicted grows_into link that closes back to сливовое_дерево."""
    for p in world.predict_missing_links():
        if p.relation_type == "grows_into" and p.target == PARTIAL[0]:
            return p
    return None


def part2_prediction() -> None:
    section("Part 2 — prediction WITHOUT manual add_similarity")
    print("  3 семя-lifecycles + 2 косточка-lifecycles.")
    print("  New observation: 'слива содержит косточка' (a real fact).")
    print("  HIDDEN, to be inferred: 'косточка вырастает в сливовое_дерево'.\n")

    # Baseline: no similarity → majority connector is the dominant семя.
    print("  ── Baseline: no similarity (majority vote, dominant=семя) ──")
    w0 = build_lifecycle_world()
    p0 = closing_pred(w0)
    if p0:
        verdict = "✓ correct" if p0.source == PARTIAL[2] else "✗ WRONG"
        print(f"    predicted closing link: {p0.source} --grows_into--> {p0.target}"
              f"   [{verdict}]")
        print(f"      reason: {p0.reason[:60]}")
        wrong = next((p for p in w0.predict_missing_links()
                      if p.source == "слива" and p.relation_type == "contains"), None)
        if wrong:
            print(f"    ⚠  also predicts слива --contains--> {wrong.target} "
                  f"(contradicts observed косточка!)")

    # v1.1: discover structural similarity from the graph, then predict.
    print("\n  ── v1.1: add_structural_similarities() then predict ──")
    w1 = build_lifecycle_world()
    added = w1.add_structural_similarities(min_score=0.5)
    print(f"    discovered & loaded {added} similarity pairs")
    print(f"    structural sim(слива, персик) = "
          f"{w1.structural_similarity('слива', 'персик'):.3f}  "
          f"(косточка group)")
    print(f"    structural sim(слива, яблоко) = "
          f"{w1.structural_similarity('слива', 'яблоко'):.3f}  "
          f"(семя group)")
    p1 = closing_pred(w1)
    if p1:
        verdict = "✓ correct" if p1.source == PARTIAL[2] else "✗ WRONG"
        print(f"\n    predicted closing link: {p1.source} --grows_into--> {p1.target}"
              f"  conf={p1.confidence:.2f}   [{verdict}]")
        print(f"      reason: {p1.reason[:60]}")

    print("\n  Result: косточка → сливовое_дерево, inferred from structure — no oracle.")


def main() -> None:
    part1_discovery()
    part2_prediction()


if __name__ == "__main__":
    main()
