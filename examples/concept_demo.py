"""Demo v0.8: automatic concept formation."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World

# Four complete lifecycles: three use "семя", one uses "косточка".
LIFECYCLES = [
    ("яблоня",            "яблоко",     "семя"),
    ("груша",             "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",  "семя"),
    ("персиковое_дерево", "персик",     "косточка"),
]

PARTIAL = ("сливовое_дерево", "слива")


def build_world() -> World:
    w = World()
    for tree, fruit, connector in LIFECYCLES:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {connector}")
        w.observe(f"{connector} вырастает в {tree}")
    w.observe(f"{PARTIAL[0]} производит {PARTIAL[1]}")
    return w


def section(title: str) -> None:
    print(f"\n{'═' * 62}")
    print(f"  {title}")
    print("═" * 62)


def main() -> None:
    # ── Before: raw world ───────────────────────────────────────────
    section("World before discover_concepts()")
    world = build_world()
    print(f"  objects   : {len(world.get_objects())}")
    print(f"  relations : {len(world.get_relations())}")
    print()
    for r in world.get_relations():
        print(f"  {r.source:22s} --{r.relation_type}--> {r.target}")

    # ── Discover ────────────────────────────────────────────────────
    section("world.discover_concepts()")
    concepts = world.discover_concepts()
    for c in concepts:
        print(f"\n  [{c.id}]  kind={c.kind}  conf={c.confidence:.2f}")
        print(f"    members          : {c.members}")
        print(f"    common_relations : {c.common_relations}")

    # ── After: member_of edges added ────────────────────────────────
    section("World after discover_concepts()")
    print(f"  objects   : {len(world.get_objects())}")
    print(f"  relations : {len(world.get_relations())}")
    print()
    member_rels = [r for r in world.get_relations() if r.relation_type == "member_of"]
    for r in member_rels:
        print(f"  {r.source:22s} --member_of--> {r.target}")

    # ── Concept membership lookup ────────────────────────────────────
    section("Membership lookup via explain_object")
    for name in ["яблоко", "персик", "семя"]:
        print(f"\n  {world.explain_object(name)}")

    # ── Similarity + concept together ───────────────────────────────
    section("Prediction after add_similarity('слива', 'персик', 0.9)")
    world.add_similarity("слива", "персик", 0.9)
    preds = world.predict_missing_links()
    for p in preds:
        print(f"  {p.source:22s} --{p.relation_type}--> {p.target:22s}  conf={p.confidence:.2f}")
        print(f"    {p.reason}")


if __name__ == "__main__":
    main()
