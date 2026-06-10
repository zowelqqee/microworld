"""Demo: feed observations into the world and explore what was learned."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World, AbstractionMiner

OBSERVATIONS = [
    "яблоня дала яблоко",
    "курица снесла яйцо",
    "завод произвел машину",
    "автор написал книгу",
    "огонь вызвал дым",
    "дождь вызвал наводнение",
    "пчела дала мёд",
    "фабрика произвёл ткань",
    "учитель дал знание",
    "солнце вызвал тепло",
]


def section(title: str) -> None:
    print(f"\n{'=' * 50}")
    print(f"  {title}")
    print("=" * 50)


def main() -> None:
    world = World()
    miner = AbstractionMiner()

    section("Feeding observations")
    for text in OBSERVATIONS:
        event = world.observe(text)
        if event:
            print(f"  OK  {event}")
        else:
            print(f"  --  (unparsed) {text!r}")

    section("All objects")
    for obj in sorted(world.get_objects(), key=lambda o: o.name):
        print(f"  {obj}")

    section("All relations")
    for rel in world.get_relations():
        print(f"  {rel}")
        print(f"    evidence: {rel.evidence[0]!r}")

    section("Explain individual objects")
    for name in ("яблоня", "огонь", "автор"):
        print()
        print(world.explain_object(name))

    section("Explain specific relation")
    print(world.explain_relation("дождь", "наводнение"))

    section("Discovered abstractions")
    abstractions = miner.mine(world.get_relations())
    for ab in abstractions:
        print(f'\n  Pattern: "{ab.pattern}"  (relation_type={ab.relation_type!r})')
        for inst in ab.instances:
            print(f"    {inst.source:12s} -> {inst.target}")

    section("Analogies")
    analogies = miner.analogies(abstractions)
    if analogies:
        for line in analogies:
            print(f"  * {line}")
    else:
        print("  (no analogies found — need at least 2 instances per pattern)")


if __name__ == "__main__":
    main()
