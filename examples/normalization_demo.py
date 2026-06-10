"""Demo v0.3: entity normalization — inflected forms resolve to canonical names."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World, EntityNormalizer, CausalReasoner


def section(title: str) -> None:
    print(f"\n{'=' * 58}")
    print(f"  {title}")
    print("=" * 58)


def main() -> None:
    world = World()

    # ------------------------------------------------------------------
    # 1. Same entity, multiple surface forms
    # ------------------------------------------------------------------
    section("Same entity — different surface forms")

    observations = [
        "семя вырастает в яблоню",   # object surface: яблоню  → canonical: яблоня
        "яблоня дала яблоко",
        "яблоко содержит семена",    # object surface: семена  → canonical: семя
        "яблони требует воду",       # subject surface: яблони → canonical: яблоня
        "засуха уменьшает воду",
        "дождь помогает урожаю",     # object surface: урожаю  → canonical: урожай
    ]

    for text in observations:
        event = world.observe(text)
        if event:
            print(f"  {text!r:45s}  →  {event.subject} --{event.verb}--> {event.object}")
        else:
            print(f"  (unparsed) {text!r}")

    # ------------------------------------------------------------------
    # 2. Objects stored under canonical names
    # ------------------------------------------------------------------
    section("Objects stored under canonical names")
    for obj in sorted(world.get_objects(), key=lambda o: o.name):
        print(f"  {obj.name}")

    # ------------------------------------------------------------------
    # 3. Relations deduplicated despite different surface forms
    # ------------------------------------------------------------------
    section("Relations (canonical names throughout)")
    for rel in world.get_relations():
        print(f"  {rel.source:12s} --{rel.relation_type}--> {rel.target}")
        for ev in rel.evidence:
            print(f"    evidence: {ev!r}")

    # ------------------------------------------------------------------
    # 4. explain_object accepts inflected input
    # ------------------------------------------------------------------
    section("explain_object with inflected input ('яблоню')")
    print(world.explain_object("яблоню"))

    # ------------------------------------------------------------------
    # 5. explain_chain with inflected source/target
    # ------------------------------------------------------------------
    section("explain_chain with inflected inputs")
    for src, tgt in [
        ("семя",   "яблоня"),   # canonical → canonical
        ("семя",   "яблоню"),   # canonical → inflected
        ("засуха", "воду"),     # both inflected-ish
    ]:
        result = world.explain_chain(src, tgt)
        print(f"\n  {src!r} → {tgt!r}:")
        for line in result:
            print(f"    {line}")

    # ------------------------------------------------------------------
    # 6. Deduplication: same relation via two different surface forms
    # ------------------------------------------------------------------
    section("Deduplication: 'яблоня содержит семя' vs 'яблоня содержит семена'")
    w2 = World()
    w2.observe("яблоня содержит семя")
    w2.observe("яблоня содержит семена")   # семена → семя
    rels = w2.get_relations()
    print(f"  Relations stored: {len(rels)}  (expected: 1)")
    if rels:
        print(f"  {rels[0]}")
        print(f"  Evidence entries: {len(rels[0].evidence)}")
        for ev in rels[0].evidence:
            print(f"    {ev!r}")

    # ------------------------------------------------------------------
    # 7. EntityNormalizer directly
    # ------------------------------------------------------------------
    section("EntityNormalizer — add custom alias at runtime")
    norm = EntityNormalizer()
    norm.add_alias("урожаю", "урожай")   # already seeded, but shown for clarity
    norm.add_alias("яблочку", "яблоко")  # custom
    print(f"  normalize('яблочку') = {norm.normalize('яблочку')!r}")
    print(f"  normalize('яблоком') = {norm.normalize('яблоком')!r}")
    print(f"  get_aliases('яблоко') = {norm.get_aliases('яблоко')}")
    print(f"  normalize('unknown')  = {norm.normalize('unknown')!r}  (unchanged)")


if __name__ == "__main__":
    main()
