"""Demo v0.4: consolidation — clusters, cycles, causal chains, confidence."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World, ConsolidationEngine


OBSERVATIONS = [
    # Apple tree lifecycle — forms a cycle + cluster
    "яблоня дала яблоко",
    "яблоко содержит семя",
    "семя вырастает в яблоню",
    "яблоня требует воду",
    "засуха уменьшает воду",
    "дождь помогает урожаю",

    # Repeated with different surface forms → more evidence on same relations
    "яблони дала яблоко",      # яблони → яблоня  (2nd evidence for яблоня->яблоко)
    "яблоко содержит семена",  # семена → семя    (2nd evidence for яблоко->семя)
    "семя вырастает яблоню",   # 2nd evidence for семя->яблоня

    # Fire chain
    "огонь вызвал дым",
    "огонь уничтожает лес",
    "лес помогает осадки",
    "осадки уменьшает засуху",

    # Business chain
    "завод произвел продукт",
    "продукт вызвал прибыль",
    "прибыль помогает инвестиции",
    "инвестиции усиливает производство",
    "производство требует ресурсы",

    # Extra produces_result instances to push cluster confidence
    "курица снесла яйцо",
    "автор написал книгу",
    "пчела дала мёд",
]


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def bar(conf: float, width: int = 20) -> str:
    filled = int(conf * width)
    return "[" + "█" * filled + "░" * (width - filled) + f"] {conf:.2f}"


def main() -> None:
    world = World()

    section("Feeding observations")
    for text in OBSERVATIONS:
        event = world.observe(text)
        if event:
            print(f"  {event.subject:14s} --{event.verb}--> {event.object}")
        else:
            print(f"  (unparsed) {text!r}")

    patterns = world.consolidate()

    # ── Relation clusters ────────────────────────────────────────────
    clusters = [p for p in patterns if p.kind == "relation_cluster"]
    section(f"Relation clusters  ({len(clusters)} found)")
    for p in sorted(clusters, key=lambda x: -x.confidence):
        print(f"\n  {p.name}  {bar(p.confidence)}")
        for m in p.members:
            print(f"    • {m}")

    # ── Cycles ───────────────────────────────────────────────────────
    cycles = [p for p in patterns if p.kind == "cycle"]
    section(f"Cycles  ({len(cycles)} found)")
    for p in cycles:
        print(f"\n  {p.name}")
        print(f"  confidence: {bar(p.confidence)}")
        print(f"  evidence ({len(p.evidence)} entries):")
        for ev in p.evidence[:4]:
            print(f"    {ev!r}")
        if len(p.evidence) > 4:
            print(f"    ... and {len(p.evidence) - 4} more")

    # ── Causal chains ────────────────────────────────────────────────
    chains = [p for p in patterns if p.kind == "causal_chain"]
    section(f"Causal chains  ({len(chains)} found)")
    for p in sorted(chains, key=lambda x: -x.confidence):
        print(f"  {bar(p.confidence)}  {p.name}")

    # ── Relation strength scores ─────────────────────────────────────
    section("Relation strength scores  (evidence-based)")
    engine = ConsolidationEngine(world.get_relations())
    scores = engine.score_relation_strength()
    for key, score in sorted(scores.items(), key=lambda kv: -kv[1]):
        if score > 0.3:
            print(f"  {bar(score)}  {key}")

    # ── Summary ──────────────────────────────────────────────────────
    section("Summary")
    print(f"  Objects   : {len(world.get_objects())}")
    print(f"  Relations : {len(world.get_relations())}")
    print(f"  Clusters  : {len(clusters)}")
    print(f"  Cycles    : {len(cycles)}")
    print(f"  Chains    : {len(chains)}")
    print(f"  Patterns  : {len(patterns)}")


if __name__ == "__main__":
    main()
