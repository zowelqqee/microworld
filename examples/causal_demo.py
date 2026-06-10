"""Demo v0.2: causal chains and what-if simulation."""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World, CausalReasoner

APPLE_TREE_CYCLE = [
    "яблоня дала яблоко",
    "яблоко содержит семя",
    "семя вырастает в яблоню",   # parser strips "в"
    "яблоня требует воду",
    "засуха уменьшает воду",
    "дождь помогает урожаю",
]

FIRE_CHAIN = [
    "огонь вызвал дым",
    "огонь уничтожает лес",
    "лес помогает осадки",
    "осадки уменьшает засуху",
]

BUSINESS_CHAIN = [
    "завод произвел продукт",
    "продукт вызвал прибыль",
    "прибыль помогает инвестиции",
    "инвестиции усиливает производство",
    "производство требует ресурсы",
    "кризис уменьшает ресурсы",
]


def section(title: str) -> None:
    print(f"\n{'=' * 56}")
    print(f"  {title}")
    print("=" * 56)


def demo_scenario(name: str, observations: list[str]) -> World:
    world = World()
    section(f"Scenario: {name}")
    for text in observations:
        event = world.observe(text)
        status = f"OK  {event}" if event else f"--  (unparsed) {text!r}"
        print(f"  {status}")
    return world


def show_forward_traces(world: World, start: str, max_depth: int = 4) -> None:
    reasoner = CausalReasoner(world.get_relations())
    chains = reasoner.trace_forward(start, max_depth=max_depth)
    print(f"\n  Forward traces from {start!r}:")
    if not chains:
        print("    (none)")
    for chain in chains:
        print(f"    {chain}")


def show_what_if(world: World, removed: str) -> None:
    reasoner = CausalReasoner(world.get_relations())
    print(f"\n  what_if_removed({removed!r}):")
    for line in reasoner.what_if_removed(removed):
        print(f"    • {line}")


def show_explain_chain(world: World, source: str, target: str) -> None:
    reasoner = CausalReasoner(world.get_relations())
    print(f"\n  explain_chain({source!r} → {target!r}):")
    for line in reasoner.explain_chain(source, target):
        print(f"    {line}")


def main() -> None:
    # ── Apple tree lifecycle ─────────────────────────────────────────
    w1 = demo_scenario("Apple tree lifecycle", APPLE_TREE_CYCLE)

    show_forward_traces(w1, "яблоня")
    show_forward_traces(w1, "засуха")

    show_what_if(w1, "яблоко")
    show_what_if(w1, "вода")

    show_explain_chain(w1, "семя", "яблоню")
    show_explain_chain(w1, "засуха", "воду")

    # ── Fire / smoke chain ───────────────────────────────────────────
    w2 = demo_scenario("Fire and ecosystem", FIRE_CHAIN)

    show_forward_traces(w2, "огонь")
    show_what_if(w2, "лес")
    show_explain_chain(w2, "огонь", "засуху")   # огонь→лес→осадки→засуху

    # ── Business / profit chain ──────────────────────────────────────
    w3 = demo_scenario("Business chain", BUSINESS_CHAIN)

    show_forward_traces(w3, "завод")
    show_what_if(w3, "ресурсы")
    show_explain_chain(w3, "завод", "инвестиции")


if __name__ == "__main__":
    main()
