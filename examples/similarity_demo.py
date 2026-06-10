"""Demo v0.7: similarity-aware prediction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World

LIFECYCLES = [
    ("яблоня",             "яблоко",  "семя"),
    ("груша",              "груша_плод", "семя"),
    ("персиковое_дерево",  "персик",  "косточка"),
]
PARTIAL = ("сливовое_дерево", "слива")


def build_world() -> World:
    w = World()
    for tree, fruit, connector in LIFECYCLES:
        w.observe(f"{tree} дала {fruit}")
        w.observe(f"{fruit} содержит {connector}")
        w.observe(f"{connector} вырастает в {tree}")
    w.observe(f"{PARTIAL[0]} дала {PARTIAL[1]}")
    return w


def section(title: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print("─" * 60)


def show_predictions(world: World) -> None:
    preds = world.predict_missing_links()
    if not preds:
        print("  No predictions found")
        return
    seen: set[tuple] = set()
    for p in preds:
        key = (p.source, p.relation_type, p.target)
        if key in seen:
            continue
        seen.add(key)
        print(f"  {p.source:22s} --{p.relation_type}--> {p.target:22s}  conf={p.confidence:.2f}")
        print(f"    {p.reason}")


def main() -> None:
    section("World: 3 complete lifecycles + 1 partial anchor")
    world = build_world()
    for tree, fruit, conn in LIFECYCLES:
        print(f"  {tree:22s} -> {fruit:12s} -> {conn}")
    print(f"  {PARTIAL[0]:22s} -> {PARTIAL[1]:12s}  (partial — connector unknown)")

    # ── Run 1: no similarity ──────────────────────────────────────────
    section("Predictions WITHOUT similarity  (majority vote)")
    show_predictions(world)

    # ── Add similarity: слива is like персик ─────────────────────────
    section("Adding: world.add_similarity('слива', 'персик', 0.9)")
    world.add_similarity("слива", "персик", 0.9)
    print(f"  similarity('слива', 'персик') = {world.similarity('слива', 'персик'):.2f}")

    # ── Run 2: with similarity ────────────────────────────────────────
    section("Predictions WITH similarity  (nearest-template match)")
    show_predictions(world)

    # ── Score sweep ───────────────────────────────────────────────────
    section("Confidence sweep: similarity score vs predicted confidence")
    print(f"  {'score':>6}  {'confidence':>12}  connector")
    for score in [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
        w = build_world()
        w.add_similarity("слива", "персик", score)
        preds = w.predict_missing_links()
        p = next((p for p in preds if p.source == "слива"), None)
        if p:
            conf = p.confidence
            conn = p.target
        else:
            conf, conn = 0.0, "—"
        print(f"  {score:>6.1f}  {conf:>12.3f}  {conn}")


if __name__ == "__main__":
    main()
