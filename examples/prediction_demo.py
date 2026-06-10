"""Demo v0.5: missing-link prediction via structural analogy."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World, PredictionEngine


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print("=" * 60)


def main() -> None:
    world = World()

    # ── Complete lifecycles (template instances) ─────────────────────
    section("Complete lifecycle observations (template source)")
    complete = [
        # яблоня lifecycle
        "яблоня производит яблоко",
        "яблоко содержит семя",
        "семя вырастает в яблоню",
        # апельсин lifecycle
        "апельсин_дерево производит апельсин",
        "апельсин содержит семя",
        "семя вырастает в апельсин_дерево",
    ]
    for text in complete:
        e = world.observe(text)
        if e:
            print(f"  {e.subject:20s} --{e.verb}--> {e.object}")
        else:
            print(f"  (unparsed) {text!r}")

    # ── Partial observations (incomplete lifecycles) ──────────────────
    section("Partial lifecycle observations (triggers prediction)")
    partial = [
        "груша производит груша_плод",
        "вишня производит вишня_плод",
    ]
    for text in partial:
        e = world.observe(text)
        if e:
            print(f"  {e.subject:20s} --{e.verb}--> {e.object}")
        else:
            print(f"  (unparsed) {text!r}")

    # ── Extract templates ─────────────────────────────────────────────
    section("Discovered templates")
    engine = PredictionEngine(world.get_relations())
    templates = engine.find_chain_templates()
    if not templates:
        print("  (no templates found)")
    for t in templates:
        print(f"  template: {' → '.join(t.relation_sequence)}")
        print(f"  instances ({len(t.instances)}):")
        for inst in t.instances:
            print(f"    {inst[0]:20s} → {inst[1]:20s} → {inst[2]}")
        connector = t.most_common_slot_value(2)
        print(f"  most common slot[2] (connector): {connector!r}")

    # ── Predictions ───────────────────────────────────────────────────
    section("Predicted missing links")
    predictions = world.predict_missing_links()
    if not predictions:
        print("  (no predictions)")
    for p in predictions:
        print()
        print(engine.explain_prediction(p))

    # ── Verification: predicted relations don't yet exist ─────────────
    section("Verification")
    existing = {(r.source, r.relation_type, r.target) for r in world.get_relations()}
    for p in predictions:
        key = (p.source, p.relation_type, p.target)
        status = "ALREADY EXISTS (bug)" if key in existing else "new (correct)"
        print(f"  {p.source} --{p.relation_type}--> {p.target}  [{status}]")

    # ── Single-instance baseline (confidence = 0.7) ───────────────────
    section("Confidence baseline: single template instance")
    w2 = World()
    for text in [
        "яблоня производит яблоко",
        "яблоко содержит семя",
        "семя вырастает в яблоню",
        "груша производит груша_плод",  # partial
    ]:
        w2.observe(text)
    preds2 = w2.predict_missing_links()
    for p in preds2:
        print(f"  conf={p.confidence:.2f}  {p.source} --{p.relation_type}--> {p.target}")


if __name__ == "__main__":
    main()
