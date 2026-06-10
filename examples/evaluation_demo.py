"""Demo v0.6: evaluation benchmark for missing-link prediction."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World
from core.evaluation import PredictionEvaluator

# 4 семя-based lifecycles + 1 косточка outlier (персик)
LIFECYCLES = [
    ("яблоня",          "яблоко",      "семя"),
    ("груша",           "груша_плод",  "семя"),
    ("апельсин_дерево", "апельсин",    "семя"),
    ("лимон_дерево",    "лимон",       "семя"),
    ("персик_дерево",   "персик",      "косточка"),   # outlier!
]


def section(title: str) -> None:
    print(f"\n{'=' * 62}")
    print(f"  {title}")
    print("=" * 62)


def bar(v: float, width: int = 20) -> str:
    return "[" + "█" * int(v * width) + "░" * (width - int(v * width)) + f"] {v:.3f}"


def build_world() -> World:
    w = World()
    for tree, fruit, connector in LIFECYCLES:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {connector}")
        w.observe(f"{connector} вырастает в {tree}")
    return w


def run_scenario(label: str, seed: int, ratio: float = 0.3) -> None:
    section(f"{label}  (seed={seed}, ratio={ratio})")
    world = build_world()
    evaluator = PredictionEvaluator()

    modified_world, hidden = evaluator.hide_relations(world, ratio=ratio, seed=seed)
    result = evaluator.evaluate_predictions(modified_world, hidden)

    print(f"\n  Hidden relations ({result.total_hidden}):")
    for r in hidden:
        print(f"    ✂  {r.source:25s} --{r.relation_type}--> {r.target}")

    from core.prediction import PredictionEngine
    predictions = PredictionEngine(modified_world.get_relations()).predict_missing_links()
    print(f"\n  Predictions ({result.total_predictions}):")
    for p in predictions:
        hidden_keys = {(r.source, r.relation_type, r.target) for r in hidden}
        marker = "✓" if (p.source, p.relation_type, p.target) in hidden_keys else "✗ (FP)"
        print(f"    {marker}  {p.source:25s} --{p.relation_type}--> {p.target}  (conf={p.confidence:.2f})")

    hidden_keys = {(r.source, r.relation_type, r.target) for r in hidden}
    pred_keys = {(p.source, p.relation_type, p.target) for p in predictions}
    missed = hidden_keys - pred_keys
    if missed:
        print(f"\n  Missed (FN):")
        for src, rel, tgt in sorted(missed):
            print(f"    ✗  {src:25s} --{rel}--> {tgt}")

    print(f"\n  Metrics:")
    print(f"    Precision  {bar(result.precision)}")
    print(f"    Recall     {bar(result.recall)}")
    print(f"    F1         {bar(result.f1)}")
    print(f"    TP={result.true_positives}  FP={result.false_positives}  FN={result.false_negatives}")


def main() -> None:
    # ── Show the world ────────────────────────────────────────────────
    section("World: 5 lifecycle chains")
    world = build_world()
    for tree, fruit, connector in LIFECYCLES:
        label = "(косточка!)" if connector == "косточка" else ""
        print(f"  {tree:20s} -> {fruit:15s} -> {connector} {label}")
    print(f"\n  Total relations : {len(world.get_relations())}")
    hideable = [r for r in world.get_relations() if r.relation_type in {"contains", "grows_into"}]
    print(f"  Hideable (contains/grows_into) : {len(hideable)}")
    print(f"  Anchor   (produces_result)     : {len(world.get_relations()) - len(hideable)}")

    # ── Scenario 1: seed=42 → perfect recovery ────────────────────────
    run_scenario("Scenario 1 — hidden relations all use 'семя' connector", seed=42)

    # ── Scenario 2: seed=0 → косточка outlier exposed ─────────────────
    run_scenario("Scenario 2 — косточка outlier exposed", seed=0)

    # ── World.evaluate_prediction_recovery convenience wrapper ────────
    section("World.evaluate_prediction_recovery (seed=42)")
    result = world.evaluate_prediction_recovery(ratio=0.3, seed=42)
    print(f"\n  {result}")

    # ── Sweep: varying ratio ──────────────────────────────────────────
    section("Precision / Recall sweep  (seed=42, varying ratio)")
    print(f"  {'ratio':>6}  {'hidden':>6}  {'preds':>6}  {'P':>6}  {'R':>6}  {'F1':>6}")
    for ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]:
        w = build_world()
        ev = PredictionEvaluator()
        mod, hid = ev.hide_relations(w, ratio=ratio, seed=42)
        res = ev.evaluate_predictions(mod, hid)
        print(
            f"  {ratio:>6.1f}  {res.total_hidden:>6}  {res.total_predictions:>6}"
            f"  {res.precision:>6.3f}  {res.recall:>6.3f}  {res.f1:>6.3f}"
        )


if __name__ == "__main__":
    main()
