"""
Demo v0.9.1: four isolated inference modes benchmarked on the same hidden set.

Research question:
  Do concept-aware and similarity-based inference improve on the majority-vote
  baseline for minority-pattern data?

World:
  - 3 complete семя-trees  (яблоня, груша, апельсин)
  - 2 complete косточка-trees  (персик, абрикос)

Partial observations (contains observed, grows_into hidden):
  - сливовое_дерево -> слива -> косточка  (minority pattern)
  - лимонное_дерево -> лимон -> семя       (dominant pattern)

Hidden (ground truth we predict):
  - косточка  --grows_into--> сливовое_дерево
  - семя      --grows_into--> лимонное_дерево
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import World
from core.evaluation import PredictionEvaluator
from core.relations import Relation

COMPLETE = [
    ("яблоня",              "яблоко",     "семя"),
    ("груша",               "груша_плод", "семя"),
    ("апельсиновое_дерево", "апельсин",   "семя"),
    ("персиковое_дерево",   "персик",     "косточка"),
    ("абрикосовое_дерево",  "абрикос",    "косточка"),
]

# Partial anchors: produces_result + contains are observed; grows_into is hidden.
PARTIAL = [
    ("сливовое_дерево", "слива",  "косточка"),   # minority
    ("лимонное_дерево", "лимон",  "семя"),        # dominant
]


def build_base_world() -> tuple[World, list[Relation]]:
    """
    Build a world with complete cycles + partial observations.
    Returns (world, hidden_relations).
    The hidden relations are the grows_into closing-links for the partial trees.
    The world itself never contains the hidden relations.
    """
    w = World()
    for tree, fruit, conn in COMPLETE:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")
        w.observe(f"{conn} вырастает в {tree}")

    # Observe produces_result + contains; withhold grows_into.
    hidden: list[Relation] = []
    for tree, fruit, conn in PARTIAL:
        w.observe(f"{tree} производит {fruit}")
        w.observe(f"{fruit} содержит {conn}")       # <-- this makes concept membership work
        hidden.append(Relation(conn, "grows_into", tree))

    return w, hidden


def section(title: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {title}")
    print("═" * 68)


def _score_row(
    label: str,
    use_sim: bool,
    use_con: bool,
    world: World,
    hidden: list[Relation],
    ev: PredictionEvaluator,
) -> None:
    # Build a fresh, unshared copy of the world for this mode so no state leaks.
    fresh = ev._build_world(world, world.get_relations())
    if use_con:
        fresh.discover_concepts()
    r = ev.evaluate_predictions(
        fresh, hidden,
        use_similarity=use_sim,
        use_concepts=use_con,
    )
    sim_tag = "yes" if use_sim else "no "
    con_tag = "yes" if use_con else "no "
    print(
        f"  {label:20s}  sim={sim_tag}  con={con_tag}  "
        f"P={r.precision:.2f}  R={r.recall:.2f}  F1={r.f1:.2f}  "
        f"TP={r.true_positives}  FP={r.false_positives}  FN={r.false_negatives}"
    )


def main() -> None:
    base_world, hidden = build_base_world()
    # Add similarity only to the base world; each mode copy decides whether to use it.
    base_world.add_similarity("слива", "персик", 0.9)

    section("World")
    print(f"  complete cycles : {len(COMPLETE)}")
    print(f"  partial anchors : {len(PARTIAL)}  (contains observed, grows_into hidden)")
    print(f"  hidden set      : {len(hidden)}")
    for r in hidden:
        print(f"    ✂  {r.source:15s} --{r.relation_type}--> {r.target}")
    print(f"  similarity      : слива ↔ персик (0.90)")

    section("Step-by-step walkthrough of each mode")
    ev = PredictionEvaluator()

    for label, use_sim, use_con in [
        ("majority_only",   False, False),
        ("similarity_only", True,  False),
        ("concept_only",    False, True),
        ("hybrid",          True,  True),
    ]:
        fresh = ev._build_world(base_world, base_world.get_relations())
        if use_con:
            concepts = fresh.discover_concepts()
            print(f"\n  [{label}]  concepts: {[c.id for c in concepts if c.kind == 'relation_group']}")
        else:
            print(f"\n  [{label}]")

        preds = fresh.predict_missing_links() if (use_sim and not use_con) else None
        from core.prediction import PredictionEngine
        sim_graph = fresh._similarity_graph if use_sim else None
        cons = fresh._concepts if use_con else []
        preds = PredictionEngine(
            fresh.get_relations(),
            similarity_graph=sim_graph,
            concepts=cons,
        ).predict_missing_links()

        hidden_keys = {(r.source, r.relation_type, r.target) for r in hidden}
        seen: set = set()
        for p in preds:
            key = (p.source, p.relation_type, p.target)
            if key in seen:
                continue
            seen.add(key)
            mark = "✓" if key in hidden_keys else "✗ FP"
            print(f"    {mark}  {p.source:20s} --{p.relation_type}--> {p.target:20s}  "
                  f"conf={p.confidence:.2f}  [{p.reason[:60]}]")

    section("Benchmark table — four isolated modes")
    print(
        f"  {'mode':20s}  {'sim':5s}  {'con':5s}  "
        f"{'P':>5}  {'R':>5}  {'F1':>5}  TP  FP  FN"
    )
    print("  " + "─" * 62)
    for label, use_sim, use_con in [
        ("majority_only",   False, False),
        ("similarity_only", True,  False),
        ("concept_only",    False, True),
        ("hybrid",          True,  True),
    ]:
        _score_row(label, use_sim, use_con, base_world, hidden, ev)

    section("World.evaluate_prediction_recovery — isolated API calls")
    for label, use_sim, use_con in [
        ("majority_only",   False, False),
        ("similarity_only", True,  False),
        ("concept_only",    False, True),
        ("hybrid",          True,  True),
    ]:
        # Build a fresh world each time so no shared state between calls.
        w, h = build_base_world()
        w.add_similarity("слива", "персик", 0.9)
        r = w.evaluate_prediction_recovery(
            ratio=0.3, seed=42,
            use_similarity=use_sim,
            use_concepts=use_con,
        )
        print(f"  {label:20s}  →  {r}")


if __name__ == "__main__":
    main()
