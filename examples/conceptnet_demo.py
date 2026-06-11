"""
Demo: ConceptNet import pipeline.

Loads data/conceptnet_sample.csv (built by scripts/build_conceptnet_sample.py),
prints stats, then runs concept discovery, structural similarity, and
link prediction on the English knowledge-graph slice.

Run the extractor first if the sample doesn't exist:
    python scripts/build_conceptnet_sample.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from collections import Counter

from core.datasets import load_relations_csv, build_world_from_relations

_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))


def section(title: str) -> None:
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print("═" * 70)


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Sample not found: {DATA_PATH}")
        print("Generate it with:")
        print("    python scripts/build_conceptnet_sample.py")
        return

    # ── load ──────────────────────────────────────────────────────────────────
    section("Loading data/conceptnet_sample.csv")
    rows = load_relations_csv(DATA_PATH)
    print(f"  rows loaded : {len(rows)}")

    w = build_world_from_relations(rows)
    print(f"  objects     : {len(w.get_objects())}")
    print(f"  relations   : {len(w.get_relations())}")

    # ── distribution ──────────────────────────────────────────────────────────
    section("Relation type distribution")
    counts = Counter(r.relation_type for r in w.get_relations())
    max_count = max(counts.values()) if counts else 1
    for rel, n in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "█" * max(1, n * 28 // max_count)
        print(f"  {rel:16s} {n:6d}  {bar}")

    # ── sample edges ──────────────────────────────────────────────────────────
    section("Sample relations (first 15)")
    for r in w.get_relations()[:15]:
        print(f"  {r.source:22s} --{r.relation_type:14s}--> {r.target}")

    # ── concept discovery ─────────────────────────────────────────────────────
    section("Concept discovery")
    concepts = w.discover_concepts()
    print(f"  concepts discovered: {len(concepts)}")
    if concepts:
        top5 = sorted(concepts, key=lambda c: len(c.members), reverse=True)[:5]
        for c in top5:
            sample = ", ".join(c.members[:4])
            suffix = " …" if len(c.members) > 4 else ""
            print(f"  [{c.kind:16s}] {c.id}")
            print(f"      members ({len(c.members)}): {sample}{suffix}")

    # ── structural similarity ─────────────────────────────────────────────────
    section("Structural similarity (Jaccard ≥ 0.5)")
    n_pairs = w.add_structural_similarities(min_score=0.5)
    print(f"  similar pairs found: {n_pairs}")
    if n_pairs > 0:
        pairs = sorted(
            w.discover_structural_similarities(min_score=0.5),
            key=lambda x: -x[2],
        )
        for a, b, score in pairs[:10]:
            print(f"  {a:22s} ~ {b:22s}  {score:.3f}")

    # ── link prediction ───────────────────────────────────────────────────────
    section("Link predictions (top 15)")
    preds = w.predict_missing_links()
    print(f"  total predictions: {len(preds)}")
    top_preds = sorted(preds, key=lambda p: -p.confidence)[:15]
    for p in top_preds:
        print(
            f"  {p.source:22s} --{p.relation_type:14s}--> {p.target:22s}"
            f"  conf={p.confidence:.2f}  ({p.reason})"
        )


if __name__ == "__main__":
    main()
