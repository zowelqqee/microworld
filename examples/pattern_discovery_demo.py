"""
Demo: pattern discovery on ConceptNet sample.

Answers: "What structures actually exist inside ConceptNet?"
This is exploratory — no predictions are made here.

Run the extractor first if the sample is missing:
    python scripts/build_conceptnet_sample.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.patterns import PatternDiscoveryEngine

_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))


def section(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print("═" * 72)


def _fmt_pattern(relations: tuple[str, ...]) -> str:
    return " -> ".join(relations)


def _fmt_example(nodes: tuple[str, ...], sep: str = " -> ") -> str:
    return sep.join(nodes)


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Sample not found: {DATA_PATH}")
        print("Generate it with:  python scripts/build_conceptnet_sample.py")
        return

    # ── load ──────────────────────────────────────────────────────────────────
    rows = load_relations_csv(DATA_PATH)
    w = build_world_from_relations(rows)
    rels = w.get_relations()

    engine = PatternDiscoveryEngine(rels)

    total_objects   = len(w.get_objects())
    total_relations = len(rels)

    # ── discover ──────────────────────────────────────────────────────────────
    bigrams_all  = engine.discover_relation_bigrams(min_count=1)
    trigrams_all = engine.discover_relation_trigrams(min_count=1)
    bigrams_5    = engine.discover_relation_bigrams(min_count=5)
    trigrams_5   = engine.discover_relation_trigrams(min_count=5)

    coverage = engine.compute_coverage(bigrams_5)

    # ── metrics ───────────────────────────────────────────────────────────────
    section("Metrics")
    print(f"  total_objects           : {total_objects}")
    print(f"  total_relations         : {total_relations}")
    print(f"  unique_bigram_patterns  : {len(bigrams_all)}"
          f"   (≥5: {len(bigrams_5)})")
    print(f"  unique_trigram_patterns : {len(trigrams_all)}"
          f"   (≥5: {len(trigrams_5)})")
    print(f"  pattern_coverage (≥5)   : {coverage:.3f}"
          f"  — fraction of edges in a discovered bigram chain")

    # ── top 20 bigrams ────────────────────────────────────────────────────────
    section("Top 20 bigrams  (min_count=5)")
    _print_table(bigrams_5[:20], total_relations)

    # ── top 20 trigrams ───────────────────────────────────────────────────────
    section("Top 20 trigrams  (min_count=5)")
    _print_table(trigrams_5[:20], total_relations)

    # ── example chains for top bigrams ────────────────────────────────────────
    section("Example chains — top 5 bigrams")
    for pat in bigrams_5[:5]:
        chain_label = _fmt_pattern(pat.relations)
        print(f"\n  Pattern: {chain_label}  (count={pat.count})")
        for ex in pat.examples:
            print(f"      {_fmt_example(ex)}")

    # ── example chains for top trigrams ──────────────────────────────────────
    if trigrams_5:
        section("Example chains — top 5 trigrams")
        for pat in trigrams_5[:5]:
            chain_label = _fmt_pattern(pat.relations)
            print(f"\n  Pattern: {chain_label}  (count={pat.count})")
            for ex in pat.examples:
                print(f"      {_fmt_example(ex)}")
    else:
        section("Trigrams")
        print("  No trigram patterns found with min_count=5.")
        if trigrams_all:
            print(f"  {len(trigrams_all)} pattern(s) found with min_count=1:")
            for pat in trigrams_all[:5]:
                print(f"    {_fmt_pattern(pat.relations):40s}  count={pat.count}")


def _print_table(patterns, total_relations: int) -> None:
    if not patterns:
        print("  (none)")
        return
    col1 = max(len(_fmt_pattern(p.relations)) for p in patterns)
    col1 = max(col1, 30)
    hdr = f"  {'Pattern':{col1}s}  {'Count':>7}  {'Support':>8}"
    print(hdr)
    print("  " + "─" * (col1 + 20))
    for p in patterns:
        label = _fmt_pattern(p.relations)
        print(f"  {label:{col1}s}  {p.count:>7d}  {p.support:>8.4f}")


if __name__ == "__main__":
    main()
