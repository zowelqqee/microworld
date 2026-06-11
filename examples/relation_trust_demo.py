"""
Demo: relation-trust confidence scaling on ConceptNet sample.

Shows before/after confidence for each relation type when the human-audit
trust priors from DEFAULT_RELATION_TRUST are applied.

Does NOT touch the lifecycle PredictionEngine.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import load_relations_csv, build_world_from_relations
from core.pattern_prediction import PatternBasedPredictor
from core.relation_trust import DEFAULT_RELATION_TRUST, UNKNOWN_RELATION_TRUST

_HERE = os.path.dirname(__file__)
DATA_PATH = os.path.normpath(os.path.join(_HERE, "..", "data", "conceptnet_sample.csv"))


def section(title: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  {title}")
    print("═" * 68)


def _group_by_relation(preds) -> dict:
    groups: dict[str, list] = {}
    for p in preds:
        groups.setdefault(p.relation_type, []).append(p)
    return groups


def main() -> None:
    if not os.path.exists(DATA_PATH):
        print(f"Sample not found: {DATA_PATH}")
        print("Generate it with:  python scripts/build_conceptnet_sample.py")
        return

    rows = load_relations_csv(DATA_PATH)
    w    = build_world_from_relations(rows)
    rels = w.get_relations()
    print(f"\n  Loaded {len(rels)} relations, {len(w.get_objects())} objects")

    predictor = PatternBasedPredictor(rels)

    preds_raw   = predictor.predict_from_bigrams(min_count=5, hub_penalty=True, relation_trust=None)
    preds_trust = predictor.predict_from_bigrams(min_count=5, hub_penalty=True,
                                                 relation_trust=DEFAULT_RELATION_TRUST)

    # ── relation trust table ──────────────────────────────────────────────────
    section("Relation trust priors  (from human audit)")
    print(f"  Unknown relation default: {UNKNOWN_RELATION_TRUST}")
    print()
    print(f"  {'Relation':20s}  {'Trust':>6}")
    print("  " + "─" * 30)
    all_rels = sorted({p.relation_type for p in preds_raw})
    for rel in all_rels:
        trust = DEFAULT_RELATION_TRUST.get(rel, UNKNOWN_RELATION_TRUST)
        marker = "  ← from audit" if rel in DEFAULT_RELATION_TRUST else "  ← default"
        print(f"  {rel:20s}  {trust:>6.3f}{marker}")

    # ── per-relation before/after ─────────────────────────────────────────────
    section("Per-relation: average confidence before vs. after trust scaling")
    raw_groups   = _group_by_relation(preds_raw)
    trust_groups = _group_by_relation(preds_trust)

    all_rel_types = sorted(raw_groups)
    w_rel = max(len(r) for r in all_rel_types) if all_rel_types else 12
    print(f"\n  {'Relation':{w_rel}s}  {'Trust':>6}  {'N (raw)':>8}  "
          f"{'Avg conf (raw)':>14}  {'N (trust)':>10}  {'Avg conf (trust)':>16}  {'Δ avg':>7}")
    print("  " + "─" * (w_rel + 68))

    for rel in all_rel_types:
        trust = DEFAULT_RELATION_TRUST.get(rel, UNKNOWN_RELATION_TRUST)
        rg = raw_groups.get(rel, [])
        tg = trust_groups.get(rel, [])
        avg_raw   = sum(p.confidence for p in rg)   / len(rg)   if rg   else 0.0
        avg_trust = sum(p.confidence for p in tg)   / len(tg)   if tg   else 0.0
        delta = avg_trust - avg_raw
        print(
            f"  {rel:{w_rel}s}  {trust:>6.3f}  {len(rg):>8d}  "
            f"{avg_raw:>14.4f}  {len(tg):>10d}  {avg_trust:>16.4f}  {delta:>+7.4f}"
        )

    # ── top predictions with trust ────────────────────────────────────────────
    section("Top 20 trust-scaled predictions")
    w_col = max(
        len(p.source) + len(p.target) + len(p.relation_type) + 6
        for p in preds_trust[:20]
    ) if preds_trust else 40
    w_col = max(w_col, 40)
    print(f"\n  {'Prediction':{w_col}s}  {'Conf':>6}  {'Trust':>6}  Reason snippet")
    print("  " + "─" * (w_col + 34))
    for p in preds_trust[:20]:
        pred_str = f"{p.source} --{p.relation_type}--> {p.target}"
        trust = DEFAULT_RELATION_TRUST.get(p.relation_type, UNKNOWN_RELATION_TRUST)
        snippet = p.reason.split("(", 1)[-1].rstrip(")")[:40]
        print(f"  {pred_str:{w_col}s}  {p.confidence:>6.3f}  {trust:>6.3f}  {snippet}")

    # ── suppressed by trust ───────────────────────────────────────────────────
    section("Predictions suppressed by trust (conf < 0.5 after trust)")
    raw_keys   = {(p.source, p.relation_type, p.target) for p in preds_raw}
    trust_keys = {(p.source, p.relation_type, p.target) for p in preds_trust}
    suppressed = raw_keys - trust_keys
    print(f"\n  Suppressed: {len(suppressed)} of {len(raw_keys)} raw predictions")
    by_rel: dict[str, int] = {}
    for src, rel, tgt in suppressed:
        by_rel[rel] = by_rel.get(rel, 0) + 1
    if by_rel:
        print()
        for rel in sorted(by_rel, key=lambda r: -by_rel[r]):
            trust = DEFAULT_RELATION_TRUST.get(rel, UNKNOWN_RELATION_TRUST)
            print(f"    {rel:20s}  suppressed={by_rel[rel]:4d}  trust={trust:.3f}")
    print()


if __name__ == "__main__":
    main()
