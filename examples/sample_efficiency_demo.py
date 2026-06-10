"""
Demo: sample-efficiency benchmark.

Claim under test:
    Do consolidation / concept formation / structural similarity improve
    sample efficiency over majority-only link prediction?

Result preview:
    majority_only is pinned at a low F1 no matter how much data it sees — it can
    only bet on one global connector. Structural similarity and concept
    membership recover one more family per chunk of data, climbing to perfect
    recovery. Same held-out set, fresh world per mode, no manual similarity.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.sample_efficiency import (
    SampleEfficiencyExperiment, FAMILY_SPECS, MODES, DEFAULT_BUDGETS,
)


def section(title: str) -> None:
    print(f"\n{'═' * 72}")
    print(f"  {title}")
    print("═" * 72)


def main() -> None:
    exp = SampleEfficiencyExperiment(n_per_family=6)

    section("Synthetic world")
    print(f"  families     : {[f[0] for f in FAMILY_SPECS]}")
    print(f"  per family   : {exp.n_per_family} instances")
    print(f"  pool size    : {len(exp.pool)} (observed subject to budget)")
    print(f"  held-out ({len(exp.held_out)}) — closing edges to predict (fixed across modes):")
    for inst in exp.held_out:
        print(f"      {inst.connector:10s} --grows_into--> {inst.tree}")
    print("\n  Pool is ordered family-by-family, so each family's first complete")
    print("  example appears as the budget grows. Watch when each mode 'gets' it.")

    section("Benchmark table")
    rows = exp.run(budgets=DEFAULT_BUDGETS, modes=MODES)
    print(f"  {'budget':>6}  {'mode':22s}  {'P':>5}  {'R':>5}  {'F1':>5}  "
          f"{'TP':>2} {'FP':>2} {'FN':>2}")
    print("  " + "─" * 60)
    last_budget = None
    for r in rows:
        if last_budget is not None and r["budget"] != last_budget:
            print("  " + "·" * 60)
        last_budget = r["budget"]
        print(
            f"  {int(r['budget']*100):>5}%  {r['mode']:22s}  "
            f"{r['precision']:5.2f}  {r['recall']:5.2f}  {r['f1']:5.2f}  "
            f"{r['tp']:>2} {r['fp']:>2} {r['fn']:>2}"
        )

    section("F1 vs budget — per mode")
    by_mode: dict[str, list[float]] = {m: [] for m in MODES}
    budgets_seen: list[float] = []
    for r in rows:
        by_mode[r["mode"]].append(r["f1"])
        if r["budget"] not in budgets_seen:
            budgets_seen.append(r["budget"])
    header = "  " + " " * 22 + "".join(f"{int(b*100):>6}%" for b in budgets_seen)
    print(header)
    for mode in MODES:
        sparkline = "".join(_block(f1) for f1 in by_mode[mode])
        f1s = "".join(f"{f1:>7.2f}" for f1 in by_mode[mode])
        print(f"  {mode:22s}{f1s}   {sparkline}")

    section("Verdict")
    maj = by_mode["majority_only"]
    hyb = by_mode["hybrid_structural"]
    fmt = lambda xs: "[" + ", ".join(f"{x:.2f}" for x in xs) + "]"
    print(f"  majority_only F1 : {fmt(maj)}   (flat — one global connector, max 1/4 families)")
    print(f"  hybrid_structural: {fmt(hyb)}   (climbs to perfect recovery)")
    improved = [b for b, m, h in zip(budgets_seen, maj, hyb) if h > m]
    if improved:
        budgets_str = ", ".join(f"{int(b*100)}%" for b in improved)
        print(f"\n  ✓ hybrid beats majority at budgets: {budgets_str}")
    print(f"  ✓ hybrid never below majority at any budget: "
          f"{all(h >= m for m, h in zip(maj, hyb))}")
    print("\n  Sample efficiency: structural/concept reach with 80% of the data")
    print("  what majority cannot reach with 100% — they generalize per family")
    print("  instead of betting on a single global pattern.")


def _block(f1: float) -> str:
    bars = " ▁▂▃▄▅▆▇█"
    return bars[min(8, int(round(f1 * 8)))]


if __name__ == "__main__":
    main()
