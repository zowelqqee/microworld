#!/usr/bin/env python3
"""Run the router over the labelled pilot cases, compute misrouting by boundary
type, and sweep the margin threshold for calibration. Throwaway.

Usage:  PYTHONPATH=. python3 artifacts/routing_v1/run_pilot.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from router import BranchRouter  # local import (run from this dir or with sys.path)

HERE = Path(__file__).resolve().parent


def evaluate(router):
    cases = json.loads((HERE / "pilot_cases.json").read_text(encoding="utf-8"))["cases"]
    rows, by_boundary = [], defaultdict(lambda: {"n": 0, "wrong": 0})
    for c in cases:
        r = router.route(c["q"])
        correct = r.branch == c["gold"]
        by_boundary[c["boundary"]]["n"] += 1
        if not correct:
            by_boundary[c["boundary"]]["wrong"] += 1
        rows.append({
            "q": c["q"], "gold": c["gold"], "routed": r.branch, "method": r.method,
            "similarity": round(r.similarity, 3), "margin": round(r.margin, 3),
            "correct": correct, "boundary": c["boundary"], "detail": r.detail,
        })
    clear = [r for r in rows if r["boundary"] == "clear"]
    ambiguous = [r for r in rows if r["boundary"] != "clear"]
    summary = {
        "threshold": router.threshold, "margin": router.margin,
        "n_total": len(rows),
        "overall_misroute_pct": round(100 * sum(not r["correct"] for r in rows) / len(rows), 1),
        "clear_misroute_pct": round(100 * sum(not r["correct"] for r in clear) / len(clear), 1) if clear else 0.0,
        "ambiguous_misroute_pct": round(100 * sum(not r["correct"] for r in ambiguous) / len(ambiguous), 1) if ambiguous else 0.0,
        "by_boundary": {b: {**v, "misroute_pct": round(100 * v["wrong"] / v["n"], 1)} for b, v in by_boundary.items()},
    }
    return summary, rows


def main():
    router = BranchRouter()
    router.build()
    summary, rows = evaluate(router)

    # Calibration sweep over margin (threshold fixed at inherited 0.85, then also
    # try a lower absolute threshold since branch phrases may sit lower than
    # predicate phrases).
    sweep = []
    for thr in (0.80, 0.85, 0.88):
        for mgn in (0.02, 0.03, 0.04, 0.05, 0.06):
            r = BranchRouter(threshold=thr, margin=mgn)
            r.build()
            s, _ = evaluate(r)
            sweep.append({"threshold": thr, "margin": mgn,
                          "overall": s["overall_misroute_pct"],
                          "clear": s["clear_misroute_pct"],
                          "ambiguous": s["ambiguous_misroute_pct"]})

    out = {"default_summary": summary, "rows": rows, "calibration_sweep": sweep}
    (HERE / "pilot_results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("DEFAULT (thr=0.85, margin=0.04):")
    print(json.dumps(summary, indent=2))
    print("\nMisrouted cases (default):")
    for r in rows:
        if not r["correct"]:
            print(f'  [{r["boundary"]}] "{r["q"]}"  gold={r["gold"]} -> {r["routed"]} ({r["method"]}, sim={r["similarity"]}, margin={r["margin"]})')
    print("\nCalibration sweep (misroute %):")
    print(f'  {"thr":>5} {"margin":>6} {"overall":>8} {"clear":>7} {"ambig":>7}')
    for s in sweep:
        print(f'  {s["threshold"]:>5} {s["margin"]:>6} {s["overall"]:>8} {s["clear"]:>7} {s["ambiguous"]:>7}')


if __name__ == "__main__":
    main()
