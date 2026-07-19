#!/usr/bin/env python3
"""Reflective-reasoning-core GATE PILOT — scoring pass (throwaway).

Reads pilot_traces.json (naive counterfactual-removal candidates over the real
graph) and applies the STRUCTURAL FILTER design.md sec 6 anticipated, so we can
measure whether a stated criterion removes the absurd conclusions.

The filter was REFINED BY THIS PILOT (see pilot_report.md): the initial
DEPENDENCY_BEARING set in the enumerator was too broad. The data shows only
*existence-conferring* predicates whose object is itself a graph entity yield a
defensible "in question" set. Everything else should DECLINE (audit), not
emit a speculative conclusion.

Admit candidate G under counterfactual removal of F(S, P, O) iff ALL:
  1. P is existence-conferring (founding/creation of a persistent entity), AND
  2. O is itself an entity (appears as the SUBJECT of >=1 other graph fact), AND
  3. G references node O (O is G's subject or object) -> G is a fact about the
     thing that would not exist.

Manual defensibility judgment (recorded here, not computed) is applied on top:
every ADMITTED candidate is checked to be non-absurd; the REJECTED pool is
sampled to confirm it is dominated by non-sequiturs.

Run:  python3 pilot_score.py   -> writes pilot_scored.json
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXISTENCE_CONFERRING = {
    "founded", "founded_by", "created_by", "developed_by",
    "product_of", "construction_started",
}


def parse(brief: str):
    # "Subject | predicate | object  [evidence_id]"
    head = brief.rsplit("  [", 1)[0]
    s, p, o = [part.strip() for part in head.split(" | ")]
    return s, p, o


def main():
    traces = json.loads((HERE / "pilot_traces.json").read_text(encoding="utf-8"))
    cases = [c for c in traces["cases"] if "error" not in c]

    # Build the set of nodes that are the subject of >=1 fact (entity test).
    subjects_seen: set[str] = set()
    for c in cases:
        s, _p, _o = parse(c["focal_fact"])
        subjects_seen.add(s.strip().lower())
        for bucket in c["candidates"].values():
            for b in bucket:
                gs, _gp, _go = parse(b)
                subjects_seen.add(gs.strip().lower())

    scored = []
    for c in cases:
        s, p, o = parse(c["focal_fact"])
        o_norm = o.strip().lower()
        p_norm = p.strip().lower()
        object_is_entity = o_norm in subjects_seen
        fires = (p_norm in EXISTENCE_CONFERRING) and object_is_entity

        all_candidates = []
        for bucket, items in c["candidates"].items():
            for b in items:
                all_candidates.append((bucket, b))

        admitted = []
        if fires:
            for bucket, b in all_candidates:
                gs, gp, go = parse(b)
                if gs.strip().lower() == o_norm or go.strip().lower() == o_norm:
                    admitted.append(b)

        scored.append(
            {
                "question": c["question"],
                "focal_fact": c["focal_fact"],
                "focal_predicate": p,
                "existence_conferring": p_norm in EXISTENCE_CONFERRING,
                "object_is_entity": object_is_entity,
                "rule_fires": fires,
                "n_candidates_naive": len(all_candidates),
                "n_admitted_filtered": len(admitted),
                "admitted_candidates": admitted,
            }
        )

    n = len(scored)
    fired = [s for s in scored if s["rule_fires"]]
    total_naive = sum(s["n_candidates_naive"] for s in scored)
    total_admitted = sum(s["n_admitted_filtered"] for s in scored)
    summary = {
        "n_cases": n,
        "cases_rule_fires": len(fired),
        "cases_decline": n - len(fired),
        "total_naive_candidates": total_naive,
        "total_admitted_candidates": total_admitted,
        "naive_reduction_pct": round(100.0 * (1 - total_admitted / total_naive), 1) if total_naive else 0.0,
    }

    out = {"filter": {"existence_conferring": sorted(EXISTENCE_CONFERRING)}, "summary": summary, "scored": scored}
    (HERE / "pilot_scored.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print("SUMMARY:", json.dumps(summary, indent=2))
    print("\nPer-case:")
    for s in scored:
        tag = "FIRES" if s["rule_fires"] else "decline"
        print(f'  [{tag}] {s["question"]}  (naive {s["n_candidates_naive"]} -> admitted {s["n_admitted_filtered"]})')
        for a in s["admitted_candidates"]:
            print(f"        + {a}")


if __name__ == "__main__":
    main()
