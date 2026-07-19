#!/usr/bin/env python3
"""Reflective-reasoning v2 EXTENSION gate pilot — throwaway enumerator.

Tests two NEW, explicitly-named composition patterns beyond the two proven ones
(founding-counterfactual, 2-hop why-might). Same discipline as all prior pilots:
real edges, NAIVE version first, then look for a structural filter, then honest
per-case defensibility judgment.

Why THESE two (data-driven, not "try more"):
  * Deep entity-chains are sparse (only 15/276 objects are themselves subjects),
    and typed-containment 2-hop chains turned out identical to the existing
    why-might bridges -> 3-hop / containment is redundant, not an extension.
  * But 32 objects have >=2 incoming subjects -> rich SHARED-ATTRIBUTE structure.
    That supports two genuinely-new patterns:

  Pattern A  CO-ATTRIBUTION: X and Y both --pred--> O  =>  "X and Y might be
             related; both <pred> <O>." (peer-linking via a shared object; NOT a
             path the existing rules build.)
  Pattern B  PROPERTY-TRANSFER (analogy): X and Y share O, and Y also --p--> Z
             (which X lacks)  =>  "X might also <p> <Z>." (classic analogical
             leap; expected to be mostly non-sequiturs — testing it to reject or
             keep it honestly.)

Run:  PYTHONPATH=. python3 artifacts/reflective_reasoning_core_v2/pilot_extension_enumerator.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parent / "compositional_grammar_v1" / "capability_overlay.json"

# Predicates that make a shared object a MEANINGFUL association (content/authorship/
# capability). Geographic co-location and valuation are excluded as too weak — this
# is the candidate structural filter for Pattern A.
CONTENT_PREDICATES = {
    "develops", "produces", "created_by", "published_by", "publishes", "uses",
    "founded", "developed_by", "runs_on", "provides", "has_topic",
}
WEAK_SHARED_PREDICATES = {"located_in", "estimated_net_worth", "leader_of", "known_for"}


def load():
    data = json.loads(OVERLAY.read_text(encoding="utf-8"))
    return [x for x in data if isinstance(x, dict) and x.get("overlay_type") == "overlay_relation"]


def main():
    rel = load()
    # object -> list of (subject, predicate)
    byobj = defaultdict(list)
    adj = defaultdict(list)  # subject -> (predicate, object)
    for r in rel:
        byobj[r["object"].strip().lower()].append((r["subject"], r["predicate"], r["object"]))
        adj[r["subject"].strip().lower()].append((r["predicate"], r["object"]))

    # ---- Pattern A: co-attribution pairs ---------------------------------- #
    a_naive, a_filtered = [], []
    for o_norm, incoming in byobj.items():
        if len(incoming) < 2:
            continue
        for (s1, p1, obj1), (s2, p2, obj2) in combinations(incoming, 2):
            if s1.strip().lower() == s2.strip().lower():
                continue
            rec = {
                "x": s1, "y": s2, "shared_object": obj1,
                "x_pred": p1, "y_pred": p2,
                "same_predicate": p1.strip().lower() == p2.strip().lower(),
                "content_predicate": p1.strip().lower() in CONTENT_PREDICATES,
            }
            a_naive.append(rec)
            # filter: both reach O via the SAME content predicate.
            if rec["same_predicate"] and rec["content_predicate"]:
                a_filtered.append(rec)

    # ---- Pattern B: property transfer via a shared-attribute link --------- #
    b_naive = []
    for o_norm, incoming in byobj.items():
        if len(incoming) < 2:
            continue
        subs = {s.strip().lower(): (s, p) for s, p, _ in incoming}
        for xn, (x, xp) in subs.items():
            for yn, (y, yp) in subs.items():
                if xn == yn:
                    continue
                x_objs = {o.strip().lower() for _p, o in adj[xn]}
                for p, z in adj[yn]:
                    if z.strip().lower() in x_objs or z.strip().lower() == o_norm:
                        continue  # X already has it, or it's the shared object
                    b_naive.append({
                        "x": x, "y": y, "shared_object": incoming[0][2],
                        "transferred_predicate": p, "transferred_object": z,
                    })

    out = {
        "pattern_A_co_attribution": {
            "naive_pairs": len(a_naive),
            "filtered_pairs": len(a_filtered),
            "filtered_sample": a_filtered[:40],
        },
        "pattern_B_property_transfer": {
            "naive_transfers": len(b_naive),
            "sample": b_naive[:40],
        },
    }
    (HERE / "pilot_extension_traces.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pattern A co-attribution: naive={len(a_naive)}  filtered(same content predicate)={len(a_filtered)}")
    seen = set()
    for rec in a_filtered:
        key = tuple(sorted([rec["x"].lower(), rec["y"].lower()])) + (rec["shared_object"].lower(), rec["x_pred"])
        if key in seen:
            continue
        seen.add(key)
        if len(seen) <= 20:
            print(f'   A: {rec["x"]} & {rec["y"]}  --both {rec["x_pred"]}-->  {rec["shared_object"]}')
    print(f"\nPattern B property-transfer: naive={len(b_naive)}")
    for rec in b_naive[:14]:
        print(f'   B: {rec["x"]} shares [{rec["shared_object"]}] with {rec["y"]}; {rec["y"]} {rec["transferred_predicate"]} {rec["transferred_object"]}'
              f'  => X might {rec["transferred_predicate"]} {rec["transferred_object"]}?')


if __name__ == "__main__":
    main()
