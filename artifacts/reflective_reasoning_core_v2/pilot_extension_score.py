#!/usr/bin/env python3
"""Extension pilot scoring pass (throwaway). Refines the Pattern A filter and
samples both patterns for manual defensibility judgment.

Pattern A refinement (learned from the naive run): the "same content predicate"
filter was too loose because DISTRIBUTION predicates create spurious cliques —
Oxford University Press `published_by` 12+ unrelated books, pairing them all as
"related". The refined filter keeps only KINSHIP predicates (shared capability /
authorship / creation), where a shared object implies genuine similarity, and
excludes distribution/location/valuation predicates.

Pattern B (property transfer) is sampled to confirm the expected unsoundness.
"""

from __future__ import annotations

import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parent / "compositional_grammar_v1" / "capability_overlay.json"

# Sharing an object via these implies real kinship (peers / co-creators / co-founders).
KINSHIP = {"develops", "produces", "created_by", "founded", "developed_by", "provides"}
# High-fan-out distribution/utility/location/valuation: a shared object here is a
# channel or coincidence, not a semantic link. Excluded from Pattern A.
DISTRIBUTION = {"published_by", "publishes", "located_in", "uses", "runs_on",
                "has_topic", "estimated_net_worth", "leader_of", "known_for"}


def load():
    data = json.loads(OVERLAY.read_text(encoding="utf-8"))
    return [x for x in data if isinstance(x, dict) and x.get("overlay_type") == "overlay_relation"]


def main():
    rel = load()
    byobj = defaultdict(list)
    adj = defaultdict(list)
    for r in rel:
        byobj[r["object"].strip().lower()].append((r["subject"], r["predicate"], r["object"]))
        adj[r["subject"].strip().lower()].append((r["predicate"], r["object"]))

    # Pattern A with refined KINSHIP filter.
    a_kinship, a_distribution = [], []
    for incoming in byobj.values():
        if len(incoming) < 2:
            continue
        for (s1, p1, obj1), (s2, p2, _o2) in combinations(incoming, 2):
            if s1.strip().lower() == s2.strip().lower():
                continue
            if p1.strip().lower() != p2.strip().lower():
                continue
            pred = p1.strip().lower()
            rec = {"x": s1, "y": s2, "shared_object": obj1, "predicate": p1}
            if pred in KINSHIP:
                a_kinship.append(rec)
            elif pred in DISTRIBUTION:
                a_distribution.append(rec)

    # dedupe kinship pairs
    seen, uniq = set(), []
    for r in a_kinship:
        k = tuple(sorted([r["x"].lower(), r["y"].lower()])) + (r["shared_object"].lower(), r["predicate"].lower())
        if k not in seen:
            seen.add(k); uniq.append(r)

    # Pattern B sample (property transfer)
    b = []
    for incoming in byobj.values():
        if len(incoming) < 2:
            continue
        subs = {s.strip().lower(): (s, p) for s, p, _ in incoming}
        for xn, (x, _xp) in subs.items():
            for yn, (y, _yp) in subs.items():
                if xn == yn:
                    continue
                x_objs = {o.strip().lower() for _p, o in adj[xn]}
                for p, z in adj[yn]:
                    if z.strip().lower() in x_objs:
                        continue
                    b.append({"x": x, "y": y, "transfer": f"{p} {z}"})

    out = {
        "pattern_A": {
            "kinship_pairs_deduped": len(uniq),
            "distribution_pairs_excluded": len(a_distribution),
            "kinship_pairs": uniq,
        },
        "pattern_B": {"naive_transfers": len(b), "sample": b[:30]},
    }
    (HERE / "pilot_extension_scored.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Pattern A  kinship(deduped)={len(uniq)}  distribution-excluded={len(a_distribution)}")
    for r in uniq:
        print(f'   {r["x"]} & {r["y"]}  --both {r["predicate"]}-->  {r["shared_object"]}')
    print(f"\nPattern B naive transfers={len(b)} (sample of transfer conclusions):")
    for r in b[:12]:
        print(f'   {r["x"]} might {r["transfer"]}?  (because it shares an attribute with {r["y"]})')


if __name__ == "__main__":
    main()
