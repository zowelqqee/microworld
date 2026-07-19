#!/usr/bin/env python3
"""Abduction pilot scoring pass (throwaway). Reads pilot_abduction_traces.json,
classifies each case by the STRUCTURAL FILTER (2-hop bridge), writes
pilot_abduction_scored.json.

Routing per case:
  * direct     -> a length-1 path (direct edge) exists: NOT speculative, this is a
                  grounded answer; the abduction rule should not fire.
  * fire       -> at least one length-2 bridge path S -> M -> O exists: emit a
                  defensible speculative explanation via intermediate M.
  * decline_spurious -> no 2-hop bridge, only length-3 paths (typically routed
                  through a shared person). Treated as too weak -> decline.
  * decline_nobridge -> no connecting path at all -> decline.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def path_len(p):
    return len(p)


def main():
    traces = json.loads((HERE / "pilot_abduction_traces.json").read_text(encoding="utf-8"))
    scored = []
    for c in traces["cases"]:
        two_hop = [p for p in c["paths_2hop"] if path_len(p) == 2]
        direct = [p for p in c["paths_2hop"] if path_len(p) == 1]
        only3 = c["paths_3hop_only"]
        if direct:
            route = "direct_grounded"
        elif two_hop:
            route = "fire_speculative"
        elif only3:
            route = "decline_spurious_3hop_only"
        else:
            route = "decline_no_bridge"
        # bridge nodes for firing cases (the M in S->M->O)
        bridges = sorted({p[0].split(" | ")[2].split("  [")[0] for p in two_hop})
        scored.append(
            {
                "question": c["question"],
                "route": route,
                "n_naive": c["n_naive"],
                "n_2hop_bridges": len(two_hop),
                "bridge_nodes": bridges,
                "example_2hop": two_hop[0] if two_hop else None,
                "n_3hop_only": len(only3),
                "example_3hop": only3[0] if only3 else None,
            }
        )

    routes = {}
    for s in scored:
        routes[s["route"]] = routes.get(s["route"], 0) + 1
    summary = {
        "n_cases": len(scored),
        "routes": routes,
        "total_naive_explanations": sum(s["n_naive"] for s in scored),
        "fire_cases": routes.get("fire_speculative", 0),
    }
    out = {"summary": summary, "scored": scored}
    (HERE / "pilot_abduction_scored.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("SUMMARY:", json.dumps(summary, indent=2))
    print()
    for s in scored:
        print(f'  [{s["route"]:<26}] {s["question"]}')
        if s["example_2hop"]:
            print("        via:", " -> ".join(x.split("  [")[0] for x in s["example_2hop"]))
        elif s["example_3hop"]:
            print("        (spurious 3-hop):", " -> ".join(x.split("  [")[0] for x in s["example_3hop"]))


if __name__ == "__main__":
    main()
