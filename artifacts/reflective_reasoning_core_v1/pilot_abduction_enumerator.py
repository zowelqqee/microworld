#!/usr/bin/env python3
"""Reflective-reasoning-core GATE PILOT #2 — why-might via abduction (throwaway).

Second inference rule proposed for the reflective reasoning core. Abduction =
inference to a plausible explanation. For a "why might S be associated with O?"
question, the rule proposes graph-grounded reasons that would make the S–O
relationship plausible.

Same discipline as the counterfactual pilot:
  * real edges from capability_overlay.json,
  * NAIVE rule first (dump the 1-hop neighbourhood of S and O as "explanations"),
  * then a STRUCTURAL FILTER (require a genuine connecting PATH S -> M -> O),
  * manual defensibility judgment recorded in pilot_abduction_report.md.

The keystone question is identical in shape: does abduction produce defensible
explanations, or does it relabel graph adjacency as reasoning?

Run:  python3 pilot_abduction_enumerator.py  -> writes pilot_abduction_traces.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parent / "compositional_grammar_v1" / "capability_overlay.json"

# "Why might S be associated with O?" query pairs. Chosen to span:
#   - pairs with a clean 2-hop bridge (expected defensible),
#   - pairs bridged only through a person/shared-founder (expected spurious),
#   - pairs with no bridge at all (expected: decline).
QUERY_PAIRS = [
    ("Elon Musk", "rockets"),
    ("Elon Musk", "spacecraft"),
    ("Elon Musk", "Electric cars"),
    ("Jeff Bezos", "rockets"),
    ("Jeff Bezos", "spacecraft"),
    ("Gwynne Shotwell", "rockets"),
    ("Elon Musk", "Starbase, Texas"),
    ("Elon Musk", "Falcon rockets"),
    ("Tesla", "rockets"),            # bridged only via Musk (Tesla<-Musk->SpaceX->rockets)
    ("Neuralink", "rockets"),        # bridged only via Musk
    ("Elon Musk", "Paris"),          # no bridge expected
    ("SpaceX", "Electric cars"),     # bridged only via Musk (SpaceX<-Musk->Tesla->cars)
    ("Blue Origin", "rockets"),      # direct edge exists (Blue Origin develops rockets)
    ("Gwynne Shotwell", "spacecraft"),
]


def load_graph():
    data = json.loads(OVERLAY.read_text(encoding="utf-8"))
    rel = [x for x in data if isinstance(x, dict) and x.get("overlay_type") == "overlay_relation"]
    edges = []
    for i, r in enumerate(rel):
        edges.append(
            {
                "evidence_id": r.get("evidence_id") or r.get("id") or f"edge_{i}",
                "subject": r["subject"], "predicate": r["predicate"], "object": str(r["object"]),
                "s": r["subject"].strip().lower(), "o": str(r["object"]).strip().lower(),
            }
        )
    return edges


def brief(e):
    return f'{e["subject"]} | {e["predicate"]} | {e["object"]}  [{e["evidence_id"]}]'


def one_hop(edges, node_norm):
    """All facts where node is subject or object (the naive explanation dump)."""
    return [e for e in edges if e["s"] == node_norm or e["o"] == node_norm]


def find_paths(edges, start_norm, goal_norm, max_depth):
    """Simple undirected paths of edges from start node to goal node, up to
    max_depth edges. Returns list of edge-lists. Direction is retained in each
    edge for rendering, but traversal is undirected (an explanation can run
    either way along a relation)."""
    adj = defaultdict(list)
    for e in edges:
        adj[e["s"]].append((e["o"], e))
        adj[e["o"]].append((e["s"], e))
    results = []

    def dfs(node, target, path_edges, visited_nodes):
        if len(path_edges) > max_depth:
            return
        if node == target and path_edges:
            results.append(list(path_edges))
            return
        if len(path_edges) == max_depth:
            return
        for nxt, e in adj[node]:
            if e["evidence_id"] in {pe["evidence_id"] for pe in path_edges}:
                continue
            if nxt in visited_nodes:
                continue
            dfs(nxt, target, path_edges + [e], visited_nodes | {nxt})

    dfs(start_norm, goal_norm, [], {start_norm})
    # Deduplicate by edge-id sequence.
    uniq = []
    seen = set()
    for p in results:
        key = tuple(e["evidence_id"] for e in p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


def main():
    edges = load_graph()
    cases = []
    for (s, o) in QUERY_PAIRS:
        s_norm, o_norm = s.strip().lower(), o.strip().lower()
        naive = one_hop(edges, s_norm) + [e for e in one_hop(edges, o_norm) if e["s"] != s_norm and e["o"] != s_norm]
        paths2 = find_paths(edges, s_norm, o_norm, max_depth=2)
        paths3 = find_paths(edges, s_norm, o_norm, max_depth=3)
        # length-3 paths that are not already length-2
        only3 = [p for p in paths3 if len(p) == 3]
        cases.append(
            {
                "question": f"Why might {s} be associated with {o}?",
                "start": s, "goal": o,
                "n_naive": len(naive),
                "naive_explanations": [brief(e) for e in naive],
                "n_paths_2hop": len(paths2),
                "paths_2hop": [[brief(e) for e in p] for p in paths2],
                "n_paths_3hop_only": len(only3),
                "paths_3hop_only": [[brief(e) for e in p] for p in only3],
            }
        )
    out = {"rule": "abduction_path_explanation", "cases": cases}
    (HERE / "pilot_abduction_traces.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    for c in cases:
        print(f'\n{c["question"]}  (naive {c["n_naive"]} | 2-hop paths {c["n_paths_2hop"]} | 3-hop-only {c["n_paths_3hop_only"]})')
        for p in c["paths_2hop"]:
            print("   2-hop:", "  ->  ".join(p))
        for p in c["paths_3hop_only"][:3]:
            print("   3-hop:", "  ->  ".join(p))


if __name__ == "__main__":
    main()
