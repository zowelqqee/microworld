#!/usr/bin/env python3
"""Reflective-reasoning-core GATE PILOT — throwaway enumerator, not production.

Implements the ONE inference rule design.md section 2 proposes — "counterfactual
removal" — over the real accepted graph, so the premises->rule->conclusion traces
are constructed from actual edges, not invented. It does NOT judge defensibility;
that is the manual step recorded in pilot_report.md.

Rule (verbatim from design.md sec 2):
  Given grounded fact F(subject, predicate, object), a "what if subject had not
  predicate object" question builds a speculative_step by finding OTHER explicit,
  graph-connected facts that share `subject` or `object` as a node, and reasoning
  that those connected facts would be the ones affected. Graph-connected only.

This script emits, per case: F, and the candidate "affected" facts split by how
they connect (co-subject / object-as-subject / shared-node). It also tags each
candidate with whether F's predicate is DEPENDENCY-BEARING (existence/creation)
vs INCIDENTAL, so the report can measure the structural filter design.md sec 6
anticipated.

Run:  python3 pilot_enumerator.py   -> writes pilot_traces.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
OVERLAY = HERE.parent / "compositional_grammar_v1" / "capability_overlay.json"

# Predicates whose removal plausibly threatens the *existence* of the object (or
# the relation is constitutive), so downstream facts about the object can defensibly
# be "in question". This is the structural filter candidate from design.md sec 6.
DEPENDENCY_BEARING = {
    "founded", "founded_by", "created_by", "developed_by", "develops",
    "product_of", "construction_started", "produces", "publishes", "published_by",
}
# Predicates that are incidental / would persist regardless of the removed fact
# (location, valuation, naming, leadership title, fame). Listed for transparency;
# anything not in DEPENDENCY_BEARING is treated as incidental.

# 13 hand-picked focal facts spanning both predicate classes so the filter's
# effect is visible. (subject, predicate, object) must exist in the overlay.
FOCAL_FACTS = [
    ("Elon Musk", "founded", "SpaceX"),
    ("Elon Musk", "leader_of", "Tesla"),
    ("Elon Musk", "known_for", "SpaceX"),
    ("Elon Musk", "estimated_net_worth", "US$1.1 trillion"),
    ("Tesla", "produces", "Electric cars"),
    ("SpaceX", "located_in", "Starbase, Texas"),
    ("SpaceX", "develops", "Rockets"),
    ("SpaceX", "produces", "Falcon rockets"),
    ("Jeff Bezos", "founded", "Blue Origin"),
    ("Blue Origin", "develops", "Rockets"),
    ("LVMH", "located_in", "Paris"),
    ("Falcon 9", "located_in", "McGregor, Texas"),
    ("Neuralink", "develops", "brain-computer interfaces"),
]


def load_graph():
    data = json.loads(OVERLAY.read_text(encoding="utf-8"))
    rel = [
        x for x in data
        if isinstance(x, dict) and x.get("overlay_type") == "overlay_relation"
    ]
    edges = []
    for i, r in enumerate(rel):
        edges.append(
            {
                "evidence_id": r.get("evidence_id") or r.get("id") or f"edge_{i}",
                "subject": r["subject"],
                "predicate": r["predicate"],
                "object": r["object"],
                "s_norm": r["subject"].strip().lower(),
                "o_norm": str(r["object"]).strip().lower(),
            }
        )
    return edges


def find_edge(edges, subj, pred, obj):
    sl, pl, ol = subj.lower(), pred.lower(), str(obj).lower()
    for e in edges:
        if e["s_norm"] == sl and e["predicate"].lower() == pl and e["o_norm"] == ol:
            return e
    return None


def counterfactual_removal(edges, focal):
    """Return the trace for 'what if <s> had not <p> <o>'."""
    s_norm, o_norm = focal["s_norm"], focal["o_norm"]
    fid = focal["evidence_id"]
    co_subject, object_as_subject, shared_node = [], [], []
    for e in edges:
        if e["evidence_id"] == fid:
            continue
        if e["s_norm"] == s_norm:
            co_subject.append(e)                 # other facts about the same subject
        elif e["s_norm"] == o_norm:
            object_as_subject.append(e)          # facts the object is itself the subject of
        elif e["o_norm"] == s_norm or e["o_norm"] == o_norm:
            shared_node.append(e)                # facts pointing at either node
    return {
        "co_subject": co_subject,
        "object_as_subject": object_as_subject,
        "shared_node": shared_node,
    }


def brief(e):
    return f'{e["subject"]} | {e["predicate"]} | {e["object"]}  [{e["evidence_id"]}]'


def main():
    edges = load_graph()
    cases = []
    for (subj, pred, obj) in FOCAL_FACTS:
        focal = find_edge(edges, subj, pred, obj)
        if focal is None:
            cases.append({"focal": f"{subj}|{pred}|{obj}", "error": "NOT FOUND in overlay"})
            continue
        conn = counterfactual_removal(edges, focal)
        # Candidate affected facts = all connected facts (naive rule, no filter).
        candidates = conn["co_subject"] + conn["object_as_subject"] + conn["shared_node"]
        cases.append(
            {
                "question": f"What if {subj} had not {pred} {obj}?",
                "focal_fact": brief(focal),
                "focal_predicate": pred,
                "focal_predicate_dependency_bearing": pred in DEPENDENCY_BEARING,
                "n_candidates_naive": len(candidates),
                "candidates": {
                    "co_subject": [brief(e) for e in conn["co_subject"]],
                    "object_as_subject": [brief(e) for e in conn["object_as_subject"]],
                    "shared_node": [brief(e) for e in conn["shared_node"]],
                },
            }
        )
    out = {
        "overlay": str(OVERLAY.relative_to(HERE.parent.parent)),
        "rule": "counterfactual_removal",
        "dependency_bearing_predicates": sorted(DEPENDENCY_BEARING),
        "cases": cases,
    }
    (HERE / "pilot_traces.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    for c in cases:
        if "error" in c:
            print("MISSING:", c["focal"], c["error"])
            continue
        dep = "DEP" if c["focal_predicate_dependency_bearing"] else "inc"
        print(f'\n[{dep}] {c["question"]}')
        print(f'   F: {c["focal_fact"]}')
        for bucket, items in c["candidates"].items():
            for it in items:
                print(f'     ({bucket}) {it}')


if __name__ == "__main__":
    main()
