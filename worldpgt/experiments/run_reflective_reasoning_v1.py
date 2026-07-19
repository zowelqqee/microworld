#!/usr/bin/env python3
"""Run the reflective-reasoning rules over the real accepted graph and emit a
per-case trace artifact. Isolated experiment: reads an overlay slice, calls the
two rules, writes results. No API/server, no change to the grounded route.

Usage:
    python3 -m worldpgt.experiments.run_reflective_reasoning_v1
"""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.reasoning.reflective_reasoning_v1 import (
    load_edges,
    reflect,
    render_reflective_plan,
)

_ROOT = Path(__file__).resolve().parents[2]
_OVERLAY = _ROOT / "artifacts" / "compositional_grammar_v1" / "capability_overlay.json"
_OUT_DIR = _ROOT / "artifacts" / "reflective_reasoning_core_v1"

# The same case families the pilots validated, exercised end-to-end through the
# shipped module (parsing -> rule -> render).
QUESTIONS = [
    "What if Elon Musk had not founded SpaceX?",
    "What if Jeff Bezos had not founded Blue Origin?",
    "What if Elon Musk had not leader_of Tesla?",           # decline (incidental)
    "What if SpaceX had not develops rockets?",             # decline (generic object)
    "What if Tesla had not produces Electric cars?",        # decline
    "Why might Elon Musk be associated with rockets?",
    "Why might Elon Musk be associated with Electric cars?",
    "Why might Jeff Bezos be associated with spacecraft?",
    "Why might Gwynne Shotwell be associated with rockets?",
    "Why might Elon Musk be associated with Starbase, Texas?",
    "Why might Tesla be associated with rockets?",          # decline (spurious 3-hop)
    "Why might Elon Musk be associated with Paris?",        # decline (no bridge)
    "Why might Blue Origin be associated with rockets?",    # grounded deferral
]


def main() -> None:
    overlay = json.loads(_OVERLAY.read_text(encoding="utf-8"))
    edges = load_edges(overlay)

    cases = []
    counts: dict[str, int] = {}
    for q in QUESTIONS:
        plan = reflect(q, overlay)
        if plan is None:
            cases.append({"question": q, "decision": "unrouted"})
            counts["unrouted"] = counts.get("unrouted", 0) + 1
            continue
        counts[plan.decision] = counts.get(plan.decision, 0) + 1
        cases.append(
            {
                "question": q,
                **plan.to_dict(),
                "rendered": render_reflective_plan(plan),
            }
        )

    summary = {
        "overlay": str(_OVERLAY.relative_to(_ROOT)),
        "n_edges": len(edges),
        "n_questions": len(QUESTIONS),
        "decision_counts": counts,
    }
    out = {"summary": summary, "cases": cases}
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUT_DIR / "build_v1_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print()
    for c in cases:
        print(f'[{c["decision"]:<18}] {c["question"]}')
        if c.get("rendered") and c["decision"] == "speculative":
            print(f'      {c["rendered"]}')


if __name__ == "__main__":
    main()
