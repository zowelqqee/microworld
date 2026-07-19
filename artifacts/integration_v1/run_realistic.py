#!/usr/bin/env python3
"""Run the realistic case set through the integrated router; record routing +
support_kind + answer for manual review; compute misrouting vs gold_branch.

'reflective' gold is satisfied by output branch in {reflective, reflective_extended}.
Usage:  PYTHONPATH=. python3 artifacts/integration_v1/run_realistic.py
"""

from __future__ import annotations

import json
import argparse
from collections import Counter
from pathlib import Path

from worldpgt.reasoning.integrated_answer_router import IntegratedAnswerRouter

HERE = Path(__file__).resolve().parent

# output branch -> gold family it satisfies
FAMILY = {
    "qa": "qa", "qa_safety": "qa",
    "reflective": "reflective", "reflective_extended": "reflective",
    "constrained_creative": "constrained_creative",
    "pure_creative": "pure_creative",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0, help="zero-based first case")
    parser.add_argument("--stop", type=int, default=None, help="exclusive final case")
    parser.add_argument("--output", type=Path, default=HERE / "realistic_results.json")
    args = parser.parse_args()
    cases = json.loads((HERE / "realistic_cases.json").read_text(encoding="utf-8"))["cases"]
    all_cases = cases
    cases = cases[args.start:args.stop]
    router = IntegratedAnswerRouter(overlay_mode="promoted")

    rows, wrong = [], []
    conf_counts = Counter()
    for c in cases:
        a = router.answer(c["q"])
        family = FAMILY.get(a.branch, a.branch)
        # A conservative QA fallback on a reflective/creative question counts as a
        # misroute for accounting, but is flagged as fail-safe (never the reverse).
        correct = family == c["gold_branch"]
        conf_counts[a.confidence_level] += 1
        row = {
            "q": c["q"], "gold_branch": c["gold_branch"], "out_branch": a.branch,
            "route_method": a.route_method, "confidence_level": a.confidence_level,
            "support_kind": a.support_kind, "decision": a.decision,
            "answer_text": a.answer_text, "caution": a.caution, "correct": correct,
        }
        rows.append(row)
        if not correct:
            fail_safe = a.branch in ("qa", "qa_safety")
            wrong.append({**{k: row[k] for k in ("q", "gold_branch", "out_branch", "route_method",
                                                 "confidence_level")}, "fail_safe_to_qa": fail_safe})

    summary = {
        "n": len(cases),
        "case_range": [args.start, args.start + len(cases)],
        "total_available_cases": len(all_cases),
        "misroute_pct": round(100 * len(wrong) / len(cases), 1),
        "n_misrouted": len(wrong),
        "misrouted": wrong,
        "confidence_level_distribution": dict(conf_counts),
        "route_method_distribution": dict(Counter(r["route_method"] for r in rows)),
        "branch_distribution": dict(Counter(r["out_branch"] for r in rows)),
    }
    args.output.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
