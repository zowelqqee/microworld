#!/usr/bin/env python3
"""Merge deterministic non-overlapping realistic-flow result chunks."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAMILY = {
    "qa": "qa", "qa_safety": "qa", "reflective": "reflective",
    "reflective_extended": "reflective", "constrained_creative": "constrained_creative",
    "pure_creative": "pure_creative",
}


def main() -> None:
    cases = json.loads((HERE / "realistic_cases.json").read_text(encoding="utf-8"))["cases"]
    rows: list[dict] = []
    for path in sorted(HERE.glob("realistic_results_chunk_*.json")):
        rows.extend(json.loads(path.read_text(encoding="utf-8"))["rows"])
    if len(rows) != len(cases):
        raise SystemExit(f"need {len(cases)} rows; found {len(rows)}")
    rows.sort(key=lambda row: cases.index({"q": row["q"], "gold_branch": row["gold_branch"]}))
    wrong = []
    for row in rows:
        if FAMILY.get(row["out_branch"], row["out_branch"]) != row["gold_branch"]:
            wrong.append({
                key: row[key] for key in (
                    "q", "gold_branch", "out_branch", "route_method", "confidence_level"
                )
            } | {"fail_safe_to_qa": row["out_branch"] in {"qa", "qa_safety"}})
    summary = {
        "n": len(rows), "misroute_pct": round(100 * len(wrong) / len(rows), 1),
        "n_misrouted": len(wrong), "misrouted": wrong,
        "confidence_level_distribution": dict(Counter(row["confidence_level"] for row in rows)),
        "route_method_distribution": dict(Counter(row["route_method"] for row in rows)),
        "branch_distribution": dict(Counter(row["out_branch"] for row in rows)),
    }
    (HERE / "realistic_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
