"""Compare Microworld continuation v1 outputs against v1.1 outputs.

Usage:
    python3 -m worldpgt.experiments.compare_v1_to_v1_1 \
        --before worldpgt/experiments/microworld_continuation_v1_outputs.csv \
        --after worldpgt/experiments/microworld_continuation_v1_1_outputs.csv \
        --output worldpgt/experiments/v1_vs_v1_1_comparison.json
"""

from __future__ import annotations

import argparse
import csv
import json
from typing import Optional


def _norm(value) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _classify(expected: Optional[str], selected: Optional[str], decision: str) -> str:
    if expected is None and selected is None:
        return "correct"
    if expected is not None and decision == "audit":
        return "no_sense"
    if expected is not None and selected is None:
        return "no_sense"
    if expected == selected:
        return "correct"
    return "wrong"


def _read_rows(path: str) -> dict[str, dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _count(rows: dict[str, dict]) -> dict[str, int]:
    counts = {
        "correct_sense_count": 0,
        "wrong_sense_count": 0,
        "audit_count": 0,
    }
    for row in rows.values():
        expected = _norm(row.get("expected_sense"))
        selected = _norm(row.get("selected_sense"))
        decision = row.get("decision", "")
        classification = _classify(expected, selected, decision)
        if classification == "correct":
            counts["correct_sense_count"] += 1
        elif classification == "wrong":
            counts["wrong_sense_count"] += 1
        if decision == "audit":
            counts["audit_count"] += 1
    return counts


def compare(before_path: str, after_path: str) -> dict:
    before_rows = _read_rows(before_path)
    after_rows = _read_rows(after_path)

    before_counts = _count(before_rows)
    after_counts = _count(after_rows)
    summary = {
        "wrong_sense_count_before": before_counts["wrong_sense_count"],
        "wrong_sense_count_after": after_counts["wrong_sense_count"],
        "correct_sense_count_before": before_counts["correct_sense_count"],
        "correct_sense_count_after": after_counts["correct_sense_count"],
        "audit_count_before": before_counts["audit_count"],
        "audit_count_after": after_counts["audit_count"],
        "rows_changed": 0,
        "wrong_to_audit": 0,
        "wrong_to_correct": 0,
        "audit_to_correct": 0,
        "correct_to_wrong": 0,
        "correct_to_audit": 0,
        "changed_rows": [],
    }

    for row_id in sorted(before_rows):
        before = before_rows[row_id]
        after = after_rows.get(row_id)
        if after is None:
            continue

        expected = _norm(before.get("expected_sense"))
        before_selected = _norm(before.get("selected_sense"))
        after_selected = _norm(after.get("selected_sense"))
        before_decision = before.get("decision", "")
        after_decision = after.get("decision", "")
        before_class = _classify(expected, before_selected, before_decision)
        after_class = _classify(expected, after_selected, after_decision)

        if before_selected == after_selected and before_decision == after_decision:
            continue

        summary["rows_changed"] += 1
        if before_class == "wrong" and after_class == "no_sense":
            summary["wrong_to_audit"] += 1
        if before_class == "wrong" and after_class == "correct":
            summary["wrong_to_correct"] += 1
        if before_decision == "audit" and after_class == "correct":
            summary["audit_to_correct"] += 1
        if before_class == "correct" and after_class == "wrong":
            summary["correct_to_wrong"] += 1
        if before_class == "correct" and after_class == "no_sense":
            summary["correct_to_audit"] += 1

        summary["changed_rows"].append(
            {
                "id": row_id,
                "prompt": before.get("prompt", ""),
                "difficulty_type": before.get("difficulty_type", ""),
                "expected_sense": before.get("expected_sense", ""),
                "before_selected_sense": before.get("selected_sense", ""),
                "after_selected_sense": after.get("selected_sense", ""),
                "before_decision": before_decision,
                "after_decision": after_decision,
                "before_reasons": before.get("reasons", ""),
                "after_reasons": after.get("reasons", ""),
            }
        )

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare v1 and v1.1 Microworld continuation outputs.")
    parser.add_argument("--before", required=True, help="v1 output CSV")
    parser.add_argument("--after", required=True, help="v1.1 output CSV")
    parser.add_argument("--output", required=True, help="Output JSON comparison path")
    args = parser.parse_args()

    summary = compare(args.before, args.after)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote comparison to {args.output}")


if __name__ == "__main__":
    main()
