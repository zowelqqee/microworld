"""Compare Microworld continuation outputs against labeled GPT-2 audit rows."""

from __future__ import annotations

import argparse
import csv
import json

from worldpgt.baselines.gpt2.summarize_gpt2_audit import summarize_rows as summarize_gpt2_rows
from worldpgt.experiments.risk_coverage_metrics import summarize_rows as summarize_microworld_rows


def _read_by_id(path: str) -> dict[str, dict]:
    with open(path, "r", newline="", encoding="utf-8") as handle:
        return {row["id"]: row for row in csv.DictReader(handle)}


def _microworld_section(rows: list[dict]) -> dict:
    metrics = summarize_microworld_rows(rows)
    return {
        "total": metrics["total"],
        "continue_count": metrics["continue_count"],
        "audit_count": metrics["audit_count"],
        "coverage_rate": metrics["coverage_rate"],
        "wrong_continue_count": metrics["wrong_continue_count"],
        "precision_on_continued": metrics["precision_on_continued"],
        "answerable_recall": metrics["answerable_recall"],
    }


def _gpt2_section(rows: list[dict]) -> dict:
    metrics = summarize_gpt2_rows(rows)
    return {
        "total": metrics["total"],
        "good": metrics["good"],
        "bad": metrics["bad"],
        "unclear": metrics["unclear"],
        "precision": metrics["precision"],
        "correct_sense_count": metrics["correct_sense_count"],
        "wrong_sense_count": metrics["wrong_sense_count"],
        "no_sense_count": metrics["no_sense_count"],
        "correct_sense_rate": metrics["correct_sense_rate"],
        "wrong_sense_rate": metrics["wrong_sense_rate"],
    }


def compare(microworld_path: str, gpt2_audit_path: str) -> dict:
    microworld = _read_by_id(microworld_path)
    gpt2 = _read_by_id(gpt2_audit_path)
    ids = [row_id for row_id in microworld if row_id in gpt2]

    head = {
        "microworld_continue_gpt2_good": 0,
        "microworld_continue_gpt2_bad": 0,
        "microworld_audit_gpt2_good": 0,
        "microworld_audit_gpt2_bad": 0,
        "both_safe_or_correct": 0,
        "microworld_safe_gpt2_bad": 0,
        "gpt2_good_microworld_audit": 0,
        "disagreements": [],
    }

    for row_id in ids:
        mw = microworld[row_id]
        gpt = gpt2[row_id]
        mw_decision = mw.get("decision", "")
        label = gpt.get("label", "")
        mw_safe = mw_decision == "audit" or (
            mw_decision == "continue"
            and mw.get("expected_sense", "").strip()
            and mw.get("selected_sense", "").strip() == mw.get("expected_sense", "").strip()
        )

        if mw_decision == "continue" and label == "good":
            head["microworld_continue_gpt2_good"] += 1
        if mw_decision == "continue" and label == "bad":
            head["microworld_continue_gpt2_bad"] += 1
        if mw_decision == "audit" and label == "good":
            head["microworld_audit_gpt2_good"] += 1
        if mw_decision == "audit" and label == "bad":
            head["microworld_audit_gpt2_bad"] += 1
        if mw_safe and label in {"good", "unclear"}:
            head["both_safe_or_correct"] += 1
        if mw_safe and label == "bad":
            head["microworld_safe_gpt2_bad"] += 1
        if label == "good" and mw_decision == "audit":
            head["gpt2_good_microworld_audit"] += 1

        if (mw_decision == "audit" and label == "good") or (
            mw_decision == "continue" and label == "bad"
        ):
            head["disagreements"].append(
                {
                    "id": row_id,
                    "prompt": mw.get("prompt", ""),
                    "difficulty_type": mw.get("difficulty_type", ""),
                    "expected_sense": mw.get("expected_sense", ""),
                    "microworld_decision": mw_decision,
                    "microworld_selected_sense": mw.get("selected_sense", ""),
                    "gpt2_label": label,
                    "gpt2_judged_sense": gpt.get("judged_sense", ""),
                    "gpt2_judged_text": gpt.get("judged_text", ""),
                }
            )

    return {
        "microworld": _microworld_section([microworld[row_id] for row_id in ids]),
        "gpt2": _gpt2_section([gpt2[row_id] for row_id in ids]),
        "head_to_head": head,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Microworld and labeled GPT-2 audit outputs.")
    parser.add_argument("--microworld", required=True, help="Microworld output CSV")
    parser.add_argument("--gpt2-audit", required=True, help="Labeled GPT-2 audit CSV")
    parser.add_argument("--output", required=True, help="Output comparison JSON")
    args = parser.parse_args()

    summary = compare(args.microworld, args.gpt2_audit)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
