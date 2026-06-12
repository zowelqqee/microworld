"""Summarize manually labeled GPT-2 continuation audit CSVs."""

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


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _classify_sense(expected: Optional[str], judged: Optional[str]) -> str:
    if expected is None and judged is None:
        return "correct"
    if expected is not None and judged is None:
        return "no_sense"
    if expected == judged:
        return "correct"
    return "wrong"


def _empty_counts() -> dict:
    return {
        "total": 0,
        "audited_count": 0,
        "unaudited_count": 0,
        "good": 0,
        "bad": 0,
        "unclear": 0,
        "correct_sense_count": 0,
        "wrong_sense_count": 0,
        "no_sense_count": 0,
    }


def _add_row(counts: dict, row: dict) -> None:
    counts["total"] += 1
    label = (row.get("label") or "").strip()
    if label:
        counts["audited_count"] += 1
        if label in {"good", "bad", "unclear"}:
            counts[label] += 1
    else:
        counts["unaudited_count"] += 1

    classification = _classify_sense(_norm(row.get("expected_sense")), _norm(row.get("judged_sense")))
    if classification == "correct":
        counts["correct_sense_count"] += 1
    elif classification == "wrong":
        counts["wrong_sense_count"] += 1
    else:
        counts["no_sense_count"] += 1


def _finalize(counts: dict) -> dict:
    total = counts["total"]
    precision_denominator = counts["good"] + counts["bad"]
    result = dict(counts)
    result.update(
        {
            "good_rate": _rate(counts["good"], total),
            "bad_rate": _rate(counts["bad"], total),
            "unclear_rate": _rate(counts["unclear"], total),
            "precision": _rate(counts["good"], precision_denominator),
            "correct_sense_rate": _rate(counts["correct_sense_count"], total),
            "wrong_sense_rate": _rate(counts["wrong_sense_count"], total),
        }
    )
    return result


def summarize_rows(rows: list[dict]) -> dict:
    global_counts = _empty_counts()
    grouped: dict[str, dict] = {}

    for row in rows:
        _add_row(global_counts, row)
        difficulty = row.get("difficulty_type", "") or "unknown"
        grouped.setdefault(difficulty, _empty_counts())
        _add_row(grouped[difficulty], row)

    summary = _finalize(global_counts)
    summary["by_difficulty_type"] = {
        difficulty: _finalize(counts)
        for difficulty, counts in sorted(grouped.items())
    }
    return summary


def summarize_file(input_path: str) -> dict:
    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        return summarize_rows(list(csv.DictReader(handle)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GPT-2 continuation audit labels.")
    parser.add_argument("--input", required=True, help="Labeled GPT-2 audit CSV")
    parser.add_argument("--output", required=True, help="Output JSON summary")
    args = parser.parse_args()

    summary = summarize_file(args.input)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
