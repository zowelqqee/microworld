"""Flag obvious deterministic continuation realization problems.

Usage:
    python3 -m worldpgt.experiments.check_realization_quality \
        --input worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
        --output worldpgt/experiments/microworld_continuation_v1_2_realization_quality.json
"""

from __future__ import annotations

import argparse
import csv
import json

from worldpgt.continuation.surface_validator import validate_surface_text


def check_rows(rows: list[dict]) -> dict:
    flagged_rows = []
    total_continued = 0

    for row in rows:
        if row.get("decision") != "continue":
            continue
        total_continued += 1
        continuation = row.get("continuation", "")
        validation = validate_surface_text(row.get("prompt", ""), continuation)
        flags = validation.matched_patterns
        if flags:
            flagged_rows.append(
                {
                    "id": row.get("id", ""),
                    "prompt": row.get("prompt", ""),
                    "selected_sense": row.get("selected_sense", ""),
                    "flags": flags,
                    "continuation": continuation,
                }
            )

    flagged_count = len(flagged_rows)
    return {
        "total_continued": total_continued,
        "flagged_count": flagged_count,
        "flagged_rate": round(flagged_count / total_continued, 4) if total_continued else 0.0,
        "flagged_rows": flagged_rows,
    }


def check_file(input_path: str) -> dict:
    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        return check_rows(list(csv.DictReader(handle)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check continuation realization quality.")
    parser.add_argument("--input", required=True, help="Input controlled continuation output CSV")
    parser.add_argument("--output", required=True, help="Output JSON quality report")
    args = parser.parse_args()

    summary = check_file(args.input)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
