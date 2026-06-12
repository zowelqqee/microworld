"""Summarize risk/coverage metrics for controlled continuation outputs.

Usage:
    python3 -m worldpgt.experiments.summarize_risk_coverage \
        --input worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
        --output worldpgt/experiments/microworld_continuation_v1_2_risk_coverage.json
"""

from __future__ import annotations

import argparse
import csv
import json

from worldpgt.experiments.risk_coverage_metrics import summarize_rows


def summarize_file(input_path: str) -> dict:
    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        return summarize_rows(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize risk/coverage metrics as JSON.")
    parser.add_argument("--input", required=True, help="Input controlled continuation output CSV")
    parser.add_argument("--output", required=True, help="Output JSON path")
    args = parser.parse_args()

    summary = summarize_file(args.input)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.write("\n")


if __name__ == "__main__":
    main()
