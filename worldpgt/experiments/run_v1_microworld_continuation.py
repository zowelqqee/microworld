"""Run the v1 Microworld controlled continuation benchmark.

Usage:
    python -m worldpgt.experiments.run_v1_microworld_continuation \
        --input worldpgt/experiments/continuation_prompts_v1.csv \
        --output worldpgt/experiments/microworld_continuation_v1_outputs.csv
"""

from __future__ import annotations

import argparse
import csv

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine

OUTPUT_FIELDS = [
    "id",
    "prompt",
    "ambiguous_term",
    "expected_sense",
    "difficulty_type",
    "notes",
    "continuation",
    "selected_sense",
    "confidence",
    "decision",
    "reasons",
    "memory_hits",
]


def run(input_path: str, output_path: str) -> list[dict]:
    engine = ControlledContinuationEngine()
    output_rows: list[dict] = []

    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            prompt = record.get("prompt", "")
            result = engine.continue_prompt(prompt)
            output_rows.append(
                {
                    "id": record.get("id", ""),
                    "prompt": prompt,
                    "ambiguous_term": record.get("ambiguous_term", "") or result.ambiguous_term or "",
                    "expected_sense": record.get("expected_sense", "") or "",
                    "difficulty_type": record.get("difficulty_type", ""),
                    "notes": record.get("notes", ""),
                    "continuation": result.continuation,
                    "selected_sense": result.selected_sense or "",
                    "confidence": f"{result.confidence:.4f}",
                    "decision": result.decision,
                    "reasons": " | ".join(result.reasons),
                    "memory_hits": " | ".join(result.memory_hits),
                }
            )

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    return output_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Microworld controlled continuation v1 over a prompt CSV.")
    parser.add_argument("--input", required=True, help="Input v1 continuation prompt CSV")
    parser.add_argument("--output", required=True, help="Output CSV path for v1 continuation results")
    args = parser.parse_args()

    rows = run(args.input, args.output)
    decisions: dict[str, int] = {}
    for row in rows:
        decisions[row["decision"]] = decisions.get(row["decision"], 0) + 1
    print(f"Wrote {len(rows)} rows to {args.output}")
    print(f"Decisions: {decisions}")


if __name__ == "__main__":
    main()
