"""Sweep controlled continuation policy thresholds and report risk/coverage.

Usage:
    python3 -m worldpgt.experiments.policy_sweep \
        --input worldpgt/experiments/continuation_prompts_v1.csv \
        --output worldpgt/experiments/policy_sweep_results.csv
"""

from __future__ import annotations

import argparse
import csv
from itertools import product

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.experiments.risk_coverage_metrics import summarize_rows


MIN_SCORES = [1.0, 2.0]
MIN_MARGINS = [0.0, 0.5, 1.0, 1.5]
OUTPUT_FIELDS = [
    "config_id",
    "min_score",
    "min_margin",
    "total",
    "continue_count",
    "audit_count",
    "suppress_count",
    "correct_continue_count",
    "wrong_continue_count",
    "coverage_rate",
    "precision_on_continued",
    "answerable_recall",
    "wrong_continue_rate",
    "abstention_rate_on_answerable",
]


def _load_prompts(input_path: str) -> list[dict]:
    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _run_config(prompt_rows: list[dict], min_score: float, min_margin: float) -> dict:
    engine = ControlledContinuationEngine(
        policy=ContinuationPolicy(min_score=min_score, min_margin=min_margin)
    )
    output_rows = []
    for row in prompt_rows:
        result = engine.continue_prompt(row.get("prompt", ""))
        output_rows.append(
            {
                "id": row.get("id", ""),
                "prompt": row.get("prompt", ""),
                "ambiguous_term": row.get("ambiguous_term", "") or result.ambiguous_term or "",
                "expected_sense": row.get("expected_sense", "") or "",
                "difficulty_type": row.get("difficulty_type", ""),
                "notes": row.get("notes", ""),
                "continuation": result.continuation,
                "selected_sense": result.selected_sense or "",
                "confidence": f"{result.confidence:.4f}",
                "decision": result.decision,
                "reasons": " | ".join(result.reasons),
                "memory_hits": " | ".join(result.memory_hits),
            }
        )
    return summarize_rows(output_rows)


def run_sweep(input_path: str, output_path: str) -> list[dict]:
    prompt_rows = _load_prompts(input_path)
    sweep_rows = []

    for index, (min_score, min_margin) in enumerate(product(MIN_SCORES, MIN_MARGINS), start=1):
        metrics = _run_config(prompt_rows, min_score=min_score, min_margin=min_margin)
        sweep_rows.append(
            {
                "config_id": f"score{min_score:g}_margin{min_margin:g}",
                "min_score": f"{min_score:g}",
                "min_margin": f"{min_margin:g}",
                "total": metrics["total"],
                "continue_count": metrics["continue_count"],
                "audit_count": metrics["audit_count"],
                "suppress_count": metrics["suppress_count"],
                "correct_continue_count": metrics["correct_continue_count"],
                "wrong_continue_count": metrics["wrong_continue_count"],
                "coverage_rate": f"{metrics['coverage_rate']:.4f}",
                "precision_on_continued": f"{metrics['precision_on_continued']:.4f}",
                "answerable_recall": f"{metrics['answerable_recall']:.4f}",
                "wrong_continue_rate": f"{metrics['wrong_continue_rate']:.4f}",
                "abstention_rate_on_answerable": f"{metrics['abstention_rate_on_answerable']:.4f}",
            }
        )

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(sweep_rows)

    return sweep_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep controlled continuation policy thresholds.")
    parser.add_argument("--input", required=True, help="Input v1 prompt CSV")
    parser.add_argument("--output", required=True, help="Output policy sweep CSV")
    args = parser.parse_args()

    rows = run_sweep(args.input, args.output)
    print(f"Wrote {len(rows)} policy configurations to {args.output}")


if __name__ == "__main__":
    main()
