"""Summarize v1 Microworld controlled continuation outputs.

Usage:
    python -m worldpgt.experiments.summarize_v1_microworld_results \
        --input worldpgt/experiments/microworld_continuation_v1_outputs.csv \
        --output worldpgt/experiments/microworld_continuation_v1_summary.json
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


def _safe_confidence(row: dict) -> float:
    try:
        return float(row.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _rate(count: int, total: int) -> float:
    return round(count / total, 4) if total else 0.0


def _empty_metrics(include_suppress: bool) -> dict:
    metrics = {
        "total": 0,
        "correct_sense_count": 0,
        "wrong_sense_count": 0,
        "no_sense_count": 0,
        "continue_count": 0,
        "audit_count": 0,
        "_confidence_sum": 0.0,
    }
    if include_suppress:
        metrics["suppress_count"] = 0
    return metrics


def _add_row(metrics: dict, row: dict) -> None:
    expected = _norm(row.get("expected_sense"))
    selected = _norm(row.get("selected_sense"))
    decision = row.get("decision", "")

    metrics["total"] += 1
    if decision == "continue":
        metrics["continue_count"] += 1
    elif decision == "audit":
        metrics["audit_count"] += 1
    elif decision == "suppress" and "suppress_count" in metrics:
        metrics["suppress_count"] += 1
    metrics["_confidence_sum"] += _safe_confidence(row)

    classification = _classify(expected, selected, decision)
    if classification == "correct":
        metrics["correct_sense_count"] += 1
    elif classification == "wrong":
        metrics["wrong_sense_count"] += 1
    else:
        metrics["no_sense_count"] += 1


def _finalize(metrics: dict, include_suppress: bool) -> dict:
    total = metrics["total"]
    finalized = {
        "total": total,
        "correct_sense_count": metrics["correct_sense_count"],
        "wrong_sense_count": metrics["wrong_sense_count"],
        "no_sense_count": metrics["no_sense_count"],
        "continue_count": metrics["continue_count"],
        "audit_count": metrics["audit_count"],
    }
    if include_suppress:
        finalized["suppress_count"] = metrics.get("suppress_count", 0)
    finalized.update(
        {
            "correct_sense_rate": _rate(metrics["correct_sense_count"], total),
            "wrong_sense_rate": _rate(metrics["wrong_sense_count"], total),
            "audit_rate": _rate(metrics["audit_count"], total),
        }
    )
    if include_suppress:
        finalized["suppress_rate"] = _rate(metrics.get("suppress_count", 0), total)
    finalized["avg_confidence"] = round(metrics["_confidence_sum"] / total, 4) if total else 0.0
    return finalized


def summarize(input_path: str) -> dict:
    global_metrics = _empty_metrics(include_suppress=True)
    grouped: dict[str, dict] = {}

    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            _add_row(global_metrics, row)
            difficulty = row.get("difficulty_type", "") or "unknown"
            grouped.setdefault(difficulty, _empty_metrics(include_suppress=False))
            _add_row(grouped[difficulty], row)

    summary = _finalize(global_metrics, include_suppress=True)
    summary["by_difficulty_type"] = {
        difficulty: _finalize(metrics, include_suppress=False)
        for difficulty, metrics in sorted(grouped.items())
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Microworld controlled continuation v1 outputs.")
    parser.add_argument("--input", required=True, help="Input v1 Microworld output CSV")
    parser.add_argument("--output", required=True, help="Output JSON summary path")
    args = parser.parse_args()

    summary = summarize(args.input)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote summary to {args.output}")


if __name__ == "__main__":
    main()
