"""Runtime summaries for Microworld and GPT-2 continuation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine


def _read_csv(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def summarize_generation_times(rows: list[dict]) -> dict:
    times = [
        parsed
        for parsed in (_float_or_none(row.get("generation_time_sec")) for row in rows)
        if parsed is not None
    ]
    summary = {
        "total_rows": len(rows),
        "has_generation_time_sec": bool(times),
        "total_generation_time_sec": None,
        "avg_generation_time_sec": None,
        "median_generation_time_sec": None,
        "min_generation_time_sec": None,
        "max_generation_time_sec": None,
    }
    if not times:
        return summary
    summary.update(
        {
            "total_generation_time_sec": round(sum(times), 6),
            "avg_generation_time_sec": round(sum(times) / len(times), 6),
            "median_generation_time_sec": round(statistics.median(times), 6),
            "min_generation_time_sec": round(min(times), 6),
            "max_generation_time_sec": round(max(times), 6),
        }
    )
    return summary


def time_microworld_prompts(prompt_rows: list[dict]) -> dict:
    engine = ControlledContinuationEngine()
    per_prompt_times: list[float] = []
    start_total = time.perf_counter()
    for row in prompt_rows:
        start = time.perf_counter()
        engine.continue_prompt(row.get("prompt", ""))
        per_prompt_times.append(time.perf_counter() - start)
    total_time = time.perf_counter() - start_total
    if not per_prompt_times:
        return {
            "prompt_count": 0,
            "total_time_sec": 0.0,
            "avg_time_sec_per_prompt": 0.0,
            "median_time_sec_per_prompt": 0.0,
            "min_time_sec_per_prompt": 0.0,
            "max_time_sec_per_prompt": 0.0,
        }
    return {
        "prompt_count": len(prompt_rows),
        "total_time_sec": round(total_time, 6),
        "avg_time_sec_per_prompt": round(sum(per_prompt_times) / len(per_prompt_times), 6),
        "median_time_sec_per_prompt": round(statistics.median(per_prompt_times), 6),
        "min_time_sec_per_prompt": round(min(per_prompt_times), 6),
        "max_time_sec_per_prompt": round(max(per_prompt_times), 6),
    }


def build_summary(
    prompts_path: str | Path,
    microworld_output_path: str | Path,
    gpt2_output_path: str | Path,
) -> dict:
    prompt_rows = _read_csv(prompts_path)
    microworld_rows = _read_csv(microworld_output_path)
    gpt2_rows = _read_csv(gpt2_output_path)
    return {
        "dataset": {
            "prompt_rows": len(prompt_rows),
            "prompts_path": str(prompts_path),
        },
        "microworld": {
            "existing_output": summarize_generation_times(microworld_rows),
            "timed_run": time_microworld_prompts(prompt_rows),
            "timing_note": "Microworld output CSV has no per-row generation_time_sec; timed_run measures a fresh deterministic engine pass.",
        },
        "gpt2": {
            "existing_output": summarize_generation_times(gpt2_rows),
            "timing_note": "GPT-2 timing uses generation_time_sec already recorded in the inference output CSV; GPT-2 was not rerun.",
        },
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize Microworld and GPT-2 runtime data.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--microworld-output", required=True)
    parser.add_argument("--gpt2-output", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = build_summary(args.prompts, args.microworld_output, args.gpt2_output)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(summary, indent=2, sort_keys=True)
    output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

