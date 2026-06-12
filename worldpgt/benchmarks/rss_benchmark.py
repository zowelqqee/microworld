"""Approximate subprocess RSS benchmark for Microworld and GPT-2 inference."""

from __future__ import annotations

import argparse
import csv
import contextlib
import io
import json
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _read_prompt_rows(path: str | Path, limit: int | None) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def _rss_mb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return raw / (1024 * 1024)
    return raw / 1024


def _worker_microworld(args: argparse.Namespace) -> dict:
    from worldpgt.continuation.continuation_engine import ControlledContinuationEngine

    rows = _read_prompt_rows(args.prompts, args.limit)
    engine = ControlledContinuationEngine()
    per_prompt = []
    start_total = time.perf_counter()
    for row in rows:
        start = time.perf_counter()
        engine.continue_prompt(row.get("prompt", ""))
        per_prompt.append(time.perf_counter() - start)
    runtime = time.perf_counter() - start_total
    return {
        "engine": "microworld",
        "prompt_count": len(rows),
        "runtime_sec": round(runtime, 6),
        "avg_time_sec_per_prompt": round(sum(per_prompt) / len(per_prompt), 6) if per_prompt else 0.0,
        "median_time_sec_per_prompt": round(statistics.median(per_prompt), 6) if per_prompt else 0.0,
        "peak_rss_mb": round(_rss_mb(), 3),
    }


def _worker_gpt2(args: argparse.Namespace) -> dict:
    from worldpgt.baselines.gpt2.run_gpt2_baseline import load_generator, resolve_nanogpt_dir

    rows = _read_prompt_rows(args.prompts, args.limit)
    nanogpt_dir = resolve_nanogpt_dir(args.nanogpt_dir)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        model_name, generate, fallback_reason = load_generator(
            nanogpt_dir=nanogpt_dir,
            device=args.device,
            seed=1337,
            compile_model=False,
        )
    per_prompt = []
    start_total = time.perf_counter()
    for row in rows:
        start = time.perf_counter()
        with contextlib.redirect_stdout(captured):
            generate(row.get("prompt", ""), 32, 0.8, 40)
        per_prompt.append(time.perf_counter() - start)
    runtime = time.perf_counter() - start_total
    return {
        "engine": "gpt2",
        "model": model_name,
        "fallback_reason": fallback_reason,
        "captured_stdout": captured.getvalue()[-1000:],
        "device": args.device,
        "prompt_count": len(rows),
        "runtime_sec": round(runtime, 6),
        "avg_time_sec_per_prompt": round(sum(per_prompt) / len(per_prompt), 6) if per_prompt else 0.0,
        "median_time_sec_per_prompt": round(statistics.median(per_prompt), 6) if per_prompt else 0.0,
        "peak_rss_mb": round(_rss_mb(), 3),
    }


def _run_worker(kind: str, args: argparse.Namespace) -> dict:
    command = [
        sys.executable,
        "-m",
        "worldpgt.benchmarks.rss_benchmark",
        "--_worker",
        kind,
        "--prompts",
        args.prompts,
        "--nanogpt-dir",
        args.nanogpt_dir,
        "--limit",
        str(args.limit),
        "--device",
        args.device,
    ]
    start = time.perf_counter()
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        return {
            "engine": kind,
            "error": completed.stderr.strip() or completed.stdout.strip(),
            "subprocess_runtime_sec": round(elapsed, 6),
        }
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {
            "engine": kind,
            "error": "Worker did not emit valid JSON.",
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "subprocess_runtime_sec": round(elapsed, 6),
        }


def build_summary(args: argparse.Namespace) -> dict:
    microworld = _run_worker("microworld", args)
    gpt2 = None if args.skip_gpt2 else _run_worker("gpt2", args)
    return {
        "prompt_count": args.limit,
        "microworld_peak_rss_mb": microworld.get("peak_rss_mb"),
        "gpt2_peak_rss_mb": None if gpt2 is None else gpt2.get("peak_rss_mb"),
        "microworld_runtime_sec": microworld.get("runtime_sec"),
        "gpt2_runtime_sec": None if gpt2 is None else gpt2.get("runtime_sec"),
        "microworld": microworld,
        "gpt2": gpt2,
        "rss_caveats": [
            "RSS is measured with resource.ru_maxrss inside subprocesses and is approximate.",
            "macOS reports ru_maxrss in bytes; Linux reports kilobytes.",
            "PyTorch and MPS memory behavior can make GPT-2 RSS noisy and environment-dependent.",
            "GPT-2 RSS includes model loading overhead for the subprocess.",
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure approximate peak RSS for Microworld and GPT-2.")
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--nanogpt-dir", default="nanogpt")
    parser.add_argument("--output")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--skip-gpt2", action="store_true")
    parser.add_argument("--_worker", choices=["microworld", "gpt2"], default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args._worker == "microworld":
        print(json.dumps(_worker_microworld(args), sort_keys=True))
        return
    if args._worker == "gpt2":
        print(json.dumps(_worker_gpt2(args), sort_keys=True))
        return

    summary = build_summary(args)
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
