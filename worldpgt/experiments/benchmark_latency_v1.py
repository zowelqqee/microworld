"""Latency benchmark v1 for AnswerOrchestrator.

Runs 1000 questions through the orchestrator, measures per-query latency,
computes percentiles, throughput, and RAM usage. Saves results to
worldpgt/experiments/benchmarks/latency_TIMESTAMP.json.

Usage::

    python3 -m worldpgt.experiments.benchmark_latency_v1 [--overlay pump-dry-run]
    python3 worldpgt/experiments/benchmark_latency_v1.py --overlay pump-dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator

_EXPERIMENTS = Path(__file__).resolve().parent
_BENCHMARKS_DIR = _EXPERIMENTS / "benchmarks"

# ---------------------------------------------------------------------------
# Question pool builders
# ---------------------------------------------------------------------------

def _load_csv_questions(csv_path: Path, col: str = "question") -> list[str]:
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as f:
        return [row[col] for row in csv.DictReader(f) if row.get(col)]


def _synthetic_questions(overlay_items: list[dict]) -> list[str]:
    """Generate synthetic question variations from overlay entities."""
    definition_templates = [
        "What is {e}?",
        "Who is {e}?",
        "Tell me about {e}.",
        "Define {e}.",
        "What kind of entity is {e}?",
        "Can you describe {e}?",
        "What do you know about {e}?",
    ]
    relation_templates_founded = [
        "Who founded {o}?",
        "Who started {o}?",
        "Who created {o}?",
        "Did {s} found {o}?",
        "What did {s} found?",
        "What companies did {s} start?",
    ]
    relation_templates_develops = [
        "What does {s} develop?",
        "What does {s} produce?",
        "What products does {s} make?",
    ]
    relation_templates_owns = [
        "Who owns {o}?",
        "What is {o} owned by?",
    ]
    adversarial_templates = [
        "Did {o} found {s}?",
        "Is {o} the founder of {s}?",
        "What is the current net worth of {s}?",
        "What is the live stock price of {s}?",
    ]

    defs = [
        item["subject"] for item in overlay_items
        if item.get("overlay_type") == "overlay_definition" and item.get("subject")
    ]

    qs: list[str] = []

    # Definition questions for every entity
    for e in defs:
        for tmpl in definition_templates:
            qs.append(tmpl.format(e=e))

    # Relation-derived questions
    for item in overlay_items:
        if item.get("overlay_type") != "overlay_relation":
            continue
        s = item.get("subject", "")
        o = item.get("object", "")
        pred = item.get("predicate", "")
        if not s or not o:
            continue
        if pred == "founded":
            for tmpl in relation_templates_founded:
                qs.append(tmpl.format(s=s, o=o))
        elif pred == "develops":
            for tmpl in relation_templates_develops:
                qs.append(tmpl.format(s=s, o=o))
        elif pred == "owned_by":
            for tmpl in relation_templates_owns:
                qs.append(tmpl.format(s=s, o=o))
        # Adversarial inversions
        if pred in ("founded", "founded_by", "leader_of", "develops"):
            for tmpl in adversarial_templates[:2]:
                qs.append(tmpl.format(s=s, o=o))

    return qs


def _build_question_pool(overlay_path: str, target: int = 1000) -> list[str]:
    """Assemble a pool of at least *target* unique questions."""
    overlay_items: list[dict] = json.loads(Path(overlay_path).read_text(encoding="utf-8"))

    seen: set[str] = set()
    pool: list[str] = []

    def _add(qs: list[str]) -> None:
        for q in qs:
            q = q.strip()
            if q and q not in seen:
                seen.add(q)
                pool.append(q)

    # 1. Existing QA artefacts
    _add(_load_csv_questions(_EXPERIMENTS / "entity_qa_prompts_v1.csv"))
    _add(_load_csv_questions(_EXPERIMENTS / "entity_qa_expansion_v1.csv"))
    _add(_load_csv_questions(_EXPERIMENTS / "entity_qa_adversarial_v1.csv"))
    _add(_load_csv_questions(_EXPERIMENTS / "assistant_surface_prompts_v1.csv"))
    _add(_load_csv_questions(_EXPERIMENTS / "cross_page_qa_v1.csv"))
    _add(_load_csv_questions(_EXPERIMENTS / "qa_generalization_test_v1.csv"))
    _add(_load_csv_questions(_EXPERIMENTS / "multihop_qa_v1" / "multihop_qa_outputs.csv"))

    # 2. Synthetic variations from overlay
    _add(_synthetic_questions(overlay_items))

    # 3. Cycle the existing pool if still short
    base = list(pool)
    idx = 0
    while len(pool) < target:
        q = base[idx % len(base)]
        variant = q.rstrip("?") + " — please explain."
        if variant not in seen:
            seen.add(variant)
            pool.append(variant)
        idx += 1

    return pool[:target]


# ---------------------------------------------------------------------------
# Percentile helper
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p))


# ---------------------------------------------------------------------------
# Hardware info
# ---------------------------------------------------------------------------

def _hardware_string() -> str:
    try:
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=2,
        ).stdout.strip()
    except Exception:
        cpu = platform.processor() or platform.machine()
    mem_gb = psutil.virtual_memory().total // (1024 ** 3)
    return f"{cpu or 'unknown CPU'}, {mem_gb} GB RAM"


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run(
    overlay_mode: str = "pump-dry-run",
    overlay_path: str | None = None,
    target: int = 1000,
    warmup: int = 5,
) -> dict:
    proc = psutil.Process(os.getpid())

    # Resolve overlay path for pool building
    from worldpgt.assistant_surface.context_selector import resolve_overlay
    resolved_path, _ = resolve_overlay(overlay_mode) if overlay_path is None else (overlay_path, None)
    overlay_items: list[dict] = json.loads(Path(resolved_path).read_text(encoding="utf-8"))

    rss_before_load = proc.memory_info().rss / (1024 ** 2)

    kwargs: dict = {"overlay_mode": overlay_mode}
    if overlay_path is not None:
        kwargs = {"overlay_path": overlay_path}

    orch = AnswerOrchestrator(**kwargs)

    rss_after_load = proc.memory_info().rss / (1024 ** 2)
    overlay_load_mb = rss_after_load - rss_before_load

    questions = _build_question_pool(resolved_path, target=target)

    # Warm-up (not measured)
    for q in questions[:warmup]:
        orch.answer(q)

    # Timed run
    latencies_ms: list[float] = []
    t_total_start = time.perf_counter()
    for q in questions:
        t0 = time.perf_counter()
        orch.answer(q)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
    total_time_s = time.perf_counter() - t_total_start

    peak_rss_mb = proc.memory_info().rss / (1024 ** 2)

    n = len(latencies_ms)
    p50 = _percentile(latencies_ms, 50)
    p95 = _percentile(latencies_ms, 95)
    p99 = _percentile(latencies_ms, 99)
    mn = min(latencies_ms)
    mx = max(latencies_ms)
    rps = n / total_time_s
    gpt4_typical_ms = 800.0
    speedup = gpt4_typical_ms / p50

    result = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overlay_mode": overlay_mode,
        "questions": n,
        "overlay_facts": len(overlay_items),
        "warmup": warmup,
        "latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "min": round(mn, 2),
            "max": round(mx, 2),
            "mean": round(float(np.mean(latencies_ms)), 2),
        },
        "throughput": {
            "requests_per_sec": round(rps, 1),
            "total_time_sec": round(total_time_s, 2),
        },
        "memory_mb": {
            "overlay_load": round(overlay_load_mb, 1),
            "peak_rss": round(peak_rss_mb, 1),
        },
        "gpt4_comparison": {
            "gpt4_typical_ms": gpt4_typical_ms,
            "speedup_vs_gpt4": round(speedup, 1),
        },
        "hardware": _hardware_string(),
    }
    return result


def _print_report(r: dict) -> None:
    lat = r["latency_ms"]
    tp = r["throughput"]
    mem = r["memory_mb"]
    cmp = r["gpt4_comparison"]

    print()
    print("MICROWORLD LATENCY BENCHMARK")
    print("=" * 38)
    print(f"Questions:        {r['questions']}")
    print(f"Overlay facts:    {r['overlay_facts']}")
    print()
    print("Latency (ms):")
    print(f"  p50:    {lat['p50']:.1f} ms")
    print(f"  p95:    {lat['p95']:.1f} ms")
    print(f"  p99:    {lat['p99']:.1f} ms")
    print(f"  min:    {lat['min']:.1f} ms")
    print(f"  max:    {lat['max']:.1f} ms")
    print()
    print("Throughput:")
    print(f"  requests/sec:   {tp['requests_per_sec']:.0f}")
    print()
    print("Memory:")
    print(f"  overlay load:   {mem['overlay_load']:.0f} MB")
    print(f"  peak RSS:       {mem['peak_rss']:.0f} MB")
    print()
    print(f"vs GPT-4 API (~{cmp['gpt4_typical_ms']:.0f}ms typical):")
    print(f"  speedup:        {cmp['speedup_vs_gpt4']:.0f}x faster")
    print()
    print(f"Hardware: {r['hardware']}")


def _save(result: dict) -> Path:
    _BENCHMARKS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = _BENCHMARKS_DIR / f"latency_{ts}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Latency benchmark for AnswerOrchestrator v1")
    parser.add_argument("--overlay", default="pump-dry-run", dest="overlay_mode",
                        help="Overlay mode (default: pump-dry-run)")
    parser.add_argument("--overlay-path", default=None,
                        help="Explicit overlay JSON path (overrides --overlay)")
    parser.add_argument("--questions", type=int, default=1000, dest="target",
                        help="Number of questions to run (default: 1000)")
    parser.add_argument("--warmup", type=int, default=5,
                        help="Warm-up questions (not timed, default: 5)")
    parser.add_argument("--no-save", action="store_true",
                        help="Skip saving JSON output")
    args = parser.parse_args(argv)

    result = run(
        overlay_mode=args.overlay_mode,
        overlay_path=args.overlay_path,
        target=args.target,
        warmup=args.warmup,
    )
    _print_report(result)

    if not args.no_save:
        out_path = _save(result)
        try:
            display_path = out_path.relative_to(_ROOT)
        except ValueError:
            display_path = out_path
        print(f"\nSaved → {display_path}")


if __name__ == "__main__":
    main()
