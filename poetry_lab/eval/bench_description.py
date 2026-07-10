"""Benchmark the description prompt battery: latency, memory, and output.

Measures, per prompt:
  - wall-clock time per call (mean/median/stdev over N repeats, ms)
  - incremental peak memory during generation (tracemalloc, KB) —
    isolates the render call from the one-time artifact-load cost
  - process peak RSS after all runs (resource.getrusage, MB) — includes
    the loaded artifact and interpreter baseline, reported once
  - output shape: sentence count, word count, character count
  - novelty ratio (from the existing novelty gate)
  - the realized paragraph itself

Artifact load (ingest → PhraseModel/ConceptGraph construction) is timed
separately since it happens once per process, not once per prompt.
"""

from __future__ import annotations

import json
import resource
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poemcore.narrative import NarrativeEngine
from poemcore.text import words

PROMPTS = [
    "Опиши вечер в Москве",
    "Опиши комнату",
    "Опиши улицу",
    "Опиши город",
    "Опиши дверь",
]

REPEATS = 20
WARMUP = 3


def _bench_one(engine: NarrativeEngine, prompt: str) -> dict:
    # Warm-up runs prime any lazy state (dict resizing, etc.) so the timed
    # runs measure steady-state cost, not first-call setup.
    for _ in range(WARMUP):
        engine.run(prompt, sentences=3, seed=f"warmup")

    times_ms: list[float] = []
    peaks_kb: list[float] = []
    result = None
    for i in range(REPEATS):
        tracemalloc.start()
        t0 = time.perf_counter()
        result = engine.run(prompt, sentences=3, seed=f"bench-{i}")
        elapsed = time.perf_counter() - t0
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times_ms.append(elapsed * 1000)
        peaks_kb.append(peak / 1024)

    text = result.paragraph.text()
    return {
        "prompt": prompt,
        "response": text,
        "time_ms_mean": statistics.mean(times_ms),
        "time_ms_median": statistics.median(times_ms),
        "time_ms_stdev": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "time_ms_min": min(times_ms),
        "time_ms_max": max(times_ms),
        "mem_kb_mean": statistics.mean(peaks_kb),
        "mem_kb_max": max(peaks_kb),
        "sentences": len(result.paragraph.sentences),
        "words": len(words(text)),
        "chars": len(text),
        "novelty_ratio": result.novelty.novelty_ratio,
        "echoed_lines": result.novelty.echoed_lines,
        "mode": result.plan.goal.mode,
    }


def main() -> None:
    load_start = time.perf_counter()
    tracemalloc.start()
    engine = NarrativeEngine()
    _current, load_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    load_ms = (time.perf_counter() - load_start) * 1000

    rows = [_bench_one(engine, prompt) for prompt in PROMPTS]

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024

    report = {
        "artifact_load_ms": round(load_ms, 2),
        "artifact_load_peak_kb": round(load_peak / 1024, 1),
        "process_peak_rss_mb": round(rss_mb, 2),
        "repeats_per_prompt": REPEATS,
        "warmup_per_prompt": WARMUP,
        "rows": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
