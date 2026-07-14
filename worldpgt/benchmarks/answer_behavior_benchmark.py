"""Performance matrix for the evidence-scoped answer-behavior planner.

This measures the real ``build_answer_plan`` / ``render_answer_plan`` path,
not a reimplementation.  Synthetic data is relation-only and stays entirely
in process: it never writes accepted memory or campaign overlays.

Examples::

    # Safe laptop run: storage through 100k plus all locality/depth cases.
    python3 -m worldpgt.benchmarks.answer_behavior_benchmark --profile standard

    # Explicitly opt in to the full 1k .. 10m storage matrix.
    python3 -m worldpgt.benchmarks.answer_behavior_benchmark \
        --profile full --allow-large --output /tmp/answer_behavior_perf.json

The 1m and 10m points intentionally require ``--allow-large``: dict-shaped
overlay fixtures are memory-heavy by design, and silently exhausting a laptop
would not be a useful benchmark result.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import resource
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, TypeVar

from worldpgt.reasoning.answer_behavior import (
    PlanningInstrumentation,
    build_answer_plan,
    prepare_evidence_graph,
)
from worldpgt.reasoning.answer_plan_renderer import render_answer_plan


STORE_SIZES = (1_000, 10_000, 100_000, 1_000_000, 10_000_000)
STANDARD_STORE_SIZES = STORE_SIZES[:3]
TARGET_DEGREES = (5, 50, 500, 5_000)
DEPTHS = (1, 2, 3, 4)
_TARGET = "benchmark target"
_QUESTION = "What does benchmark target enable?"
T = TypeVar("T")


@dataclass(frozen=True)
class Measurement:
    value: T
    elapsed_ns: int
    peak_python_bytes: int
    peak_rss_bytes: int


def _edge(subject: str, predicate: str, obj: str, number: int) -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_text": f"{subject} {predicate.replace('_', ' ')} {obj}.",
        "source_url": f"https://benchmark.invalid/{number}",
        "stability": "semi_stable",
        "risk": "medium",
        "trust": "proposal_benchmark_only",
    }


def _rss_bytes() -> int:
    # macOS reports bytes; Linux and most BSD-compatible CI environments
    # report KiB.  This is a process high-water mark, so ``tracemalloc`` is
    # also reported for a resettable per-scenario peak.
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def _measure(operation: Callable[[], T]) -> Measurement:
    gc.collect()
    tracemalloc.start()
    tracemalloc.reset_peak()
    started = time.perf_counter_ns()
    value = operation()
    elapsed = time.perf_counter_ns() - started
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Measurement(value, elapsed, peak, _rss_bytes())


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _latency_summary(samples_ns: Iterable[int]) -> dict[str, float]:
    samples_ms = [sample / 1_000_000 for sample in samples_ns]
    return {
        "p50_ms": round(_percentile(samples_ms, 0.50), 4),
        "p95_ms": round(_percentile(samples_ms, 0.95), 4),
        "p99_ms": round(_percentile(samples_ms, 0.99), 4),
    }


def synthetic_overlay(size: int, *, target_degree: int = 50) -> list[dict]:
    """Return a graph with a controlled hot target and disconnected filler."""
    if size < target_degree:
        raise ValueError("size must be at least target_degree")
    items = [
        _edge(_TARGET, "enables", f"target capability capability_{index:05d}", index)
        for index in range(target_degree)
    ]
    items.extend(
        _edge(
            f"filler_subject_{index:05d}",
            "relates_to",
            f"filler_object_{index:05d}",
            index,
        )
        for index in range(target_degree, size)
    )
    return items


def _resolve_entity(question: str, entity_index: dict[str, str]) -> str | None:
    # This is deliberately a narrow timing of the benchmark's entity-index
    # lookup.  Full semantic parsing belongs to the API benchmark, not this
    # graph-layer benchmark.
    normalized = " ".join(question.casefold().split())
    return next((canonical for surface, canonical in entity_index.items() if surface in normalized), None)


def _run_query(prepared, *, max_blocks: int) -> tuple[dict, int, int]:
    entity_index = {_TARGET: _TARGET}
    resolve_started = time.perf_counter_ns()
    target = _resolve_entity(_QUESTION, entity_index)
    resolve_ns = time.perf_counter_ns() - resolve_started
    if target is None:
        raise RuntimeError("synthetic target failed to resolve")
    instrumentation = PlanningInstrumentation()
    planning_started = time.perf_counter_ns()
    plan = build_answer_plan(
        _QUESTION,
        [],
        targets=[target],
        max_blocks=max_blocks,
        prepared_edges=prepared,
        instrumentation=instrumentation,
    )
    planning_ns = time.perf_counter_ns() - planning_started
    rendering_started = time.perf_counter_ns()
    rendered = render_answer_plan(plan) if plan is not None else ""
    rendering_ns = time.perf_counter_ns() - rendering_started
    metric = instrumentation.to_dict(
        planning_ns=planning_ns,
        rendering_ns=rendering_ns,
    )
    metric["entity_resolution_ms"] = round(resolve_ns / 1_000_000, 4)
    metric["rendered_chars"] = len(rendered)
    return metric, planning_ns + rendering_ns + resolve_ns, rendering_ns


def _warm_summary(prepared, *, repetitions: int, max_blocks: int) -> dict:
    timings: list[int] = []
    metrics: list[dict] = []
    for _ in range(repetitions):
        metric, elapsed_ns, _render_ns = _run_query(prepared, max_blocks=max_blocks)
        timings.append(elapsed_ns)
        metrics.append(metric)
    middle = metrics[len(metrics) // 2]
    return {
        **_latency_summary(timings),
        "median_breakdown": middle,
    }


def run_storage_case(size: int, *, repetitions: int, target_degree: int = 50) -> dict:
    """Whole-store scale: cold preparation, index build, and warm queries."""
    generated = _measure(lambda: synthetic_overlay(size, target_degree=target_degree))
    overlay = generated.value
    index_build = _measure(lambda: prepare_evidence_graph(overlay))
    prepared = index_build.value
    warm = _warm_summary(prepared, repetitions=repetitions, max_blocks=4)
    return {
        "store_items": size,
        "target_degree": target_degree,
        "fixture_build_ms": round(generated.elapsed_ns / 1_000_000, 4),
        "cold_start_ms": round(index_build.elapsed_ns / 1_000_000, 4),
        # The current layer has a prepared-edge cache rather than a separate
        # adjacency index.  Keep the requested field, but name its exact work
        # honestly: validation/filtering/deduplication into prepared edges.
        "index_build": {
            "kind": "prepared_evidence_graph",
            "ms": round(index_build.elapsed_ns / 1_000_000, 4),
        },
        "peak_python_alloc_mb": round(index_build.peak_python_bytes / 1024**2, 3),
        "process_peak_rss_mb": round(index_build.peak_rss_bytes / 1024**2, 3),
        "warm_query": warm,
    }


def run_local_bundle_case(degree: int, *, repetitions: int) -> dict:
    """Hot-node scale with no unrelated storage noise."""
    prepared = prepare_evidence_graph(synthetic_overlay(degree, target_degree=degree))
    return {
        "target_degree": degree,
        "warm_query": _warm_summary(prepared, repetitions=repetitions, max_blocks=4),
    }


def synthetic_depth_overlay(depth: int, *, branch_factor: int = 3) -> list[dict]:
    if depth < 1:
        raise ValueError("depth must be at least one")
    items: list[dict] = []
    current = _TARGET
    for level in range(depth):
        nxt = f"depth_node_{level + 1:05d}"
        # "advances" sorts before branch predicates, so equal-quality choices
        # take the intended chain while the side edges remain real pruned
        # alternatives for the metrics.
        items.append(_edge(current, "advances", nxt, level))
        items.extend(
            _edge(
                current,
                f"branch_{branch}",
                f"side_node_{level:05d}_{branch:05d}",
                level * 10 + branch,
            )
            for branch in range(branch_factor)
        )
        current = nxt
    return items


def run_depth_case(depth: int, *, repetitions: int) -> dict:
    prepared = prepare_evidence_graph(synthetic_depth_overlay(depth))
    return {
        "max_depth": depth,
        "warm_query": _warm_summary(prepared, repetitions=repetitions, max_blocks=depth),
    }


def run_matrix(
    *,
    store_sizes: Iterable[int],
    repetitions: int,
    include_storage: bool,
    include_bundle: bool,
    include_depth: bool,
) -> dict:
    return {
        "benchmark": "answer_behavior_v1",
        "notes": [
            "Synthetic relation-only data; no accepted-memory writes.",
            "process_peak_rss_mb is a process high-water mark; peak_python_alloc_mb resets per case.",
            "entity_resolution_ms measures the harness surface-index lookup; semantic-parser timing belongs to API benchmarks.",
        ],
        "storage_scale": [
            run_storage_case(size, repetitions=repetitions)
            for size in store_sizes
        ] if include_storage else [],
        "local_bundle": [
            run_local_bundle_case(degree, repetitions=repetitions)
            for degree in TARGET_DEGREES
        ] if include_bundle else [],
        "depth": [
            run_depth_case(depth, repetitions=repetitions)
            for depth in DEPTHS
        ] if include_depth else [],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("standard", "full"), default="standard")
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--group", action="append", choices=("storage", "bundle", "depth"))
    parser.add_argument("--allow-large", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    sizes = STORE_SIZES if args.profile == "full" else STANDARD_STORE_SIZES
    if max(sizes) > 100_000 and not args.allow_large:
        parser.error("1m and 10m fixtures require --allow-large")
    groups = set(args.group or ("storage", "bundle", "depth"))
    result = run_matrix(
        store_sizes=sizes,
        repetitions=args.repetitions,
        include_storage="storage" in groups,
        include_bundle="bundle" in groups,
        include_depth="depth" in groups,
    )
    serialized = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    print(serialized, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
