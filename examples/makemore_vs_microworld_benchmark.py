"""Compare Microworld name generation with a makemore-style neural baseline.

PyTorch is optional.  If it is unavailable, the benchmark still reports
Microworld metrics and writes an empty makemore audit CSV for later use.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.surname_generator import (
    SurnameTransitionGraph,
    context_to_str,
    load_surnames,
    load_trust_profile,
)
from core.memory_benchmark import (
    MemoryTracker,
    empty_memory_metrics,
    memory_mb_summary,
    phase_memory_metrics,
)
from examples.makemore_baseline import (
    DEFAULT_OUTPUT as DEFAULT_MAKEMORE_CSV,
    run_baseline as run_makemore_baseline,
    write_makemore_audit_csv,
)
from examples.surname_audit_summary import compute_summary, read_labeled_rows
from examples.surname_generate import generate_names
from examples.surname_trust_learn import learn_trust

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
MEMORY_NOTES = [
    "RSS includes Python interpreter and library overhead.",
    "Peak RSS is sampled and approximate.",
    "State size and runtime RSS are different metrics.",
]


def _parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _safe_ratio(numerator, denominator):
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _file_size(path: str | None) -> int:
    return os.path.getsize(path) if path and os.path.exists(path) else 0


def _json_size_bytes(obj) -> int:
    return len(json.dumps(obj, sort_keys=True).encode("utf-8"))


def transition_graph_state_size_bytes(graph: SurnameTransitionGraph) -> int:
    state = {
        "order": graph.order,
        "transition_counts": {
            context_to_str(context): counts
            for context, counts in sorted(graph.transition_counts.items())
        },
    }
    return _json_size_bytes(state)


def transition_count(graph: SurnameTransitionGraph) -> int:
    return sum(len(bucket) for bucket in graph.transition_counts.values())


def quality_metrics_from_audit(path: str | None) -> dict | None:
    if not path or not os.path.exists(path):
        return None
    try:
        rows = read_labeled_rows(path)
    except ValueError:
        return None
    if not rows:
        return None
    summary = compute_summary(rows)
    counts = summary["counts"]
    return {
        "reviewed": summary["total"],
        "good": counts["good"],
        "bad": counts["bad"],
        "unclear": counts["unclear"],
        "good_rate": summary["good_rate"],
        "bad_rate": summary["bad_rate"],
        "unclear_rate": summary["unclear_rate"],
        "generation_precision": summary["generation_precision"],
    }


def run_microworld(
    input_path: str,
    *,
    count: int,
    order: int,
    seed: int,
    trust_profile_path: str | None,
    audit_path: str | None,
    track_memory: bool = True,
    memory_sample_interval_ms: int = 10,
) -> dict:
    names = load_surnames(input_path)

    with MemoryTracker(
        "microworld_build",
        enabled=track_memory,
        interval_ms=memory_sample_interval_ms,
    ) as build_memory_tracker:
        build_start = time.perf_counter()
        graph = SurnameTransitionGraph(order=order).build(names)
        build_time = time.perf_counter() - build_start
    build_memory = build_memory_tracker.to_dict()

    trust_profile = None
    trust_load_time = 0.0
    with MemoryTracker(
        "microworld_trust_load",
        enabled=track_memory,
        interval_ms=memory_sample_interval_ms,
    ) as trust_memory_tracker:
        if trust_profile_path:
            trust_start = time.perf_counter()
            trust_profile = load_trust_profile(trust_profile_path)
            trust_load_time = time.perf_counter() - trust_start
    trust_memory = trust_memory_tracker.to_dict()

    with MemoryTracker(
        "microworld_generation",
        enabled=track_memory,
        interval_ms=memory_sample_interval_ms,
    ) as generation_memory_tracker:
        gen_start = time.perf_counter()
        generated = generate_names(
            graph,
            count,
            rng=random.Random(seed),
            source_set=set(names),
            avoid_duplicates=True,
            soft_max_length=10,
            length_end_bias=1.5,
            trust_profile=trust_profile,
        )
        gen_time = time.perf_counter() - gen_start
    generation_memory = generation_memory_tracker.to_dict()

    adaptation_time = None
    adaptation_rows = 0
    adaptation_state_size = None
    if audit_path and os.path.exists(audit_path):
        with MemoryTracker(
            "microworld_audit_adaptation",
            enabled=track_memory,
            interval_ms=memory_sample_interval_ms,
        ) as adaptation_memory_tracker:
            rows = read_labeled_rows(audit_path)
            adaptation_rows = len(rows)
            adapt_start = time.perf_counter()
            adapted = learn_trust(rows, order)
            adaptation_time = time.perf_counter() - adapt_start
            adaptation_state_size = _json_size_bytes(adapted)
        adaptation_memory = adaptation_memory_tracker.to_dict()
    else:
        adaptation_memory = None

    graph_size = transition_graph_state_size_bytes(graph)
    trust_size = _file_size(trust_profile_path)
    memory = {
        "available": build_memory["available"],
        "memory_metrics_available": build_memory["available"],
        "skipped_reason": build_memory["skipped_reason"],
    }
    memory.update(phase_memory_metrics("build", build_memory))
    memory.update(phase_memory_metrics("trust_load", trust_memory))
    memory.update(phase_memory_metrics("generation", generation_memory))
    memory.update(phase_memory_metrics("audit_adaptation", adaptation_memory))
    memory["memory_mb"] = memory_mb_summary(memory)
    return {
        "build_transition_graph_time_sec": build_time,
        "trust_profile_load_time_sec": trust_load_time,
        "generation_time_sec": gen_time,
        "generated_count": len(generated),
        "transition_count": transition_count(graph),
        "transition_graph_state_size_bytes": graph_size,
        "trust_profile_size_bytes": trust_size,
        "total_explicit_state_size_bytes": graph_size + trust_size,
        "trainable_parameter_count": 0,
        "uses_backpropagation": False,
        "uses_neural_weights": False,
        "audit_adaptation_time_sec": adaptation_time,
        "audit_rows_used": adaptation_rows,
        "audit_adaptation_state_size_bytes": adaptation_state_size,
        "memory": memory,
    }


def run_makemore(
    input_path: str,
    *,
    count: int,
    seed: int,
    steps: int,
    embedding_dim: int,
    hidden_dim: int,
    block_size: int,
    output_csv: str = DEFAULT_MAKEMORE_CSV,
    batch_size: int = 32,
    learning_rate: float = 0.1,
    temperature: float = 0.8,
    max_length: int = 16,
    min_length: int = 3,
    force: bool = False,
    track_memory: bool = True,
    memory_sample_interval_ms: int = 10,
) -> dict:
    return run_makemore_baseline(
        input_path,
        count=count,
        seed=seed,
        steps=steps,
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        block_size=block_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        temperature=temperature,
        max_length=max_length,
        min_length=min_length,
        output_csv=output_csv,
        force=force,
        track_memory=track_memory,
        memory_sample_interval_ms=memory_sample_interval_ms,
    )


def run_benchmark(
    *,
    input_path: str,
    count: int,
    order: int,
    seed: int,
    audit_path: str | None,
    trust_profile_path: str | None,
    run_neural: bool,
    makemore_steps: int,
    makemore_embedding_dim: int,
    makemore_hidden_dim: int,
    makemore_block_size: int,
    makemore_csv: str = DEFAULT_MAKEMORE_CSV,
    makemore_batch_size: int = 32,
    makemore_learning_rate: float = 0.1,
    makemore_temperature: float = 0.8,
    makemore_max_length: int = 16,
    makemore_min_length: int = 3,
    makemore_force: bool = False,
    track_memory: bool = True,
    memory_sample_interval_ms: int = 10,
) -> dict:
    microworld = run_microworld(
        input_path,
        count=count,
        order=order,
        seed=seed,
        trust_profile_path=trust_profile_path,
        audit_path=audit_path,
        track_memory=track_memory,
        memory_sample_interval_ms=memory_sample_interval_ms,
    )
    if run_neural:
        makemore = run_makemore(
            input_path,
            count=count,
            seed=seed,
            steps=makemore_steps,
            embedding_dim=makemore_embedding_dim,
            hidden_dim=makemore_hidden_dim,
            block_size=makemore_block_size,
            output_csv=makemore_csv,
            batch_size=makemore_batch_size,
            learning_rate=makemore_learning_rate,
            temperature=makemore_temperature,
            max_length=makemore_max_length,
            min_length=makemore_min_length,
            force=makemore_force,
            track_memory=track_memory,
            memory_sample_interval_ms=memory_sample_interval_ms,
        )
    else:
        write_makemore_audit_csv([], makemore_csv)
        makemore = {
            "available": False,
            "skipped_reason": "--run-makemore false",
            "uses_backpropagation": True,
            "uses_neural_weights": True,
            "generated_csv": makemore_csv,
            "memory": empty_memory_metrics(["training", "generation"], enabled=track_memory),
        }

    makemore_quality = quality_metrics_from_audit(makemore_csv)
    return {
        "input": {
            "input_path": input_path,
            "count": count,
            "order": order,
            "seed": seed,
            "audit_path": audit_path,
            "trust_profile_path": trust_profile_path,
            "run_makemore": run_neural,
            "makemore_steps": makemore_steps,
            "makemore_embedding_dim": makemore_embedding_dim,
            "makemore_hidden_dim": makemore_hidden_dim,
            "makemore_block_size": makemore_block_size,
            "makemore_batch_size": makemore_batch_size,
            "makemore_learning_rate": makemore_learning_rate,
            "makemore_temperature": makemore_temperature,
            "makemore_max_length": makemore_max_length,
            "makemore_min_length": makemore_min_length,
            "track_memory": track_memory,
            "memory_sample_interval_ms": memory_sample_interval_ms,
        },
        "microworld": microworld,
        "makemore": makemore,
        "quality": {
            "audited_microworld": quality_metrics_from_audit(audit_path),
            "audited_makemore": makemore_quality,
        },
        "efficiency_ratios": {
            "state_size_ratio_makemore_to_microworld": _safe_ratio(
                makemore.get("model_state_size_bytes"),
                microworld.get("total_explicit_state_size_bytes"),
            ),
            "training_time_ratio_makemore_to_microworld": _safe_ratio(
                makemore.get("training_time_sec"),
                microworld.get("build_transition_graph_time_sec"),
            ),
            "generation_time_ratio_makemore_to_microworld": _safe_ratio(
                makemore.get("generation_time_sec"),
                microworld.get("generation_time_sec"),
            ),
            "parameter_count_ratio_makemore_to_microworld": None,
            "peak_training_rss_ratio_makemore_to_microworld_build": _safe_ratio(
                makemore.get("memory", {}).get("training_peak_rss_bytes"),
                microworld.get("memory", {}).get("build_peak_rss_bytes"),
            ),
            "peak_generation_rss_ratio_makemore_to_microworld_generation": _safe_ratio(
                makemore.get("memory", {}).get("generation_peak_rss_bytes"),
                microworld.get("memory", {}).get("generation_peak_rss_bytes"),
            ),
            "rss_delta_training_ratio_makemore_to_microworld_build": _safe_ratio(
                makemore.get("memory", {}).get("training_rss_delta_bytes"),
                microworld.get("memory", {}).get("build_rss_delta_bytes"),
            ),
            "rss_delta_generation_ratio_makemore_to_microworld_generation": _safe_ratio(
                makemore.get("memory", {}).get("generation_rss_delta_bytes"),
                microworld.get("memory", {}).get("generation_rss_delta_bytes"),
            ),
        },
        "memory_notes": MEMORY_NOTES,
    }


def write_json(result: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Benchmark Microworld name generation against a makemore-style baseline."
    )
    ap.add_argument("--input", default=os.path.join(_ROOT, "data", "surnames.txt"))
    ap.add_argument("--count", type=int, default=100)
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--seed", type=int, default=50)
    ap.add_argument("--audit", default=None)
    ap.add_argument("--trust-profile", default=None, dest="trust_profile")
    ap.add_argument("--output", required=True)
    ap.add_argument("--run-makemore", default="true", dest="run_makemore")
    ap.add_argument("--makemore-steps", type=int, default=50000)
    ap.add_argument("--makemore-embedding-dim", type=int, default=16)
    ap.add_argument("--makemore-hidden-dim", type=int, default=200)
    ap.add_argument("--makemore-block-size", type=int, default=3)
    ap.add_argument("--makemore-batch-size", type=int, default=32)
    ap.add_argument("--makemore-learning-rate", type=float, default=0.1)
    ap.add_argument("--makemore-temperature", type=float, default=0.8)
    ap.add_argument("--makemore-max-length", type=int, default=16)
    ap.add_argument("--makemore-min-length", type=int, default=3)
    ap.add_argument("--makemore-force", default="false")
    ap.add_argument("--makemore-output", default=DEFAULT_MAKEMORE_CSV)
    ap.add_argument("--track-memory", default="true")
    ap.add_argument("--memory-sample-interval-ms", type=int, default=10)
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    result = run_benchmark(
        input_path=args.input,
        count=args.count,
        order=args.order,
        seed=args.seed,
        audit_path=args.audit,
        trust_profile_path=args.trust_profile,
        run_neural=_parse_bool(args.run_makemore),
        makemore_steps=args.makemore_steps,
        makemore_embedding_dim=args.makemore_embedding_dim,
        makemore_hidden_dim=args.makemore_hidden_dim,
        makemore_block_size=args.makemore_block_size,
        makemore_csv=args.makemore_output,
        makemore_batch_size=args.makemore_batch_size,
        makemore_learning_rate=args.makemore_learning_rate,
        makemore_temperature=args.makemore_temperature,
        makemore_max_length=args.makemore_max_length,
        makemore_min_length=args.makemore_min_length,
        makemore_force=_parse_bool(args.makemore_force),
        track_memory=_parse_bool(args.track_memory),
        memory_sample_interval_ms=args.memory_sample_interval_ms,
    )
    write_json(result, args.output)
    print(f"Wrote benchmark JSON -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
