"""Run cases through Microworld's public in-process API path."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
import time
import tracemalloc
from typing import Iterable

from worldpgt.api import server
from .dataset import read_jsonl


def _answer(case: dict, sequence: int) -> dict:
    started = time.perf_counter_ns()
    try:
        response = server.ask(server.AskRequest(
            question=case["question"], enable_reasoning=True, enable_multihop=False,
            web_search=False, community_context=False, cognitive_patterns=False,
            session_id=f"open-book-qa-{sequence}",
        ))
        elapsed = (time.perf_counter_ns() - started) / 1_000_000
        plan = response.answer_plan or {}
        selected = [block.get("step", {}).get("edge", {}).get("evidence_id") for block in plan.get("blocks", [])]
        return {"id": case["id"], "answer": response.answer, "decision": response.decision,
                "support_kind": response.support, "total_latency_ms": elapsed,
                "planner_latency_ms": None, "selected_relation_ids": [x for x in selected if x],
                "answer_plan_blocks": len(plan.get("blocks", [])), "audit_reason": None,
                "trace": {"answer_plan": response.answer_plan, "resolved_references": response.resolved_references},
                "exception": None}
    except Exception as exc:  # preserve failures as benchmark data
        return {"id": case["id"], "answer": "", "decision": "exception", "support_kind": None,
                "total_latency_ms": (time.perf_counter_ns() - started) / 1_000_000,
                "planner_latency_ms": None, "selected_relation_ids": [], "answer_plan_blocks": 0,
                "audit_reason": None, "trace": None, "exception": repr(exc)}


def run(dataset: Iterable[dict], *, warmups: int = 50, repeats: int = 5, seed: int = 42,
        experimental_graph_paths: Iterable[str | Path] | None = None) -> tuple[list[dict], dict]:
    """Startup and warm queries are intentionally measured separately."""
    startup = time.perf_counter_ns()
    server._startup("pump-dry-run", include_experimental_web_graph=experimental_graph_paths is None,
                    experimental_graph_paths=experimental_graph_paths,
                    community_context_path=None, cognitive_patterns_path=None,
                    # Creative phrase-graph training is unrelated to factual
                    # QA and would dominate cold startup in an isolated run.
                    warm_phrase_graph_on_startup=False)
    startup_ms = (time.perf_counter_ns() - startup) / 1_000_000
    rows = list(dataset)
    if not rows:
        return [], {"startup_ms": startup_ms}
    # The runtime normally logs acquisition audits.  Suppress only that write
    # side effect for a reproducible read-only benchmark; answer routing is unchanged.
    server.log_audit_event = lambda *args, **kwargs: None
    for index in range(warmups):
        _answer(rows[index % len(rows)], -index - 1)
    order = list(range(len(rows))) * repeats
    random.Random(seed).shuffle(order)
    tracemalloc.start(); tracemalloc.reset_peak()
    results = []
    for sequence, index in enumerate(order):
        result = _answer(rows[index], sequence)
        result["repeat"] = sequence // len(rows)
        results.append(result)
    _current, peak = tracemalloc.get_traced_memory(); tracemalloc.stop()
    metadata = {"system": "MicroWorld explicit graph runtime", "startup_ms": startup_ms,
                "warmup_queries": warmups, "repeats": repeats, "peak_incremental_python_heap_mib": peak / 1024**2,
                "generated_timestamp": datetime.now(timezone.utc).isoformat(),
                "overlay": "isolated-experimental-graph" if experimental_graph_paths is not None else "pump-dry-run+experimental-web-graph",
                "persistent_sqlite": True}
    return results, metadata


def run_file(dataset_path: str | Path, output: str | Path, **kwargs: object) -> dict:
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    results, metadata = run(read_jsonl(dataset_path), **kwargs)
    (output / "microworld_results.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results), encoding="utf-8")
    (output / "microworld_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    (output / "latency_samples_microworld.csv").write_text("id,total_latency_ms\n" + "".join(f"{row['id']},{row['total_latency_ms']:.6f}\n" for row in results), encoding="utf-8")
    failures = [row for row in results if row["exception"]]
    (output / "failures_microworld.jsonl").write_text("".join(json.dumps(row) + "\n" for row in failures), encoding="utf-8")
    return metadata
