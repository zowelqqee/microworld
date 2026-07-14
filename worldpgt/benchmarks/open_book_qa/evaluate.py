"""Deterministic (not LLM-as-judge) metrics for the open-book comparison."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import re
import unicodedata

from .dataset import read_jsonl


def normalize(value: object) -> str:
    value = unicodedata.normalize("NFKC", str(value or "")).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", value).split())


def percentile(values: list[float], point: float) -> float | None:
    if not values: return None
    values = sorted(values); position = (len(values) - 1) * point; low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def _contains(answer: str, value: str) -> bool:
    return normalize(value) in normalize(answer)


def _measure(case: dict, result: dict, *, system: str, all_objects: set[str]) -> dict:
    answer = result.get("answer", ""); expected = case["expected_objects"]
    unknown = result.get("exact_unknown", False) if system == "qwen" else result.get("decision") == "audit"
    hits = sum(_contains(answer, value) for value in expected)
    mentioned = {item for item in all_objects if _contains(answer, item)}
    context_text = " ".join(case["contexts"])
    unsupported = {item for item in mentioned if item not in expected and not _contains(context_text, item)}
    if case["expected_decision"] == "unknown": correct = unknown
    else: correct = not unknown and hits == len(expected)
    provenance = None
    if system == "microworld":
        selected = set(result.get("selected_relation_ids") or [])
        provenance = bool(selected) and selected.issubset(set(case["relation_ids"]))
    return {"correct": correct, "negative_correct": correct if case["category"] == "negative" else None,
            "object_recall": hits / len(expected) if expected else None,
            "object_precision": hits / len(mentioned) if mentioned else (1.0 if hits else None),
            "unsupported": bool(unsupported), "predicate_adherence": bool(hits == len(expected) and not unknown) if expected else unknown,
            "provenance": provenance, "latency": result.get("total_latency_ms"), "ttft": result.get("ttft_ms"),
            "tokens_per_second": result.get("tokens_per_second"), "answer_length": len(answer)}


def _average(values: list[float | bool | None]) -> float | None:
    actual = [float(value) for value in values if value is not None]
    return sum(actual) / len(actual) if actual else None


def evaluate(dataset: list[dict], microworld: list[dict], qwen: list[dict]) -> tuple[dict, list[dict]]:
    cases = {case["id"]: case for case in dataset}; all_objects = {value for case in dataset for value in case["expected_objects"]}
    rows = []
    for system, results in (("microworld", microworld), ("qwen", qwen)):
        by_category: dict[str, list[dict]] = defaultdict(list)
        for result in results:
            case = cases.get(result.get("id"))
            if case: by_category[case["category"]].append(_measure(case, result, system=system, all_objects=all_objects))
        for category, values in by_category.items():
            latency = [value["latency"] for value in values if value["latency"] is not None]
            ttft = [value["ttft"] for value in values if value["ttft"] is not None]
            rows.append({"system": "MicroWorld explicit graph runtime" if system == "microworld" else "Qwen2.5-0.5B-Instruct 4-bit",
                         "category": category, "cases": len(values), "answer_accuracy": _average([v["correct"] for v in values]),
                         "negative_accuracy": _average([v["negative_correct"] for v in values]), "object_precision": _average([v["object_precision"] for v in values]),
                         "object_recall": _average([v["object_recall"] for v in values]), "predicate_adherence": _average([v["predicate_adherence"] for v in values]),
                         "unsupported_claim_rate": _average([v["unsupported"] for v in values]), "divergence_accuracy": None,
                         "exact_evidence_provenance_accuracy": _average([v["provenance"] for v in values]),
                         "latency_p50_ms": percentile(latency, .5), "latency_p95_ms": percentile(latency, .95), "latency_p99_ms": percentile(latency, .99),
                         "ttft_p50_ms": percentile(ttft, .5), "tokens_per_second": _average([v["tokens_per_second"] for v in values]),
                         "answer_length": _average([v["answer_length"] for v in values])})
    return {"methodology": {"judge": "deterministic exact normalized surface coverage", "limitation": "unsupported-claim detection only detects known dataset objects and is not a semantic verifier."}, "rows": rows}, rows


def write_evaluation(dataset_path: str | Path, microworld_path: str | Path, qwen_path: str | Path, output: str | Path) -> dict:
    summary, rows = evaluate(read_jsonl(dataset_path), read_jsonl(microworld_path), read_jsonl(qwen_path)); output = Path(output); output.mkdir(parents=True, exist_ok=True)
    (output / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = ["system", "category", "cases", "answer_accuracy", "negative_accuracy", "object_precision", "object_recall", "predicate_adherence", "unsupported_claim_rate", "divergence_accuracy", "exact_evidence_provenance_accuracy", "answer_length", "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "ttft_p50_ms", "tokens_per_second", "startup_ms", "artifact_size_mib", "extra_memory_mib"]
    with (output / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    return summary
