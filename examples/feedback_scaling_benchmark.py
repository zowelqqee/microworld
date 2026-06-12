"""
Feedback compression scaling benchmark.

Synthetic audit rows grow from 100 to 10,000 rows while the learned trust state
aggregates into a small fixed vocabulary of relation/rule/drift/evidence
buckets.  This shows the intended shape: raw audit context grows linearly with
feedback rows, while Microworld's trust state grows by learned bucket count.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.trust_learning import TrustProfile, learn_trust_from_audits
from examples.efficiency_benchmark import (
    compute_compression_ratio,
    estimate_tokens,
)

DEFAULT_SCALES = (100, 500, 1_000, 5_000, 10_000)
FIELDNAMES = [
    "source",
    "relation_type",
    "target",
    "confidence",
    "reason",
    "evidence",
    "rule",
    "manual_label",
    "notes",
]

RELATIONS = ("made_of", "part_of", "is_a", "used_for")
RULES = (
    "made_of->made_of=>made_of",
    "part_of->part_of=>part_of",
    "is_a->is_a=>is_a",
    "is_a->used_for=>used_for",
)
DRIFTS = ("none", "raw_material", "atomic_component", "abstract_component")
EVIDENCE_NODES = ("paper", "wood", "organism", "tool", "human_audit", "relation_trust")
LABELS = ("correct", "plausible", "wrong", "unclear")


@dataclass
class ScalingResult:
    rows: int
    raw_audit_bytes: int
    raw_audit_tokens: float
    trust_profile_bytes: int
    trust_profile_tokens: float
    compression_ratio: float
    learn_time_ms: float


def generate_synthetic_audit_rows(row_count: int) -> list[dict[str, str]]:
    """Generate deterministic audit rows over a fixed bucket vocabulary."""
    rows: list[dict[str, str]] = []
    for index in range(row_count):
        relation = RELATIONS[index % len(RELATIONS)]
        rule = RULES[index % len(RULES)]
        drift = DRIFTS[index % len(DRIFTS)]
        evidence = EVIDENCE_NODES[index % len(EVIDENCE_NODES)]
        label = LABELS[(index * 3 + 1) % len(LABELS)]
        rows.append(
            {
                "source": f"synthetic_source_{index:05d}",
                "relation_type": relation,
                "target": f"synthetic_target_{index:05d}",
                "confidence": f"{0.35 + (index % 50) / 100:.2f}",
                "reason": (
                    f"synthetic audit rule={rule} drift={drift} "
                    f"support={(index % 17) + 1}/20"
                ),
                "evidence": evidence,
                "rule": rule,
                "manual_label": label,
                "notes": "synthetic feedback row",
            }
        )
    return rows


def write_synthetic_audit_csv(rows: list[dict[str, str]], path: str) -> int:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    return os.path.getsize(path)


def run_scale(row_count: int, workdir: str) -> ScalingResult:
    rows = generate_synthetic_audit_rows(row_count)
    audit_path = os.path.join(workdir, f"synthetic_audit_{row_count}.csv")
    profile_path = os.path.join(workdir, f"trust_profile_{row_count}.json")

    raw_bytes = write_synthetic_audit_csv(rows, audit_path)
    start = time.perf_counter()
    profile = learn_trust_from_audits([audit_path])
    learn_time_ms = (time.perf_counter() - start) * 1000
    profile.to_json(profile_path)
    profile_bytes = os.path.getsize(profile_path)

    raw_tokens = estimate_tokens(raw_bytes)
    profile_tokens = estimate_tokens(profile_bytes)
    return ScalingResult(
        rows=row_count,
        raw_audit_bytes=raw_bytes,
        raw_audit_tokens=raw_tokens,
        trust_profile_bytes=profile_bytes,
        trust_profile_tokens=profile_tokens,
        compression_ratio=compute_compression_ratio(raw_tokens, profile_tokens),
        learn_time_ms=learn_time_ms,
    )


def run_feedback_scaling_benchmark(
    scales: tuple[int, ...] = DEFAULT_SCALES,
) -> list[ScalingResult]:
    with tempfile.TemporaryDirectory(prefix="microworld_feedback_scaling_") as tmpdir:
        return [run_scale(scale, tmpdir) for scale in scales]


def _print_results(results: list[ScalingResult]) -> None:
    print("Feedback Compression Scaling Benchmark")
    print("======================================")
    print(
        f"{'rows':>8} | {'audit_tokens':>13} | {'trust_tokens':>12} | "
        f"{'compression':>12} | {'learn_ms':>9}"
    )
    print("-" * 68)
    for result in results:
        print(
            f"{result.rows:>8d} | "
            f"{result.raw_audit_tokens:>13.1f} | "
            f"{result.trust_profile_tokens:>12.1f} | "
            f"{result.compression_ratio:>11.2f}x | "
            f"{result.learn_time_ms:>9.3f}"
        )
    print()
    print(
        "Conclusion: Feedback rows grow linearly; learned trust state grows by "
        "number of learned buckets."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark audit-row growth against compact trust state."
    )
    parser.add_argument(
        "--scales",
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
        help="Comma-separated row counts, default: 100,500,1000,5000,10000",
    )
    args = parser.parse_args()
    scales = tuple(int(item.strip()) for item in args.scales.split(",") if item.strip())
    results = run_feedback_scaling_benchmark(scales)
    _print_results(results)


if __name__ == "__main__":
    main()
