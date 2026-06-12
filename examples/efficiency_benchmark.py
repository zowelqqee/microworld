"""
Efficiency benchmark for Microworld's explicit audit-learning loop.

No LLM API calls are made.  The "context baseline" is an estimate of how many
tokens would be replayed if the same audit feedback were sent as prompt context
on every query instead of being compacted into an explicit trust profile.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.datasets import build_world_from_relations, load_relations_csv
from core.pattern_prediction import PatternBasedPredictor, PatternPrediction
from core.relation_trust import DEFAULT_RELATION_TRUST
from core.trust_learning import TrustProfile, learn_trust_from_audits

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DATA_PATH = os.path.join(_ROOT, "data", "conceptnet_sample.csv")
AUDIT_PATHS = [
    os.path.join(_ROOT, "data", "audit_all.csv"),
    os.path.join(_ROOT, "data", "audit_mixed_out.csv"),
    os.path.join(_ROOT, "data", "audit_drift_aware.csv"),
]


@dataclass
class EfficiencyBenchmarkResult:
    audit_rows: int
    audit_csv_bytes: int
    audit_context_tokens: float
    audit_context_tokens_repeated: float
    trust_profile_bytes: int
    trust_profile_tokens: float
    trust_profile_tokens_repeated: float
    compression_ratio: float
    update_time_ms: float
    prediction_time_ms: float
    baseline_total_predictions: int
    learned_total_predictions: int
    baseline_accepted: int
    learned_accepted: int
    suppressed: int
    repeated_queries: int
    threshold: float


def estimate_tokens(byte_count: int) -> float:
    """Deterministic rough token estimate used for context-size comparison."""
    return byte_count / 4.0


def compute_compression_ratio(
    audit_context_tokens: float,
    trust_profile_tokens: float,
) -> float:
    if trust_profile_tokens <= 0:
        return 0.0
    return audit_context_tokens / trust_profile_tokens


def audit_csv_bytes(paths: list[str]) -> int:
    total = 0
    for path in paths:
        if not os.path.exists(path):
            continue
        with open(path, "rb") as f:
            total += len(f.read())
    return total


def trust_profile_json_bytes(profile: TrustProfile) -> int:
    text = json.dumps(asdict(profile), indent=2, sort_keys=True) + "\n"
    return len(text.encode("utf-8"))


def merge_learned_relation_trust(
    baseline_trust: dict[str, float],
    learned_trust: dict[str, float],
) -> dict[str, float]:
    merged = dict(baseline_trust)
    merged.update(learned_trust)
    return merged


def accepted_count(
    predictions: list[PatternPrediction],
    threshold: float,
) -> int:
    return sum(1 for prediction in predictions if prediction.confidence >= threshold)


def run_predictions(
    relations,
    relation_trust: dict[str, float],
) -> list[PatternPrediction]:
    return PatternBasedPredictor(relations).predict_from_bigrams(
        min_count=5,
        min_confidence=0.0,
        hub_penalty=True,
        relation_trust=relation_trust,
        use_relation_drift=True,
    )


def run_efficiency_benchmark(
    data_path: str = DATA_PATH,
    audit_paths: list[str] | None = None,
    threshold: float = 0.4,
    repeated_queries: int = 100,
) -> EfficiencyBenchmarkResult:
    paths = audit_paths or AUDIT_PATHS
    existing_audits = [path for path in paths if os.path.exists(path)]

    update_start = time.perf_counter()
    profile = learn_trust_from_audits(existing_audits)
    update_time_ms = (time.perf_counter() - update_start) * 1000

    rows = load_relations_csv(data_path)
    world = build_world_from_relations(rows)
    relations = world.get_relations()
    learned_trust = merge_learned_relation_trust(
        DEFAULT_RELATION_TRUST,
        profile.relation_trust,
    )

    prediction_start = time.perf_counter()
    baseline_predictions = run_predictions(relations, DEFAULT_RELATION_TRUST)
    learned_predictions = run_predictions(relations, learned_trust)
    prediction_time_ms = (time.perf_counter() - prediction_start) * 1000

    audit_bytes = audit_csv_bytes(existing_audits)
    profile_bytes = trust_profile_json_bytes(profile)
    audit_tokens = estimate_tokens(audit_bytes)
    profile_tokens = estimate_tokens(profile_bytes)
    baseline_accepted = accepted_count(baseline_predictions, threshold)
    learned_accepted = accepted_count(learned_predictions, threshold)

    return EfficiencyBenchmarkResult(
        audit_rows=int(profile.counts.get("used_rows", 0)),
        audit_csv_bytes=audit_bytes,
        audit_context_tokens=audit_tokens,
        audit_context_tokens_repeated=audit_tokens * repeated_queries,
        trust_profile_bytes=profile_bytes,
        trust_profile_tokens=profile_tokens,
        trust_profile_tokens_repeated=profile_tokens * repeated_queries,
        compression_ratio=compute_compression_ratio(audit_tokens, profile_tokens),
        update_time_ms=update_time_ms,
        prediction_time_ms=prediction_time_ms,
        baseline_total_predictions=len(baseline_predictions),
        learned_total_predictions=len(learned_predictions),
        baseline_accepted=baseline_accepted,
        learned_accepted=learned_accepted,
        suppressed=max(0, baseline_accepted - learned_accepted),
        repeated_queries=repeated_queries,
        threshold=threshold,
    )


def _print_result(result: EfficiencyBenchmarkResult) -> None:
    print("Microworld Efficiency Benchmark")
    print("===============================")
    print(f"threshold                         : {result.threshold:.2f}")
    print(f"repeated queries                  : {result.repeated_queries}")
    print(f"audit rows                        : {result.audit_rows}")
    print(f"audit CSV bytes                   : {result.audit_csv_bytes}")
    print(f"audit_context_tokens              : {result.audit_context_tokens:.1f}")
    print(
        "audit_context_tokens repeated    : "
        f"{result.audit_context_tokens_repeated:.1f}"
    )
    print(f"trust_profile.json bytes          : {result.trust_profile_bytes}")
    print(f"trust_profile_tokens              : {result.trust_profile_tokens:.1f}")
    print(
        "trust_profile_tokens repeated    : "
        f"{result.trust_profile_tokens_repeated:.1f}"
    )
    print(f"compression_ratio                 : {result.compression_ratio:.2f}x")
    print(f"update_time_ms                    : {result.update_time_ms:.3f}")
    print(f"prediction_time_ms                : {result.prediction_time_ms:.3f}")
    print(f"baseline total predictions        : {result.baseline_total_predictions}")
    print(f"learned total predictions         : {result.learned_total_predictions}")
    print(f"baseline accepted                 : {result.baseline_accepted}")
    print(f"learned accepted                  : {result.learned_accepted}")
    print(f"suppressed                        : {result.suppressed}")
    print()
    print(
        "Conclusion: Microworld stores feedback as compact trust state instead "
        "of replaying audit context."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark explicit trust learning against replayed audit context."
    )
    parser.add_argument("--input", default=DATA_PATH)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--queries", type=int, default=100)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    result = run_efficiency_benchmark(
        data_path=args.input,
        threshold=args.threshold,
        repeated_queries=args.queries,
    )
    _print_result(result)


if __name__ == "__main__":
    main()
