"""Contract tests for the answer-behavior performance benchmark."""

from __future__ import annotations

from worldpgt.benchmarks.answer_behavior_benchmark import (
    run_depth_case,
    run_local_bundle_case,
    run_storage_case,
)


def test_storage_case_reports_latency_memory_and_stage_breakdown():
    result = run_storage_case(20, repetitions=3, target_degree=5)

    assert result["store_items"] == 20
    assert result["index_build"]["kind"] == "prepared_evidence_graph"
    assert {"p50_ms", "p95_ms", "p99_ms"} <= result["warm_query"].keys()
    breakdown = result["warm_query"]["median_breakdown"]
    assert {"entity_resolution_ms", "candidate_fetch_ms", "scoring_ms",
            "diversity_penalty_ms", "plan_assembly_ms", "rendering_ms"} <= breakdown.keys()
    assert breakdown["selected_steps"] >= 1
    assert result["peak_python_alloc_mb"] >= 0


def test_local_bundle_records_hot_node_candidate_pressure():
    result = run_local_bundle_case(12, repetitions=3)
    breakdown = result["warm_query"]["median_breakdown"]

    assert result["target_degree"] == 12
    assert breakdown["candidate_evaluations"] >= 12
    assert breakdown["candidates_discarded"] > 0
    assert 0.0 <= breakdown["branch_pruning_rate"] <= 1.0


def test_depth_case_records_graph_traversal_counts():
    result = run_depth_case(3, repetitions=3)
    breakdown = result["warm_query"]["median_breakdown"]

    assert result["max_depth"] == 3
    assert breakdown["selected_steps"] == 3
    assert breakdown["visited_nodes"] >= 4
    assert breakdown["visited_edges"] >= 3
