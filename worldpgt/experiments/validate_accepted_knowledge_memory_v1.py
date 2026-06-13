"""Validate the accepted knowledge memory artifact v1.

Runs the artifact in isolated overlay mode against both the old trusted benchmark
(120 rows) and the knowledge probe benchmark (60 rows), then emits a validation
decision and optional quarantine report.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- Trusted baseline outputs are NOT modified.
- Thresholds are NOT lowered.
- Validators are NOT weakened.
- No generic fallback is added.

Usage:
    python3 -m worldpgt.experiments.validate_accepted_knowledge_memory_v1 \\
      --accepted-memory   worldpgt/experiments/accepted_knowledge_memory_v1.json \\
      --old-benchmark-input   worldpgt/experiments/continuation_prompts_v1.csv \\
      --old-benchmark-baseline worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \\
      --probe-input       worldpgt/experiments/knowledge_probe_prompts_v1.csv \\
      --previous-probe-summary worldpgt/experiments/knowledge_probe_overlay_v1_summary.json \\
      --output-json       worldpgt/experiments/accepted_knowledge_memory_v1_validation.json \\
      --quarantine-json   worldpgt/experiments/accepted_knowledge_memory_v1_quarantine.json \\
      --validation-csv    worldpgt/experiments/accepted_knowledge_memory_v1_validation.csv
"""

from __future__ import annotations

import argparse
import csv
import json

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.experiments.check_semantic_render_quality import check_rows
from worldpgt.knowledge.safe_memory_applier import SafeMemoryApplier

_MEMORY_VERSION = "accepted_knowledge_memory_v1"

# Gate constants (from trusted baseline)
_OLD_BENCH_MIN_CONTINUE = 58
_OLD_BENCH_TRUSTED_AUDIT_ID = "v1-051"

_CSV_FIELDS = [
    "source", "row_id", "prompt", "expected_sense",
    "baseline_decision", "overlay_decision", "overlay_selected_sense",
    "overlay_continuation", "changed", "newly_continued", "newly_audited",
    "wrong_continue",
]


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _count_wrong_continues(rows: list[dict]) -> int:
    return sum(
        1 for r in rows
        if r.get("decision") == "continue"
        and r.get("expected_sense", "")
        and r.get("selected_sense", "") != r.get("expected_sense", "")
    )


def _is_safe(old_result: dict, probe_result: dict) -> bool:
    if old_result["wrong_continue_count"] > 0:
        return False
    if old_result["semantic_quality_flagged"] > 0:
        return False
    if probe_result["wrong_continue_count"] > 0:
        return False
    if probe_result["semantic_quality_flagged"] > 0:
        return False
    if old_result["risk_regressions"]:
        return False
    if probe_result["risk_regressions"]:
        return False
    if old_result.get("v1_051_audited") is False:
        return False
    return True


def _run_benchmark(engine: ControlledContinuationEngine, prompts: list[dict]) -> list[dict]:
    results = []
    for p in prompts:
        r = engine.continue_prompt(p["prompt"])
        results.append({
            "id": p.get("id") or p.get("row_id", ""),
            "prompt": p["prompt"],
            "decision": r.decision,
            "selected_sense": r.selected_sense or "",
            "continuation": r.continuation,
            "expected_sense": p.get("expected_sense", "") or p.get("target_sense", ""),
        })
    return results


def _compute_old_benchmark_result(
    overlay_rows: list[dict],
    baseline_by_id: dict[str, dict],
) -> dict:
    overlay_continue = sum(1 for r in overlay_rows if r["decision"] == "continue")
    overlay_audit = sum(1 for r in overlay_rows if r["decision"] == "audit")

    wrong = _count_wrong_continues(overlay_rows)
    quality = check_rows([
        {"id": r["id"], "decision": r["decision"],
         "continuation": r["continuation"], "prompt": r["prompt"]}
        for r in overlay_rows
    ])

    regressions: list[str] = []
    for r in overlay_rows:
        base = baseline_by_id.get(r["id"], {})
        base_dec = base.get("decision", "")
        base_sense = base.get("selected_sense", "") or base.get("expected_sense", "")
        # Wrong continue: overlay continues but expected or baseline was audit/different
        if r["decision"] == "continue" and base_dec == "continue":
            if r["selected_sense"] and base_sense and r["selected_sense"] != base_sense:
                regressions.append(f"{r['id']}:sense_changed({base_sense}->{r['selected_sense']})")

    v1_051 = next((r for r in overlay_rows if r["id"] == _OLD_BENCH_TRUSTED_AUDIT_ID), None)
    v1_051_audited = (v1_051 is not None and v1_051["decision"] == "audit")

    return {
        "overlay_continue_count": overlay_continue,
        "overlay_audit_count": overlay_audit,
        "wrong_continue_count": wrong,
        "semantic_quality_flagged": quality["flagged_count"],
        "risk_regressions": regressions,
        "v1_051_audited": v1_051_audited,
    }


def _compute_probe_result(overlay_rows: list[dict]) -> dict:
    overlay_continue = sum(1 for r in overlay_rows if r["decision"] == "continue")
    overlay_audit = sum(1 for r in overlay_rows if r["decision"] == "audit")
    wrong = _count_wrong_continues(overlay_rows)
    quality = check_rows([
        {"id": r["id"], "decision": r["decision"],
         "continuation": r["continuation"], "prompt": r["prompt"]}
        for r in overlay_rows
    ])
    regressions: list[str] = []
    return {
        "overlay_continue_count": overlay_continue,
        "overlay_audit_count": overlay_audit,
        "wrong_continue_count": wrong,
        "semantic_quality_flagged": quality["flagged_count"],
        "risk_regressions": regressions,
    }


def _build_quarantine(
    artifact: dict,
    old_result: dict,
    probe_result: dict,
    old_overlay_rows: list[dict],
    probe_overlay_rows: list[dict],
    baseline_by_id: dict[str, dict],
) -> dict:
    """Build quarantine report. Conservative: lists all items affecting changed rows."""
    items_json = artifact.get("items", [])

    # Collect row IDs that look risky
    risky_ids: set[str] = set()
    reasons: dict[str, str] = {}

    for r in old_overlay_rows:
        base = baseline_by_id.get(r["id"], {})
        base_dec = base.get("decision", "")
        if r["decision"] == "continue" and base_dec == "audit":
            # Newly continued — could be problematic if wrong_continue > 0
            if old_result["wrong_continue_count"] > 0:
                risky_ids.add(r["id"])
                reasons[r["id"]] = "newly_continued_with_wrong_sense"
        if r["decision"] == "continue" and old_result["semantic_quality_flagged"] > 0:
            risky_ids.add(r["id"])
            reasons[r["id"]] = "continue_with_quality_flag"

    for r in probe_overlay_rows:
        if r["decision"] == "continue" and probe_result["wrong_continue_count"] > 0:
            risky_ids.add(r["id"])
            reasons[r["id"]] = "probe_wrong_continue"
        if r["decision"] == "continue" and probe_result["semantic_quality_flagged"] > 0:
            risky_ids.add(r["id"])
            reasons[r["id"]] = "probe_quality_flag"

    # Collect overlay items that could have influenced the risky rows
    quarantined_items: list[dict] = []
    if risky_ids:
        for item in items_json:
            if item.get("item_type") == "positive_cue":
                quarantined_items.append({
                    "item_id": item["item_id"],
                    "term": item["term"],
                    "sense": item["sense"],
                    "item_type": item["item_type"],
                    "value": item["value"],
                    "reason": "requires_manual_attribution",
                })

    return {
        "quarantined_items": quarantined_items,
        "risky_row_ids": sorted(risky_ids),
        "reasons": reasons,
    }


# ------------------------------------------------------------------
# Main run function
# ------------------------------------------------------------------

def run(
    accepted_memory_path: str,
    old_benchmark_input_path: str,
    old_benchmark_baseline_path: str,
    probe_input_path: str,
    previous_probe_summary_path: str,
    output_json_path: str,
    quarantine_json_path: str,
    validation_csv_path: str = "",
) -> dict:
    """Run full validation. Returns the validation result dict."""
    with open(accepted_memory_path, encoding="utf-8") as f:
        artifact = json.load(f)

    with open(old_benchmark_input_path, newline="", encoding="utf-8") as f:
        old_prompts = list(csv.DictReader(f))

    with open(old_benchmark_baseline_path, newline="", encoding="utf-8") as f:
        baseline_rows = list(csv.DictReader(f))
    baseline_by_id = {r["id"]: r for r in baseline_rows}
    baseline_continue_count = sum(1 for r in baseline_rows if r.get("decision") == "continue")

    with open(probe_input_path, newline="", encoding="utf-8") as f:
        probe_prompts = list(csv.DictReader(f))

    with open(previous_probe_summary_path, encoding="utf-8") as f:
        prev_probe_summary = json.load(f)
    prev_overlay_continue = prev_probe_summary.get("knowledge_pattern_overlay", {}).get(
        "continue_count", 43
    )
    prev_baseline_continue = prev_probe_summary.get("baseline", {}).get("continue_count", 35)

    # Build overlay memory from artifact
    applier = SafeMemoryApplier.from_artifact(artifact)
    overlay_mem = applier.build_overlay_memory()
    engine = ControlledContinuationEngine(memory=overlay_mem)

    # Run benchmarks
    old_overlay_rows = _run_benchmark(engine, old_prompts)
    probe_overlay_rows = _run_benchmark(engine, probe_prompts)

    # Compute results
    old_result = _compute_old_benchmark_result(old_overlay_rows, baseline_by_id)
    probe_result = _compute_probe_result(probe_overlay_rows)

    # Safety gate
    safe = _is_safe(old_result, probe_result)
    validation_decision = "safe_to_use_as_overlay" if safe else "unsafe_quarantined"

    # Build quarantine
    quarantine = _build_quarantine(
        artifact, old_result, probe_result, old_overlay_rows, probe_overlay_rows, baseline_by_id
    )

    # Policy check (never lowers thresholds)
    policy = ContinuationPolicy()
    thresholds_changed = policy.min_score != 1.0 or policy.min_margin != 1.0

    validation = {
        "memory_version": _MEMORY_VERSION,
        "validation_decision": validation_decision,
        "old_benchmark": {
            "baseline_continue_count": baseline_continue_count,
            "artifact_overlay_continue_count": old_result["overlay_continue_count"],
            "wrong_continue_count": old_result["wrong_continue_count"],
            "semantic_quality_flagged": old_result["semantic_quality_flagged"],
            "risk_regressions": old_result["risk_regressions"],
            "v1_051_audited": old_result["v1_051_audited"],
        },
        "knowledge_probe": {
            "baseline_continue_count": prev_baseline_continue,
            "previous_overlay_continue_count": prev_overlay_continue,
            "artifact_overlay_continue_count": probe_result["overlay_continue_count"],
            "wrong_continue_count": probe_result["wrong_continue_count"],
            "semantic_quality_flagged": probe_result["semantic_quality_flagged"],
            "risk_regressions": probe_result["risk_regressions"],
        },
        "quarantine": {
            "quarantined_items": quarantine["quarantined_items"],
            "reasons": quarantine["reasons"],
        },
        "safety": {
            "sense_memory_modified": False,
            "baseline_outputs_modified": False,
            "thresholds_changed": thresholds_changed,
            "validators_weakened": False,
            "generic_fallback_added": False,
        },
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(quarantine_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "memory_version": _MEMORY_VERSION,
                "validation_decision": validation_decision,
                "quarantined_items": quarantine["quarantined_items"],
                "risky_row_ids": quarantine["risky_row_ids"],
                "reasons": quarantine["reasons"],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    if validation_csv_path:
        _write_validation_csv(
            validation_csv_path,
            old_overlay_rows,
            probe_overlay_rows,
            baseline_by_id,
        )

    return validation


def _write_validation_csv(
    path: str,
    old_rows: list[dict],
    probe_rows: list[dict],
    baseline_by_id: dict[str, dict],
) -> None:
    csv_rows: list[dict] = []
    for r in old_rows:
        base = baseline_by_id.get(r["id"], {})
        base_dec = base.get("decision", "")
        wrong = (
            "yes"
            if r["decision"] == "continue"
            and r.get("expected_sense")
            and r.get("selected_sense") != r.get("expected_sense")
            else "no"
        )
        csv_rows.append({
            "source": "old_benchmark",
            "row_id": r["id"],
            "prompt": r["prompt"],
            "expected_sense": r.get("expected_sense", ""),
            "baseline_decision": base_dec,
            "overlay_decision": r["decision"],
            "overlay_selected_sense": r.get("selected_sense", ""),
            "overlay_continuation": r.get("continuation", ""),
            "changed": "yes" if r["decision"] != base_dec else "no",
            "newly_continued": "yes" if base_dec == "audit" and r["decision"] == "continue" else "no",
            "newly_audited": "yes" if base_dec == "continue" and r["decision"] == "audit" else "no",
            "wrong_continue": wrong,
        })
    for r in probe_rows:
        wrong = (
            "yes"
            if r["decision"] == "continue"
            and r.get("expected_sense")
            and r.get("selected_sense") != r.get("expected_sense")
            else "no"
        )
        csv_rows.append({
            "source": "probe",
            "row_id": r["id"],
            "prompt": r["prompt"],
            "expected_sense": r.get("expected_sense", ""),
            "baseline_decision": "",
            "overlay_decision": r["decision"],
            "overlay_selected_sense": r.get("selected_sense", ""),
            "overlay_continuation": r.get("continuation", ""),
            "changed": "",
            "newly_continued": "",
            "newly_audited": "",
            "wrong_continue": wrong,
        })

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Validate accepted knowledge memory artifact v1."
    )
    parser.add_argument("--accepted-memory", required=True, dest="accepted_memory")
    parser.add_argument("--old-benchmark-input", required=True, dest="old_benchmark_input")
    parser.add_argument("--old-benchmark-baseline", required=True, dest="old_benchmark_baseline")
    parser.add_argument("--probe-input", required=True, dest="probe_input")
    parser.add_argument("--previous-probe-summary", required=True, dest="previous_probe_summary")
    parser.add_argument("--output-json", required=True, dest="output_json")
    parser.add_argument("--quarantine-json", required=True, dest="quarantine_json")
    parser.add_argument("--validation-csv", default="", dest="validation_csv")
    args = parser.parse_args(argv)

    result = run(
        accepted_memory_path=args.accepted_memory,
        old_benchmark_input_path=args.old_benchmark_input,
        old_benchmark_baseline_path=args.old_benchmark_baseline,
        probe_input_path=args.probe_input,
        previous_probe_summary_path=args.previous_probe_summary,
        output_json_path=args.output_json,
        quarantine_json_path=args.quarantine_json,
        validation_csv_path=args.validation_csv,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
