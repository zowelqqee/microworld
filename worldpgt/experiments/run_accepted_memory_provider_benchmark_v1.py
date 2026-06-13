"""AcceptedKnowledgeMemoryProvider benchmark v1.

Runs both the old continuation benchmark and the knowledge probe under:
  A. Default baseline (builtin ExplicitSenseMemory, unchanged)
  B. AcceptedKnowledgeMemoryProvider overlay mode

The default engine behavior is NOT changed. The provider is used only in
the explicit overlay path of this runner.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- Default trusted benchmark outputs are NOT modified.
- Validators are NOT weakened.
- Thresholds are NOT lowered.
- No generic fallback is added.
- No neural weights, GPT renderer, or training.
- nanogpt/ is untouched.

Usage::

    python3 -m worldpgt.experiments.run_accepted_memory_provider_benchmark_v1 \\
      --accepted-memory worldpgt/experiments/accepted_knowledge_memory_v1.json \\
      --old-input worldpgt/experiments/continuation_prompts_v1.csv \\
      --old-baseline worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \\
      --probe-input worldpgt/experiments/knowledge_probe_prompts_v1.csv \\
      --previous-probe-summary worldpgt/experiments/knowledge_probe_overlay_v1_summary.json \\
      --old-output worldpgt/experiments/accepted_memory_provider_v1_old_outputs.csv \\
      --probe-output worldpgt/experiments/accepted_memory_provider_v1_probe_outputs.csv \\
      --summary-json worldpgt/experiments/accepted_memory_provider_v1_summary.json \\
      --delta-csv worldpgt/experiments/accepted_memory_provider_v1_delta.csv
"""

from __future__ import annotations

import argparse
import csv
import json

from worldpgt.continuation.continuation_engine import ControlledContinuationEngine
from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.experiments.check_semantic_render_quality import check_rows
from worldpgt.experiments.risk_coverage_metrics import summarize_rows
from worldpgt.knowledge.accepted_memory_provider import (
    MEMORY_VERSION as _PROVIDER_MEMORY_VERSION,
    AcceptedKnowledgeMemoryProvider,
)

_TRUE_UNSAFE_IDS = frozenset({
    "v1-081", "v1-082", "v1-083", "v1-085", "v1-086",
    "v1-088", "v1-089", "v1-090", "v1-091", "v1-092",
    "v1-093", "v1-094",
})
_NO_SAFE_REPAIRED_IDS = frozenset({"v1-051"})

_OLD_OUTPUT_FIELDS = [
    "id", "prompt", "ambiguous_term", "expected_sense", "difficulty_type", "notes",
    "continuation", "selected_sense", "confidence", "decision", "reasons", "memory_hits",
]
_PROBE_OUTPUT_FIELDS = [
    "row_id", "term", "target_sense", "prompt",
    "expected_safe_behavior", "expected_cue", "risk_level",
    "baseline_decision", "baseline_selected_sense", "baseline_continuation",
    "provider_decision", "provider_selected_sense", "provider_continuation",
    "baseline_reasons", "provider_reasons",
    "provider_memory_hits",
    "changed", "newly_continued", "newly_audited", "provider_influenced",
]
_DELTA_FIELDS = [
    "id", "baseline_decision", "provider_decision",
    "baseline_selected_sense", "provider_selected_sense",
    "changed", "risk_regression", "provider_influenced",
    "baseline_continuation", "provider_continuation",
]


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def _engine_rows_from_csv(engine: ControlledContinuationEngine, input_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(input_path, newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            prompt = record.get("prompt", "")
            result = engine.continue_prompt(prompt)
            rows.append({
                "id": record.get("id", ""),
                "prompt": prompt,
                "ambiguous_term": record.get("ambiguous_term", "") or result.ambiguous_term or "",
                "expected_sense": record.get("expected_sense", "") or "",
                "difficulty_type": record.get("difficulty_type", ""),
                "notes": record.get("notes", ""),
                "continuation": result.continuation,
                "selected_sense": result.selected_sense or "",
                "confidence": f"{result.confidence:.4f}",
                "decision": result.decision,
                "reasons": " | ".join(result.reasons),
                "memory_hits": " | ".join(result.memory_hits),
            })
    return rows


def _engine_rows_from_probe_csv(
    engine: ControlledContinuationEngine,
    probe_rows: list[dict],
) -> list[dict]:
    results: list[dict] = []
    for row in probe_rows:
        prompt = row["prompt"]
        result = engine.continue_prompt(prompt)
        results.append({
            "id": row["row_id"],
            "decision": result.decision,
            "selected_sense": result.selected_sense or "",
            "continuation": result.continuation,
            "reasons": " | ".join(result.reasons),
            "memory_hits": " | ".join(result.memory_hits),
            "expected_sense": row.get("target_sense", ""),
        })
    return results


def _count_wrong_continues(rows: list[dict]) -> int:
    return sum(
        1 for r in rows
        if r.get("decision") == "continue"
        and r.get("expected_sense", "")
        and r.get("selected_sense", "") != r.get("expected_sense", "")
    )


def _provider_influenced_cues(
    row: dict,
    provider_new_cues: dict[tuple[str, str], set[str]],
) -> list[str]:
    """Return provider-trace markers for any provider-only cues that matched in this row."""
    term = row.get("ambiguous_term", "")
    if not term:
        return []
    markers: list[str] = []
    for part in row.get("memory_hits", "").split(" | "):
        if not part.startswith("positive_cue="):
            continue
        rest = part[len("positive_cue="):]
        if " -> " not in rest:
            continue
        cue, sense_id = rest.split(" -> ", 1)
        if cue in provider_new_cues.get((term, sense_id), set()):
            markers.append(f"provider_cue={cue} -> {term}:{sense_id}")
    return markers


def _compute_provider_new_cues(
    baseline_mem: ExplicitSenseMemory,
    provider: AcceptedKnowledgeMemoryProvider,
) -> dict[tuple[str, str], set[str]]:
    """Map (term, sense_id) -> set of cues that provider adds beyond builtin."""
    result: dict[tuple[str, str], set[str]] = {}
    for term in baseline_mem.known_terms():
        for entry in baseline_mem.get_senses(term):
            new = provider.new_cues_for(baseline_mem, term, entry.sense_id)
            if new:
                result[(term, entry.sense_id)] = new
    return result


def _append_provider_trace(
    row: dict,
    provider: AcceptedKnowledgeMemoryProvider,
    cue_markers: list[str],
) -> None:
    base = provider.provider_trace_markers()
    all_markers = base + cue_markers
    existing = row.get("memory_hits", "")
    row["memory_hits"] = (existing + " | " + " | ".join(all_markers)) if existing else " | ".join(all_markers)


# ------------------------------------------------------------------
# Old benchmark
# ------------------------------------------------------------------

def _run_old_benchmark(
    provider: AcceptedKnowledgeMemoryProvider,
    input_path: str,
    baseline_output_path: str,
    output_csv_path: str,
    delta_csv_path: str,
) -> dict:
    baseline_mem = ExplicitSenseMemory(include_builtin=True)
    overlay_mem = provider.build_overlay_memory()
    provider_new_cues = _compute_provider_new_cues(baseline_mem, provider)

    provider_engine = ControlledContinuationEngine(memory=overlay_mem)
    raw_rows = _engine_rows_from_csv(provider_engine, input_path)

    # Append provider trace markers where provider cues influenced output
    provider_rows: list[dict] = []
    for row in raw_rows:
        cue_markers = _provider_influenced_cues(row, provider_new_cues)
        if cue_markers:
            _append_provider_trace(row, provider, cue_markers)
        provider_rows.append(row)

    with open(baseline_output_path, newline="", encoding="utf-8") as fh:
        baseline_rows = list(csv.DictReader(fh))

    ov_stats = summarize_rows(provider_rows)
    ov_quality = check_rows(provider_rows)
    base_quality = check_rows(baseline_rows)

    b_continue = sum(1 for r in baseline_rows if r.get("decision") == "continue")
    b_audit = sum(1 for r in baseline_rows if r.get("decision") == "audit")
    b_wrong = _count_wrong_continues(baseline_rows)
    b_flagged = base_quality["flagged_count"]

    o_continue = ov_stats["continue_count"]
    o_audit = ov_stats["audit_count"]
    o_wrong = ov_stats["wrong_continue_count"]
    o_flagged = ov_quality["flagged_count"]

    base_idx = {r["id"]: r for r in baseline_rows}
    newly_continued: list[str] = []
    newly_audited: list[str] = []
    changed_ids: list[str] = []
    regressions: list[str] = []
    delta_rows: list[dict] = []

    prov_by_id = {r["id"]: r for r in provider_rows}

    for prow in provider_rows:
        rid = prow["id"]
        b = base_idx.get(rid, {})
        b_dec = b.get("decision", "")
        p_dec = prow.get("decision", "")
        b_sense = b.get("selected_sense", "")
        p_sense = prow.get("selected_sense", "")
        b_cont = b.get("continuation", "")
        p_cont = prow.get("continuation", "")

        is_changed = b_dec != p_dec or b_sense != p_sense or b_cont != p_cont
        influenced = bool(_provider_influenced_cues(prow, provider_new_cues))

        is_regression = False
        if rid in _TRUE_UNSAFE_IDS and p_dec == "continue":
            is_regression = True
            regressions.append(f"{rid}:true_unsafe_became_continue")
        if rid in _NO_SAFE_REPAIRED_IDS and p_dec == "continue":
            is_regression = True
            regressions.append(f"{rid}:no_safe_repaired_became_continue")

        exp = prow.get("expected_sense", "")
        if p_dec == "continue" and exp and p_sense != exp:
            if b_dec == "audit" or (b_dec == "continue" and b_sense == exp):
                is_regression = True
                regressions.append(f"{rid}:new_wrong_continue(expected={exp},got={p_sense})")

        if b_dec == "audit" and p_dec == "continue":
            newly_continued.append(rid)
        if b_dec == "continue" and p_dec == "audit":
            newly_audited.append(rid)
        if is_changed:
            changed_ids.append(rid)

        delta_rows.append({
            "id": rid,
            "baseline_decision": b_dec,
            "provider_decision": p_dec,
            "baseline_selected_sense": b_sense,
            "provider_selected_sense": p_sense,
            "changed": "yes" if is_changed else "no",
            "risk_regression": "yes" if is_regression else "no",
            "provider_influenced": "yes" if influenced else "no",
            "baseline_continuation": b_cont,
            "provider_continuation": p_cont,
        })

    true_unsafe_ok = all(
        prov_by_id.get(uid, {}).get("decision") == "audit" for uid in _TRUE_UNSAFE_IDS
    )
    no_safe_ok = all(
        prov_by_id.get(uid, {}).get("decision") == "audit" for uid in _NO_SAFE_REPAIRED_IDS
    )

    policy = ContinuationPolicy()

    with open(output_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_OLD_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(provider_rows)

    with open(delta_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_DELTA_FIELDS)
        writer.writeheader()
        writer.writerows(delta_rows)

    return {
        "baseline_continue_count": b_continue,
        "baseline_audit_count": b_audit,
        "baseline_wrong_continue_count": b_wrong,
        "baseline_semantic_quality_flagged": b_flagged,
        "provider_continue_count": o_continue,
        "provider_audit_count": o_audit,
        "wrong_continue_count": o_wrong,
        "semantic_quality_flagged": o_flagged,
        "risk_regressions": regressions,
        "newly_continued_rows": sorted(newly_continued),
        "newly_audited_rows": sorted(newly_audited),
        "changed_output_rows": sorted(changed_ids),
        "true_unsafe_rows_still_audited": true_unsafe_ok,
        "no_safe_repaired_candidate_rows_still_audited": no_safe_ok,
        "thresholds_changed": policy.min_score != 1.0 or policy.min_margin != 1.0,
    }


# ------------------------------------------------------------------
# Knowledge probe benchmark
# ------------------------------------------------------------------

def _run_probe_benchmark(
    provider: AcceptedKnowledgeMemoryProvider,
    probe_path: str,
    output_csv_path: str,
    previous_summary_path: str,
) -> dict:
    with open(probe_path, newline="", encoding="utf-8") as fh:
        probe_rows = list(csv.DictReader(fh))

    baseline_mem = ExplicitSenseMemory(include_builtin=True)
    baseline_engine = ControlledContinuationEngine(memory=baseline_mem)

    overlay_mem = provider.build_overlay_memory()
    provider_engine = ControlledContinuationEngine(memory=overlay_mem)

    provider_new_cues = _compute_provider_new_cues(baseline_mem, provider)

    baseline_results = _engine_rows_from_probe_csv(baseline_engine, probe_rows)
    provider_results_raw = _engine_rows_from_probe_csv(provider_engine, probe_rows)

    out_rows: list[dict] = []
    newly_continued: list[str] = []
    newly_audited: list[str] = []
    changed_ids: list[str] = []
    regressions: list[str] = []

    for b_r, p_r_raw, probe in zip(baseline_results, provider_results_raw, probe_rows):
        rid = probe["row_id"]
        b_dec = b_r["decision"]
        p_dec = p_r_raw["decision"]

        # Build cue trace markers for provider-influenced rows
        cue_markers: list[str] = []
        for part in p_r_raw["memory_hits"].split(" | "):
            if part.startswith("positive_cue="):
                rest = part[len("positive_cue="):]
                if " -> " in rest:
                    cue, sense_id = rest.split(" -> ", 1)
                    for (term, sid), new in provider_new_cues.items():
                        if sid == sense_id and cue in new:
                            cue_markers.append(f"provider_cue={cue} -> {term}:{sid}")
                            break

        provider_memory_hits = p_r_raw["memory_hits"]
        if cue_markers:
            base_trace = provider.provider_trace_markers()
            all_trace = base_trace + cue_markers
            provider_memory_hits = (
                (provider_memory_hits + " | " + " | ".join(all_trace))
                if provider_memory_hits
                else " | ".join(all_trace)
            )

        provider_influenced = bool(cue_markers)
        changed = b_dec != p_dec or b_r["selected_sense"] != p_r_raw["selected_sense"]

        exp_sense = probe.get("target_sense", "")
        if p_dec == "continue" and exp_sense and p_r_raw["selected_sense"] != exp_sense:
            regressions.append(
                f"{rid}:wrong_sense(expected={exp_sense},got={p_r_raw['selected_sense']})"
            )

        if b_dec == "audit" and p_dec == "continue":
            newly_continued.append(rid)
        if b_dec == "continue" and p_dec == "audit":
            newly_audited.append(rid)
        if changed:
            changed_ids.append(rid)

        out_rows.append({
            "row_id": rid,
            "term": probe.get("term", ""),
            "target_sense": probe.get("target_sense", ""),
            "prompt": probe.get("prompt", ""),
            "expected_safe_behavior": probe.get("expected_safe_behavior", ""),
            "expected_cue": probe.get("expected_cue", ""),
            "risk_level": probe.get("risk_level", ""),
            "baseline_decision": b_dec,
            "baseline_selected_sense": b_r["selected_sense"],
            "baseline_continuation": b_r["continuation"],
            "provider_decision": p_dec,
            "provider_selected_sense": p_r_raw["selected_sense"],
            "provider_continuation": p_r_raw["continuation"],
            "baseline_reasons": b_r["reasons"],
            "provider_reasons": p_r_raw["reasons"],
            "provider_memory_hits": provider_memory_hits,
            "changed": "yes" if changed else "no",
            "newly_continued": "yes" if (b_dec == "audit" and p_dec == "continue") else "no",
            "newly_audited": "yes" if (b_dec == "continue" and p_dec == "audit") else "no",
            "provider_influenced": "yes" if provider_influenced else "no",
        })

    baseline_for_metrics = [
        {"id": r["id"], "decision": r["decision"], "selected_sense": r["selected_sense"],
         "continuation": r["continuation"], "expected_sense": r["expected_sense"]}
        for r in baseline_results
    ]
    provider_for_metrics = [
        {"id": row["row_id"], "decision": row["provider_decision"],
         "selected_sense": row["provider_selected_sense"],
         "continuation": row["provider_continuation"],
         "expected_sense": row["target_sense"],
         "prompt": row["prompt"]}
        for row in out_rows
    ]

    b_stats = summarize_rows(baseline_for_metrics)
    p_stats = summarize_rows(provider_for_metrics)
    b_quality = check_rows(baseline_for_metrics)
    p_quality = check_rows(provider_for_metrics)
    b_wrong = _count_wrong_continues(baseline_for_metrics)
    p_wrong = _count_wrong_continues(provider_for_metrics)

    previous_overlay_continue = 0
    try:
        with open(previous_summary_path, encoding="utf-8") as fh:
            prev = json.load(fh)
        previous_overlay_continue = (
            prev.get("knowledge_pattern_overlay", {}).get("continue_count", 0)
        )
    except (OSError, KeyError, json.JSONDecodeError):
        pass

    with open(output_csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_PROBE_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "baseline_continue_count": b_stats["continue_count"],
        "previous_overlay_continue_count": previous_overlay_continue,
        "provider_continue_count": p_stats["continue_count"],
        "provider_audit_count": p_stats["audit_count"],
        "wrong_continue_count": p_wrong,
        "semantic_quality_flagged": p_quality["flagged_count"],
        "risk_regressions": regressions,
        "newly_continued_rows": sorted(newly_continued),
        "newly_audited_rows": sorted(newly_audited),
        "changed_output_rows": sorted(changed_ids),
    }


# ------------------------------------------------------------------
# Main runner
# ------------------------------------------------------------------

def run(
    accepted_memory_path: str,
    old_input_path: str,
    old_baseline_path: str,
    probe_input_path: str,
    previous_probe_summary_path: str,
    old_output_path: str,
    probe_output_path: str,
    summary_json_path: str,
    delta_csv_path: str,
) -> dict:
    """Run both benchmarks and write all output artifacts. Returns the summary dict."""
    provider = AcceptedKnowledgeMemoryProvider(accepted_memory_path)

    old_result = _run_old_benchmark(
        provider=provider,
        input_path=old_input_path,
        baseline_output_path=old_baseline_path,
        output_csv_path=old_output_path,
        delta_csv_path=delta_csv_path,
    )

    probe_result = _run_probe_benchmark(
        provider=provider,
        probe_path=probe_input_path,
        output_csv_path=probe_output_path,
        previous_summary_path=previous_probe_summary_path,
    )

    policy = ContinuationPolicy()

    summary: dict = {
        "memory_version": _PROVIDER_MEMORY_VERSION,
        "mode": "read_only_provider",
        "default_behavior_changed": False,
        "old_benchmark": {
            "baseline_continue_count": old_result["baseline_continue_count"],
            "provider_continue_count": old_result["provider_continue_count"],
            "wrong_continue_count": old_result["wrong_continue_count"],
            "semantic_quality_flagged": old_result["semantic_quality_flagged"],
            "risk_regressions": old_result["risk_regressions"],
        },
        "knowledge_probe": {
            "baseline_continue_count": probe_result["baseline_continue_count"],
            "previous_overlay_continue_count": probe_result["previous_overlay_continue_count"],
            "provider_continue_count": probe_result["provider_continue_count"],
            "wrong_continue_count": probe_result["wrong_continue_count"],
            "semantic_quality_flagged": probe_result["semantic_quality_flagged"],
            "risk_regressions": probe_result["risk_regressions"],
        },
        "provider_stats": provider.stats,
        "safety": {
            "sense_memory_modified": False,
            "thresholds_changed": policy.min_score != 1.0 or policy.min_margin != 1.0,
            "validators_weakened": False,
            "generic_fallback_added": False,
        },
    }

    with open(summary_json_path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="AcceptedKnowledgeMemoryProvider benchmark v1."
    )
    parser.add_argument(
        "--accepted-memory", required=True, dest="accepted_memory",
        help="Path to accepted_knowledge_memory_v1.json",
    )
    parser.add_argument(
        "--old-input", required=True, dest="old_input",
        help="continuation_prompts_v1.csv",
    )
    parser.add_argument(
        "--old-baseline", required=True, dest="old_baseline",
        help="microworld_continuation_v1_2_outputs.csv (trusted baseline, read-only)",
    )
    parser.add_argument(
        "--probe-input", required=True, dest="probe_input",
        help="knowledge_probe_prompts_v1.csv",
    )
    parser.add_argument(
        "--previous-probe-summary", required=True, dest="previous_probe_summary",
        help="knowledge_probe_overlay_v1_summary.json",
    )
    parser.add_argument(
        "--old-output", required=True, dest="old_output",
        help="accepted_memory_provider_v1_old_outputs.csv",
    )
    parser.add_argument(
        "--probe-output", required=True, dest="probe_output",
        help="accepted_memory_provider_v1_probe_outputs.csv",
    )
    parser.add_argument(
        "--summary-json", required=True, dest="summary_json",
        help="accepted_memory_provider_v1_summary.json",
    )
    parser.add_argument(
        "--delta-csv", required=True, dest="delta_csv",
        help="accepted_memory_provider_v1_delta.csv (optional delta)",
    )
    args = parser.parse_args(argv)

    summary = run(
        accepted_memory_path=args.accepted_memory,
        old_input_path=args.old_input,
        old_baseline_path=args.old_baseline,
        probe_input_path=args.probe_input,
        previous_probe_summary_path=args.previous_probe_summary,
        old_output_path=args.old_output,
        probe_output_path=args.probe_output,
        summary_json_path=args.summary_json,
        delta_csv_path=args.delta_csv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
