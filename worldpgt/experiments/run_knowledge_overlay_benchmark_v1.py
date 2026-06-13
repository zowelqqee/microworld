"""Knowledge overlay dry-run benchmark v1.

Loads accepted_auto items from the auto-review artifact, builds a temporary
overlay ExplicitSenseMemory, and runs the continuation benchmark in isolation.
Compares to the committed trusted baseline and writes separate output artifacts.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- Thresholds are NOT lowered.
- Validators are NOT weakened.
- Baseline benchmark output is NOT modified.
- All output goes to separate overlay artifact files only.

Usage:
    python3 -m worldpgt.experiments.run_knowledge_overlay_benchmark_v1 \\
      --input worldpgt/experiments/continuation_prompts_v1.csv \\
      --baseline-output worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \\
      --auto-review worldpgt/experiments/knowledge_ingestion_v1_auto_review.json \\
      --output-csv worldpgt/experiments/knowledge_overlay_v1_outputs.csv \\
      --output-json worldpgt/experiments/knowledge_overlay_v1_summary.json \\
      --delta-csv worldpgt/experiments/knowledge_overlay_v1_delta.csv
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
from worldpgt.knowledge.knowledge_overlay import KnowledgeOverlay

_TRUE_UNSAFE_IDS = frozenset({
    "v1-081", "v1-082", "v1-083", "v1-085", "v1-086",
    "v1-088", "v1-089", "v1-090", "v1-091", "v1-092",
    "v1-093", "v1-094",
})
_NO_SAFE_REPAIRED_IDS = frozenset({"v1-051"})

OUTPUT_FIELDS = [
    "id", "prompt", "ambiguous_term", "expected_sense", "difficulty_type", "notes",
    "continuation", "selected_sense", "confidence", "decision", "reasons", "memory_hits",
]
DELTA_FIELDS = [
    "id", "baseline_decision", "overlay_decision",
    "baseline_selected_sense", "overlay_selected_sense",
    "changed", "risk_regression", "overlay_influenced",
    "baseline_continuation", "overlay_continuation",
]


def _engine_rows(engine: ControlledContinuationEngine, input_path: str) -> list[dict]:
    rows: list[dict] = []
    with open(input_path, newline="", encoding="utf-8") as f:
        for record in csv.DictReader(f):
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


def _compute_overlay_only_cues(
    baseline_mem: ExplicitSenseMemory,
    overlay_mem: ExplicitSenseMemory,
) -> dict[tuple[str, str], set[str]]:
    """Cues present in overlay_mem but absent in baseline_mem, per (term, sense_id)."""
    result: dict[tuple[str, str], set[str]] = {}
    for term in baseline_mem.known_terms():
        for base in baseline_mem.get_senses(term):
            ov_list = [e for e in overlay_mem.get_senses(term) if e.sense_id == base.sense_id]
            if not ov_list:
                continue
            new = set(ov_list[0].cues) - set(base.cues)
            if new:
                result[(term, base.sense_id)] = new
    return result


def _cue_trace_markers(
    row: dict,
    overlay_only: dict[tuple[str, str], set[str]],
) -> list[str]:
    """Return overlay_cue trace markers for overlay-only cues that matched in this row."""
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
        if cue in overlay_only.get((term, sense_id), set()):
            markers.append(f"overlay_cue={cue} -> {term}:{sense_id}")
    return markers


def _append_overlay_trace(
    row: dict,
    overlay: KnowledgeOverlay,
    cue_markers: list[str],
) -> None:
    """Append overlay trace markers to memory_hits in-place."""
    base_markers = overlay.overlay_trace_markers(len(cue_markers))
    all_markers = base_markers + cue_markers
    existing = row.get("memory_hits", "")
    row["memory_hits"] = (existing + " | " + " | ".join(all_markers)) if existing else " | ".join(all_markers)


def _count_wrong_continues(rows: list[dict]) -> int:
    return sum(
        1 for r in rows
        if r.get("decision") == "continue"
        and r.get("expected_sense", "")
        and r.get("selected_sense", "") != r.get("expected_sense", "")
    )


def _build_delta(
    baseline_rows: list[dict],
    overlay_rows: list[dict],
    overlay_only: dict[tuple[str, str], set[str]],
) -> tuple[list[dict], list[str], list[str], list[str], list[str]]:
    base_idx = {r["id"]: r for r in baseline_rows}
    newly_continued: list[str] = []
    newly_audited: list[str] = []
    changed_ids: list[str] = []
    regressions: list[str] = []
    delta: list[dict] = []

    for ov in overlay_rows:
        rid = ov["id"]
        b = base_idx.get(rid, {})

        b_dec = b.get("decision", "")
        o_dec = ov.get("decision", "")
        b_sense = b.get("selected_sense", "")
        o_sense = ov.get("selected_sense", "")
        b_cont = b.get("continuation", "")
        o_cont = ov.get("continuation", "")

        is_changed = b_dec != o_dec or b_sense != o_sense or b_cont != o_cont
        influenced = len(_cue_trace_markers(ov, overlay_only)) > 0

        is_regression = False
        if rid in _TRUE_UNSAFE_IDS and o_dec == "continue":
            is_regression = True
            regressions.append(f"{rid}:true_unsafe_became_continue")
        if rid in _NO_SAFE_REPAIRED_IDS and o_dec == "continue":
            is_regression = True
            regressions.append(f"{rid}:no_safe_repaired_became_continue")

        exp = ov.get("expected_sense", "")
        if o_dec == "continue" and exp and o_sense != exp:
            if b_dec == "audit" or (b_dec == "continue" and b_sense == exp):
                is_regression = True
                regressions.append(
                    f"{rid}:new_wrong_continue(expected={exp},got={o_sense})"
                )

        if b_dec == "audit" and o_dec == "continue":
            newly_continued.append(rid)
        if b_dec == "continue" and o_dec == "audit":
            newly_audited.append(rid)
        if is_changed:
            changed_ids.append(rid)

        delta.append({
            "id": rid,
            "baseline_decision": b_dec,
            "overlay_decision": o_dec,
            "baseline_selected_sense": b_sense,
            "overlay_selected_sense": o_sense,
            "changed": "yes" if is_changed else "no",
            "risk_regression": "yes" if is_regression else "no",
            "overlay_influenced": "yes" if influenced else "no",
            "baseline_continuation": b_cont,
            "overlay_continuation": o_cont,
        })

    return delta, newly_continued, newly_audited, changed_ids, regressions


def run(
    input_path: str,
    baseline_output_path: str,
    auto_review_path: str,
    output_csv_path: str,
    output_json_path: str,
    delta_csv_path: str,
) -> dict:
    """Run the overlay benchmark and write output artifacts. Returns the summary dict."""
    with open(auto_review_path, "r", encoding="utf-8") as f:
        auto_review_data = json.load(f)

    overlay = KnowledgeOverlay(auto_review_data)

    # Build both memories to compute the set of genuinely new cues
    baseline_mem = ExplicitSenseMemory(include_builtin=True)
    overlay_mem = overlay.build_overlay_memory()
    overlay_only = _compute_overlay_only_cues(baseline_mem, overlay_mem)

    # Run with the isolated overlay engine (fresh instance, no live memory mutation)
    overlay_engine = ControlledContinuationEngine(memory=overlay_mem)
    raw_rows = _engine_rows(overlay_engine, input_path)

    # Append trace markers to rows where overlay-only cues contributed scoring evidence
    overlay_rows: list[dict] = []
    for row in raw_rows:
        cue_markers = _cue_trace_markers(row, overlay_only)
        if cue_markers:
            _append_overlay_trace(row, overlay, cue_markers)
        overlay_rows.append(row)

    # Load baseline for comparison (read-only; this file is never opened for writing)
    with open(baseline_output_path, newline="", encoding="utf-8") as f:
        baseline_rows = list(csv.DictReader(f))

    ov_stats = summarize_rows(overlay_rows)
    ov_quality = check_rows(overlay_rows)
    base_quality = check_rows(baseline_rows)

    b_continue = sum(1 for r in baseline_rows if r.get("decision") == "continue")
    b_audit = sum(1 for r in baseline_rows if r.get("decision") == "audit")
    b_wrong = _count_wrong_continues(baseline_rows)
    b_flagged = base_quality["flagged_count"]

    o_continue = ov_stats["continue_count"]
    o_audit = ov_stats["audit_count"]
    o_wrong = ov_stats["wrong_continue_count"]
    o_flagged = ov_quality["flagged_count"]

    delta_rows, newly_cont, newly_aud, changed_ids, regressions = _build_delta(
        baseline_rows, overlay_rows, overlay_only
    )

    ov_by_id = {r["id"]: r for r in overlay_rows}
    true_unsafe_ok = all(
        ov_by_id.get(uid, {}).get("decision") == "audit" for uid in _TRUE_UNSAFE_IDS
    )
    no_safe_ok = all(
        ov_by_id.get(uid, {}).get("decision") == "audit" for uid in _NO_SAFE_REPAIRED_IDS
    )

    policy = ContinuationPolicy()
    thresholds_changed = policy.min_score != 1.0 or policy.min_margin != 1.0

    summary: dict = {
        "baseline": {
            "continue_count": b_continue,
            "audit_count": b_audit,
            "wrong_continue_count": b_wrong,
            "semantic_quality_flagged": b_flagged,
        },
        "overlay": {
            "continue_count": o_continue,
            "audit_count": o_audit,
            "wrong_continue_count": o_wrong,
            "semantic_quality_flagged": o_flagged,
        },
        "delta": {
            "continue_count": o_continue - b_continue,
            "audit_count": o_audit - b_audit,
            "newly_continued_rows": sorted(newly_cont),
            "newly_audited_rows": sorted(newly_aud),
            "changed_output_rows": sorted(changed_ids),
            "risk_regressions": regressions,
        },
        "safety": {
            "thresholds_changed": thresholds_changed,
            "validators_weakened": False,
            "sense_memory_modified": False,
            "trusted_baseline_modified": False,
            "true_unsafe_rows_still_audited": true_unsafe_ok,
            "no_safe_repaired_candidate_rows_still_audited": no_safe_ok,
        },
        "overlay_stats": overlay.stats,
    }

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(overlay_rows)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(delta_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DELTA_FIELDS)
        writer.writeheader()
        writer.writerows(delta_rows)

    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Knowledge overlay dry-run benchmark v1.")
    parser.add_argument("--input", required=True, help="Input v1 continuation prompts CSV")
    parser.add_argument(
        "--baseline-output", required=True, dest="baseline_output",
        help="Trusted baseline output CSV (microworld_continuation_v1_2_outputs.csv)",
    )
    parser.add_argument(
        "--auto-review", required=True, dest="auto_review",
        help="knowledge_ingestion_v1_auto_review.json",
    )
    parser.add_argument("--output-csv", required=True, dest="output_csv",
                        help="Overlay benchmark output CSV")
    parser.add_argument("--output-json", required=True, dest="output_json",
                        help="Overlay summary JSON")
    parser.add_argument("--delta-csv", required=True, dest="delta_csv",
                        help="Baseline vs overlay delta CSV")
    args = parser.parse_args(argv)

    summary = run(
        input_path=args.input,
        baseline_output_path=args.baseline_output,
        auto_review_path=args.auto_review,
        output_csv_path=args.output_csv,
        output_json_path=args.output_json,
        delta_csv_path=args.delta_csv,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
