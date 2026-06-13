"""Run the knowledge probe overlay benchmark v1.

Runs the 60-row probe benchmark under both the baseline and the
knowledge-pattern overlay, then writes comparison artifacts.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- Thresholds are NOT lowered.
- Validators are NOT weakened.
- Trusted baseline outputs are NOT modified.
- No generic fallback is added.
- No neural weights, GPT renderer, or training.

Usage:
    python3 -m worldpgt.experiments.run_knowledge_probe_overlay_v1 \\
      --probe    worldpgt/experiments/knowledge_probe_prompts_v1.csv \\
      --patterns worldpgt/experiments/wiki_pattern_candidates_v1.json \\
      --auto-review worldpgt/experiments/knowledge_ingestion_v1_auto_review.json \\
      --output-csv  worldpgt/experiments/knowledge_probe_overlay_v1_outputs.csv \\
      --output-json worldpgt/experiments/knowledge_probe_overlay_v1_summary.json
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

_OUTPUT_FIELDS = [
    "row_id", "term", "target_sense", "prompt",
    "expected_safe_behavior", "expected_cue", "risk_level",
    "baseline_decision", "baseline_selected_sense", "baseline_continuation",
    "overlay_decision", "overlay_selected_sense", "overlay_continuation",
    "baseline_reasons", "overlay_reasons",
    "overlay_memory_hits",
    "changed", "newly_continued", "newly_audited", "overlay_influenced",
]


def _run_engine(engine: ControlledContinuationEngine, probe_rows: list[dict]) -> list[dict]:
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
            "expected_sense": row["target_sense"],
        })
    return results


def _has_overlay_cue(memory_hits: str) -> bool:
    return "overlay_cue=" in memory_hits


def _count_wrong_continues(rows: list[dict]) -> int:
    return sum(
        1 for r in rows
        if r.get("decision") == "continue"
        and r.get("expected_sense", "")
        and r.get("selected_sense", "") != r.get("expected_sense", "")
    )


def _build_overlay_only_cues(
    baseline_mem: ExplicitSenseMemory,
    overlay_mem: ExplicitSenseMemory,
) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = {}
    for term in baseline_mem.known_terms():
        for base_entry in baseline_mem.get_senses(term):
            ov_entries = [
                e for e in overlay_mem.get_senses(term)
                if e.sense_id == base_entry.sense_id
            ]
            if not ov_entries:
                continue
            new = set(ov_entries[0].cues) - set(base_entry.cues)
            if new:
                result[(term, base_entry.sense_id)] = new
    return result


def _cue_trace_markers(
    row: dict,
    overlay_only: dict[tuple[str, str], set[str]],
) -> list[str]:
    term = row.get("ambiguous_term", "")
    if not term:
        # Probe rows may not expose ambiguous_term from engine; derive from id
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


def _append_overlay_trace(row: dict, overlay: KnowledgeOverlay, markers: list[str]) -> None:
    base_markers = overlay.overlay_trace_markers(len(markers))
    all_markers = base_markers + markers
    existing = row.get("memory_hits", "")
    row["memory_hits"] = (existing + " | " + " | ".join(all_markers)) if existing else " | ".join(all_markers)


def _knowledge_utilization(
    auto_review_data: dict,
    baseline_mem: ExplicitSenseMemory,
    overlay_rows: list[dict],
    pattern_candidates: list[dict],
) -> dict:
    # accepted_auto cues that are genuinely new (not in builtin)
    builtin_cues: dict[tuple[str, str], set[str]] = {}
    for term in baseline_mem.known_terms():
        for entry in baseline_mem.get_senses(term):
            builtin_cues[(term, entry.sense_id)] = set(entry.cues)

    new_cues: set[tuple[str, str, str]] = set()
    for pr in auto_review_data.get("proposal_reviews", []):
        term, sense = pr["term"], pr["sense"]
        for item in pr.get("items", []):
            if item.get("decision") == "accepted_auto" and item.get("item_type") == "positive_cue":
                v = item["value"].lower()
                if v not in builtin_cues.get((term, sense), set()):
                    new_cues.add((term, sense, v))

    # Which new cues appear in any probe prompt (via overlay trace)
    observed: set[tuple[str, str, str]] = set()
    for row in overlay_rows:
        for part in row.get("overlay_memory_hits", "").split(" | "):
            if part.startswith("overlay_cue="):
                rest = part[len("overlay_cue="):]
                if " -> " in rest:
                    cue_val, ts = rest.split(" -> ", 1)
                    if ":" in ts:
                        t, s = ts.split(":", 1)
                        observed.add((t, s, cue_val.strip()))

    accepted_patterns = [p for p in pattern_candidates if p.get("decision") == "accepted_auto"]
    # Pattern "used" = its (term, sense) has at least one overlay-influenced probe row
    influenced_pairs: set[tuple[str, str]] = set()
    for row in overlay_rows:
        if row.get("overlay_influenced") == "yes":
            for part in row.get("overlay_memory_hits", "").split(" | "):
                if part.startswith("overlay_cue="):
                    rest = part[len("overlay_cue="):]
                    if " -> " in rest:
                        _, ts = rest.split(" -> ", 1)
                        if ":" in ts:
                            t, s = ts.split(":", 1)
                            influenced_pairs.add((t, s))

    patterns_used = sum(
        1 for p in accepted_patterns
        if (p["term"], p["sense"]) in influenced_pairs
    )
    patterns_unused = len(accepted_patterns) - patterns_used

    return {
        "accepted_auto_cues_observed": len(observed),
        "accepted_auto_cues_unused": len(new_cues) - len(observed),
        "patterns_used": patterns_used,
        "patterns_unused": patterns_unused,
    }


def run(
    probe_path: str,
    patterns_path: str,
    auto_review_path: str,
    output_csv_path: str,
    output_json_path: str,
) -> dict:
    """Run probe overlay benchmark. Returns the summary dict."""
    with open(probe_path, newline="", encoding="utf-8") as f:
        probe_rows = list(csv.DictReader(f))

    with open(patterns_path, encoding="utf-8") as f:
        patterns_data = json.load(f)
    pattern_candidates: list[dict] = patterns_data.get("pattern_candidates", [])
    pattern_stats: dict = patterns_data.get("stats", {})

    with open(auto_review_path, encoding="utf-8") as f:
        auto_review_data = json.load(f)

    # Build engines
    baseline_mem = ExplicitSenseMemory(include_builtin=True)
    baseline_engine = ControlledContinuationEngine(memory=baseline_mem)

    overlay = KnowledgeOverlay(auto_review_data)
    overlay_mem = overlay.build_overlay_memory()
    overlay_engine = ControlledContinuationEngine(memory=overlay_mem)

    overlay_only = _build_overlay_only_cues(baseline_mem, overlay_mem)

    # Run both engines
    baseline_results = _run_engine(baseline_engine, probe_rows)
    overlay_results_raw = _run_engine(overlay_engine, probe_rows)

    # Build output rows with trace markers and comparison
    out_rows: list[dict] = []
    newly_continued: list[str] = []
    newly_audited: list[str] = []
    changed_ids: list[str] = []
    regressions: list[str] = []

    for b_r, o_r, probe in zip(baseline_results, overlay_results_raw, probe_rows):
        rid = probe["row_id"]
        b_dec = b_r["decision"]
        o_dec = o_r["decision"]

        # Derive cue trace for overlay
        # Reuse the engine's memory_hits but need term; parse from memory_hits
        cue_markers: list[str] = []
        for part in o_r["memory_hits"].split(" | "):
            if part.startswith("positive_cue="):
                rest = part[len("positive_cue="):]
                if " -> " in rest:
                    cue, sense_id = rest.split(" -> ", 1)
                    # Look up all known terms to match
                    for (term, sid), new in overlay_only.items():
                        if sid == sense_id and cue in new:
                            cue_markers.append(f"overlay_cue={cue} -> {term}:{sid}")
                            break

        overlay_memory_hits = o_r["memory_hits"]
        if cue_markers:
            base_trace = overlay.overlay_trace_markers(len(cue_markers))
            all_trace = base_trace + cue_markers
            overlay_memory_hits = (overlay_memory_hits + " | " + " | ".join(all_trace)) if overlay_memory_hits else " | ".join(all_trace)

        overlay_influenced = bool(cue_markers)
        changed = b_dec != o_dec or b_r["selected_sense"] != o_r["selected_sense"]

        exp_sense = probe.get("target_sense", "")
        # Safety check: wrong continue
        if o_dec == "continue" and exp_sense and o_r["selected_sense"] != exp_sense:
            regressions.append(f"{rid}:wrong_sense(expected={exp_sense},got={o_r['selected_sense']})")

        if b_dec == "audit" and o_dec == "continue":
            newly_continued.append(rid)
        if b_dec == "continue" and o_dec == "audit":
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
            "overlay_decision": o_dec,
            "overlay_selected_sense": o_r["selected_sense"],
            "overlay_continuation": o_r["continuation"],
            "baseline_reasons": b_r["reasons"],
            "overlay_reasons": o_r["reasons"],
            "overlay_memory_hits": overlay_memory_hits,
            "changed": "yes" if changed else "no",
            "newly_continued": "yes" if (b_dec == "audit" and o_dec == "continue") else "no",
            "newly_audited": "yes" if (b_dec == "continue" and o_dec == "audit") else "no",
            "overlay_influenced": "yes" if overlay_influenced else "no",
        })

    # Metrics: use id/decision/selected_sense/continuation for summarize_rows
    baseline_for_metrics = [
        {"id": r["id"], "decision": r["decision"], "selected_sense": r["selected_sense"],
         "continuation": r["continuation"], "expected_sense": r["expected_sense"]}
        for r in baseline_results
    ]
    overlay_for_metrics = [
        {"id": row["row_id"], "decision": row["overlay_decision"],
         "selected_sense": row["overlay_selected_sense"],
         "continuation": row["overlay_continuation"],
         "expected_sense": row["target_sense"],
         "prompt": row["prompt"]}
        for row in out_rows
    ]

    b_stats = summarize_rows(baseline_for_metrics)
    o_stats = summarize_rows(overlay_for_metrics)
    b_quality = check_rows(baseline_for_metrics)
    o_quality = check_rows(overlay_for_metrics)
    b_wrong = _count_wrong_continues(baseline_for_metrics)
    o_wrong = _count_wrong_continues(overlay_for_metrics)

    policy = ContinuationPolicy()

    ku = _knowledge_utilization(auto_review_data, baseline_mem, out_rows, pattern_candidates)

    summary = {
        "probe_total": len(probe_rows),
        "baseline": {
            "continue_count": b_stats["continue_count"],
            "audit_count": b_stats["audit_count"],
            "wrong_continue_count": b_wrong,
            "semantic_quality_flagged": b_quality["flagged_count"],
        },
        "knowledge_pattern_overlay": {
            "continue_count": o_stats["continue_count"],
            "audit_count": o_stats["audit_count"],
            "wrong_continue_count": o_wrong,
            "semantic_quality_flagged": o_quality["flagged_count"],
        },
        "delta": {
            "continue_count": o_stats["continue_count"] - b_stats["continue_count"],
            "newly_continued_rows": sorted(newly_continued),
            "newly_audited_rows": sorted(newly_audited),
            "changed_output_rows": sorted(changed_ids),
            "risk_regressions": regressions,
        },
        "pattern_stats": {
            "pattern_candidates": pattern_stats.get("total", len(pattern_candidates)),
            "accepted_auto": pattern_stats.get("by_decision", {}).get("accepted_auto", 0),
            "needs_review": pattern_stats.get("by_decision", {}).get("needs_review", 0),
            "rejected_auto": pattern_stats.get("by_decision", {}).get("rejected_auto", 0),
            "by_pattern_type": pattern_stats.get("by_pattern_type", {}),
        },
        "knowledge_utilization": ku,
        "safety": {
            "thresholds_changed": policy.min_score != 1.0 or policy.min_margin != 1.0,
            "validators_weakened": False,
            "sense_memory_modified": False,
            "trusted_baseline_modified": False,
        },
    }

    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
        f.write("\n")

    return summary


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Knowledge probe overlay benchmark v1.")
    parser.add_argument("--probe", required=True, help="knowledge_probe_prompts_v1.csv")
    parser.add_argument("--patterns", required=True, help="wiki_pattern_candidates_v1.json")
    parser.add_argument("--auto-review", required=True, dest="auto_review",
                        help="knowledge_ingestion_v1_auto_review.json")
    parser.add_argument("--output-csv", required=True, dest="output_csv",
                        help="Output probe overlay CSV")
    parser.add_argument("--output-json", required=True, dest="output_json",
                        help="Output probe overlay summary JSON")
    args = parser.parse_args(argv)

    summary = run(
        probe_path=args.probe,
        patterns_path=args.patterns,
        auto_review_path=args.auto_review,
        output_csv_path=args.output_csv,
        output_json_path=args.output_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
