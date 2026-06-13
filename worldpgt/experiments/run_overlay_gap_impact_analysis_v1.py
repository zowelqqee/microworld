"""CLI runner: overlay gap impact analysis v1.

Loads the committed overlay artifacts and baseline audit metadata, runs the
gap impact analyzer, and writes three output files:
  - knowledge_overlay_v1_gap_impact_analysis.json  (full analysis)
  - knowledge_overlay_v1_gap_impact_analysis.csv   (per-row analysis)
  - knowledge_overlay_v1_targeted_acquisition_plan.csv  (acquisition plan)

READ-ONLY with respect to all source files, memory, policy, and benchmark outputs.

Usage:
    python3 -m worldpgt.experiments.run_overlay_gap_impact_analysis_v1 \\
      --baseline-output worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \\
      --overlay-output  worldpgt/experiments/knowledge_overlay_v1_outputs.csv \\
      --overlay-summary worldpgt/experiments/knowledge_overlay_v1_summary.json \\
      --audit-reasons   worldpgt/experiments/microworld_continuation_v1_2_audit_reasons.json \\
      --curriculum      worldpgt/experiments/microworld_continuation_v1_2_coverage_gap_curriculum.json \\
      --auto-review     worldpgt/experiments/knowledge_ingestion_v1_auto_review.json \\
      --output-json     worldpgt/experiments/knowledge_overlay_v1_gap_impact_analysis.json \\
      --output-csv      worldpgt/experiments/knowledge_overlay_v1_gap_impact_analysis.csv \\
      --plan-csv        worldpgt/experiments/knowledge_overlay_v1_targeted_acquisition_plan.csv
"""

from __future__ import annotations

import argparse
import csv
import json

from worldpgt.experiments.overlay_gap_impact_analyzer import run_analysis

_ANALYSIS_CSV_FIELDS = [
    "row_id", "term", "baseline_decision", "overlay_decision",
    "change_type", "is_newly_audited", "overlay_influenced",
    "primary_audit_reason", "blocker_type", "required_input",
    "expected_effect", "curriculum_task_id", "why_overlay_insufficient",
    "targetable",
]

_PLAN_CSV_FIELDS = [
    "row_id", "term", "blocker_type", "required_input",
    "expected_effect", "source_strategy", "priority", "curriculum_task_id",
]


def _load_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_csv(path: str, rows: list[dict], fields: list[str]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run(
    baseline_output_path: str,
    overlay_output_path: str,
    overlay_summary_path: str,
    audit_reasons_path: str,
    curriculum_path: str,
    auto_review_path: str,
    output_json_path: str,
    output_csv_path: str,
    plan_csv_path: str,
) -> dict:
    """Load artifacts, run analysis, write output files. Returns the full analysis dict."""
    analysis = run_analysis(
        baseline_rows=_load_csv(baseline_output_path),
        overlay_rows=_load_csv(overlay_output_path),
        overlay_summary=_load_json(overlay_summary_path),
        audit_reasons_data=_load_json(audit_reasons_path),
        curriculum_data=_load_json(curriculum_path),
        auto_review_data=_load_json(auto_review_path),
    )

    _write_csv(output_csv_path, analysis["analysis_csv_rows"], _ANALYSIS_CSV_FIELDS)
    _write_csv(plan_csv_path, analysis["targeted_acquisition_plan"], _PLAN_CSV_FIELDS)

    # Exclude analysis_csv_rows from the JSON (already in the CSV)
    json_payload = {k: v for k, v in analysis.items() if k != "analysis_csv_rows"}
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(json_payload, f, indent=2, sort_keys=True)
        f.write("\n")

    return analysis


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Overlay gap impact analysis v1.")
    parser.add_argument("--baseline-output", required=True, dest="baseline_output",
                        help="microworld_continuation_v1_2_outputs.csv")
    parser.add_argument("--overlay-output", required=True, dest="overlay_output",
                        help="knowledge_overlay_v1_outputs.csv")
    parser.add_argument("--overlay-summary", required=True, dest="overlay_summary",
                        help="knowledge_overlay_v1_summary.json")
    parser.add_argument("--audit-reasons", required=True, dest="audit_reasons",
                        help="microworld_continuation_v1_2_audit_reasons.json")
    parser.add_argument("--curriculum", required=True,
                        help="microworld_continuation_v1_2_coverage_gap_curriculum.json")
    parser.add_argument("--auto-review", required=True, dest="auto_review",
                        help="knowledge_ingestion_v1_auto_review.json")
    parser.add_argument("--output-json", required=True, dest="output_json",
                        help="Output gap impact analysis JSON")
    parser.add_argument("--output-csv", required=True, dest="output_csv",
                        help="Output gap impact analysis CSV")
    parser.add_argument("--plan-csv", required=True, dest="plan_csv",
                        help="Output targeted acquisition plan CSV")
    args = parser.parse_args(argv)

    analysis = run(
        baseline_output_path=args.baseline_output,
        overlay_output_path=args.overlay_output,
        overlay_summary_path=args.overlay_summary,
        audit_reasons_path=args.audit_reasons,
        curriculum_path=args.curriculum,
        auto_review_path=args.auto_review,
        output_json_path=args.output_json,
        output_csv_path=args.output_csv,
        plan_csv_path=args.plan_csv,
    )
    print(json.dumps(analysis["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
