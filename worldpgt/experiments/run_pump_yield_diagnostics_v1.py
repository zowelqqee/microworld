"""Knowledge Pump Yield Diagnostics v1.

Reads existing pump artifacts and produces a deterministic yield funnel
report explaining where each batch loses usable knowledge.

Usage:
    python3 worldpgt/experiments/run_pump_yield_diagnostics_v1.py

No network calls. No writes to accepted memory, overlays, or snapshot files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge_pump.yield_diagnostics import (
    run_yield_diagnostics,
    update_pump_summary_with_yield,
    update_pump_report_with_yield,
)

_EXPERIMENTS = Path(__file__).resolve().parent
_PUMP_DIR = _EXPERIMENTS / "knowledge_pump_v1"
_OUT_DIR = _PUMP_DIR / "yield_diagnostics_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pump-dir", default=str(_PUMP_DIR),
        help="Path to knowledge_pump_v1 directory (default: %(default)s)",
    )
    parser.add_argument(
        "--out-dir", default=str(_OUT_DIR),
        help="Output directory for yield diagnostics artifacts (default: %(default)s)",
    )
    parser.add_argument(
        "--no-update-summary", action="store_true", default=False,
        help="Skip additive update of pump_summary.json and pump_report.json",
    )
    args = parser.parse_args(argv)

    pump_dir = Path(args.pump_dir)
    out_dir = Path(args.out_dir)

    if not pump_dir.exists():
        print(f"ERROR: pump_dir does not exist: {pump_dir}", file=sys.stderr)
        return 1

    print("Knowledge Pump Yield Diagnostics v1")
    print(f"  pump_dir: {pump_dir}")
    print(f"  out_dir:  {out_dir}")

    result = run_yield_diagnostics(pump_dir, out_dir)

    if not args.no_update_summary:
        update_pump_summary_with_yield(pump_dir, result)
        update_pump_report_with_yield(pump_dir, result)

    # Print funnel
    funnel = result.get("funnel", {})
    print("\nYield Funnel:")
    for key, val in funnel.items():
        if key == "attribution_mode":
            continue
        print(f"  {key}: {val}")
    print(f"  attribution_mode: {funnel.get('attribution_mode')}")

    # Print frontier quality
    fq = result.get("frontier_quality", {})
    print("\nFrontier Title Quality:")
    print(f"  total: {fq.get('frontier_title_total')}")
    print(f"  rejected_by_hygiene: {fq.get('frontier_title_rejected_by_hygiene_count')}")
    print(f"  kept_by_hygiene: {fq.get('frontier_title_kept_by_hygiene_count')}")
    for reason, count in (fq.get("frontier_title_rejection_by_reason") or {}).items():
        print(f"    {reason}: {count}")

    # Print sink classification
    sinks = result.get("sink_classification", {})
    print("\nSink Classification:")
    for k, v in (sinks.get("sink_counts") or {}).items():
        print(f"  {k}: {v}")
    print(f"  dominant_sinks: {sinks.get('dominant_sinks')}")
    print(f"  fresh_recompute_appears_to_use_new_ready_docs: "
          f"{sinks.get('fresh_recompute_appears_to_use_new_ready_docs')}")
    print(f"  merge_appears_to_include_fresh_candidates: "
          f"{sinks.get('merge_appears_to_include_fresh_candidates')}")
    print(f"  precision_accepted_fact_count_increased: "
          f"{sinks.get('precision_accepted_fact_count_increased')}")
    print(f"  qa_fact_count_increased: {sinks.get('qa_fact_count_increased')}")

    # Print batch history sanity
    hs = result.get("batch_history_sanity", {})
    print("\nBatch History Sanity:")
    for k, v in hs.items():
        print(f"  {k}: {v}")

    print(f"\nArtifacts written to: {out_dir}")
    for a in result.get("artifacts_written", []):
        print(f"  {Path(a).name}")

    print("\nConfirmations:")
    print(f"  network_calls: {result.get('network_calls')}")
    print(f"  trusted_memory_modified: {result.get('trusted_memory_modified')}")
    print(f"  accepted_overlay_modified: {result.get('accepted_overlay_modified')}")
    print(f"  promoted_overlay_modified: {result.get('promoted_overlay_modified')}")
    print(f"  snapshot_dry_run_overlay_modified: {result.get('snapshot_dry_run_overlay_modified')}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
