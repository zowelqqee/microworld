"""Gap-driven pump orchestrator: audit log → gap report → frontier update.

Reads accumulated audit events, classifies knowledge gaps, injects
gap-entity boost signals into the yield-ranked frontier scorer, and
writes an updated batch plan under yield_ranked_frontier_v1/.

Usage::

    # Full run: gap report + updated frontier batch plan
    python3 worldpgt/experiments/run_audit_driven_pump_v1.py

    # Dry-run: only write gap report, stop before frontier update
    python3 worldpgt/experiments/run_audit_driven_pump_v1.py --dry-run

    # Narrow window
    python3 worldpgt/experiments/run_audit_driven_pump_v1.py --period-days 7

    # Custom audit log location
    python3 worldpgt/experiments/run_audit_driven_pump_v1.py --log-path /path/to/audit_log.jsonl

Safety: read-only over accepted overlays and memory. Writes only under
knowledge_pump_v1/ and knowledge_pump_v1/yield_ranked_frontier_v1/.
No network calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge_pump.audit_gap_analyzer import analyze_gaps
from worldpgt.knowledge_pump.yield_ranked_frontier import (
    normalize_page,
    run_yield_ranked_frontier,
)

_PUMP_DIR = Path(__file__).resolve().parent / "knowledge_pump_v1"
_OUT_DIR = _PUMP_DIR / "yield_ranked_frontier_v1"
_PUMP_OVERLAY = _PUMP_DIR / "pump_dry_run_overlay.json"


def _print_report_summary(report) -> None:
    print(f"  Audit events in window : {report.total_audit_events}")
    print(f"  Acquisition candidates : {len(report.acquisition_candidates)} entities")
    print(f"  Stale/recheck candidates: {len(report.stale_candidates)} facts")
    print(f"  Policy-blocked (skip)  : {len(report.policy_blocked)} entities")
    if report.acquisition_candidates:
        print("  Top acquisition targets:")
        for entry in report.acquisition_candidates[:5]:
            print(f"    [{entry.gap_type:14s}] {entry.entity!r}  (×{entry.count})")
    if report.stale_candidates:
        print("  Top stale/recheck facts:")
        for entry in report.stale_candidates[:5]:
            print(
                f"    [{entry.staleness_ratio:.2f}×] "
                f"{entry.subject} {entry.predicate} {entry.object} "
                f"(as_of={entry.as_of})"
            )


def _stale_fact_signals(report) -> dict[str, dict]:
    signals: dict[str, dict] = {}
    for candidate in report.stale_candidates:
        key = normalize_page(candidate.subject)
        if not key:
            continue
        row = candidate.to_dict()
        previous = signals.get(key)
        if previous is None or row["staleness_ratio"] > previous.get("staleness_ratio", 0):
            signals[key] = row
    return signals


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Audit-driven pump orchestrator: gap analysis → frontier boost.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Stop after writing gap_report.json; do not update the frontier.",
    )
    parser.add_argument(
        "--period-days",
        type=int,
        default=30,
        metavar="N",
        help="Days of audit history to analyse (default 30; 0 = all time).",
    )
    parser.add_argument(
        "--log-path",
        default=None,
        metavar="PATH",
        help="Override the default audit_log.jsonl path.",
    )
    parser.add_argument(
        "--current-date",
        default=None,
        metavar="YYYY-MM-DD",
        help="Override today's date for deterministic freshness checks.",
    )
    args = parser.parse_args(argv)

    log_path = Path(args.log_path) if args.log_path else None

    # --- Step 1: gap analysis ---
    print(f"[1/2] Analysing audit log (period_days={args.period_days}) ...")
    report = analyze_gaps(
        log_path,
        period_days=args.period_days,
        overlay_path=_PUMP_OVERLAY,
        current_date=args.current_date,
        require_acquisition_eligibility=True,
    )
    _print_report_summary(report)

    gap_report_path = _PUMP_DIR / "gap_report.json"
    gap_report_path.parent.mkdir(parents=True, exist_ok=True)
    gap_report_path.write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"  → gap_report.json written")

    if args.dry_run:
        print("\n--dry-run: stopping before frontier update.")
        return 0

    # --- Step 2: frontier scoring with gap boost ---
    print("\n[2/2] Updating yield-ranked frontier with gap-entity signals ...")
    gap_entities: set[str] = {
        normalize_page(e.entity)
        for e in report.acquisition_candidates
        if e.entity and e.entity != "[unknown entity]"
    }
    stale_signals = _stale_fact_signals(report)
    print(f"  Gap entities injected into frontier scorer: {len(gap_entities)}")
    print(f"  Stale fact recheck signals injected: {len(stale_signals)}")

    result = run_yield_ranked_frontier(
        _PUMP_DIR,
        _OUT_DIR,
        gap_entities=gap_entities,
        stale_fact_signals=stale_signals,
    )

    plan = result.get("plan", [])
    gap_boosted = [t for t in plan if "gap_driven_entity" in t.get("positive_reasons", [])]
    stale_boosted = [t for t in plan if "stale_snapshot_recheck" in t.get("positive_reasons", [])]
    print(
        f"  Batch plan: {len(plan)} titles total, "
        f"{len(gap_boosted)} gap-boosted, {len(stale_boosted)} stale-recheck"
    )
    print(f"  → yield_ranked_batch_plan.json written")
    print("\nDone. No network calls. No accepted memory or overlay modified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
