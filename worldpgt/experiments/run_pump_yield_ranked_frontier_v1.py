"""Knowledge Pump Yield-Ranked Frontier v1.9 runner.

Reads existing pump artifacts and produces:
  * an incremental (per-batch) yield trace,
  * a yield-gate recommendation (blind vs yield-ranked fetch),
  * a yield-ranked next-batch plan of up to 250 titles,
  * a low-yield blocklist and high-yield frontier list.

Usage:
    python3 worldpgt/experiments/run_pump_yield_ranked_frontier_v1.py

No network calls. No writes to accepted memory, overlays, or snapshot files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge_pump.incremental_yield_trace import DEFAULT_RECENT_WINDOW
from worldpgt.knowledge_pump.yield_gate import DEFAULT_MIN_YIELD_PER_READY_DOC
from worldpgt.knowledge_pump.yield_ranked_frontier import run_yield_ranked_frontier

_EXPERIMENTS = Path(__file__).resolve().parent
_PUMP_DIR = _EXPERIMENTS / "knowledge_pump_v1"
_OUT_DIR = _PUMP_DIR / "yield_ranked_frontier_v1"
_SNAPSHOTS = _EXPERIMENTS / "wiki_snapshots_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pump-dir", default=str(_PUMP_DIR))
    parser.add_argument("--out-dir", default=str(_OUT_DIR))
    parser.add_argument("--snapshots-dir", default=str(_SNAPSHOTS))
    parser.add_argument("--recent-window", type=int, default=DEFAULT_RECENT_WINDOW)
    parser.add_argument("--min-yield-per-ready-doc", type=float,
                        default=DEFAULT_MIN_YIELD_PER_READY_DOC)
    parser.add_argument("--batch-plan-limit", type=int, default=250)
    parser.add_argument("--force-low-yield-fetch", action="store_true", default=False)
    args = parser.parse_args(argv)

    pump_dir = Path(args.pump_dir)
    if not pump_dir.exists():
        print(f"ERROR: pump_dir does not exist: {pump_dir}", file=sys.stderr)
        return 1

    print("Knowledge Pump Yield-Ranked Frontier v1.9")
    print(f"  pump_dir: {pump_dir}")
    print(f"  out_dir:  {args.out_dir}")

    result = run_yield_ranked_frontier(
        pump_dir,
        Path(args.out_dir),
        snapshots_dir=Path(args.snapshots_dir),
        recent_window=args.recent_window,
        min_yield_per_ready_doc=args.min_yield_per_ready_doc,
        batch_plan_limit=args.batch_plan_limit,
        force=args.force_low_yield_fetch,
    )

    summ = result["incremental_summary"]
    gate = result["gate"]

    print("\nIncremental Yield (recent window):")
    for key in (
        "recent_batches_analyzed", "recent_titles_fetched", "recent_fetch_success",
        "recent_ready_docs", "recent_new_answerable_facts", "recent_new_relations",
        "recent_new_definitions", "recent_answerable_yield_per_ready_doc",
        "recent_answerable_yield_per_fetch_success", "zero_yield_batch_count",
        "negative_yield_detected", "cumulative_answerable_facts",
        "attribution_mode", "attribution_confidence",
    ):
        print(f"  {key}: {summ.get(key)}")

    print("\nYield Gate:")
    for key in (
        "blind_fetch_recommended", "yield_ranked_fetch_recommended",
        "force_required_for_blind_fetch", "blind_fetch_allowed",
        "low_yield_detected", "all_recent_batches_zero_yield",
    ):
        print(f"  {key}: {gate.get(key)}")
    for warning in gate.get("warnings", []):
        print(f"  WARNING: {warning}")

    print("\nYield-Ranked Frontier:")
    print(f"  yield_ranked_selected_count: {summ.get('yield_ranked_selected_count')}")
    print(f"  yield_ranked_available_count: {summ.get('yield_ranked_available_count')}")
    print(f"  high_yield_frontier_count: {summ.get('high_yield_frontier_count')}")
    print(f"  low_yield_blocklist_count: {summ.get('low_yield_blocklist_count')}")

    print("\nTop high-yield seed pages (by accepted facts):")
    for entry in result["high_yield"][:10]:
        print(f"  {entry['title']}  score={entry['score']}  class={entry['expected_yield_class']}")

    print("\nTop blocked low-yield titles:")
    for entry in result["blocklist"][:10]:
        print(f"  {entry['title']}  reasons={entry['block_reasons']}")

    print("\nSample next selected titles:")
    for entry in result["plan"][:15]:
        print(f"  {entry['title']}  score={entry['score']}  class={entry['expected_yield_class']}")

    print(f"\nArtifacts written to: {args.out_dir}")
    for name in result["artifacts_written"]:
        print(f"  {name}")

    print(f"\n  network_calls: {result['network_calls']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
