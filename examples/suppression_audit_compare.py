"""
Compare two suppression audit CSV files (labeled or unlabeled).

For labeled files (any non-empty manual_label values) it computes:
  - total reviewed, should_suppress, should_keep, unclear
  - suppression_precision = should_suppress / (should_suppress + should_keep)

For unlabeled files it reports:
  - total rows
  - avg/min/max delta, avg baseline/learned confidence
  - relation_type distribution

Usage:
    python examples/suppression_audit_compare.py \\
        --old data/suppression_audit.csv \\
        --new data/suppression_audit_calibrated.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter

_HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(_HERE, ".."))

from examples.suppression_audit_summary import compute_summary


def read_all_rows(path: str) -> list[dict]:
    """Read every row from a CSV, regardless of manual_label."""
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def is_labeled(rows: list[dict]) -> bool:
    """Return True if any row has a non-empty manual_label."""
    return any(r.get("manual_label", "").strip() for r in rows)


def compute_stats(rows: list[dict]) -> dict:
    """Compute numeric stats for a set of audit rows (labeled or not)."""
    n = len(rows)
    if n == 0:
        return {"total": 0}
    deltas = [float(r["delta"]) for r in rows]
    baselines = [float(r["baseline_confidence"]) for r in rows]
    learneds = [float(r["learned_confidence"]) for r in rows]
    rel_dist = dict(Counter(r.get("relation_type", "") for r in rows))
    return {
        "total": n,
        "avg_baseline_confidence": sum(baselines) / n,
        "avg_learned_confidence": sum(learneds) / n,
        "avg_delta": sum(deltas) / n,
        "min_delta": min(deltas),
        "max_delta": max(deltas),
        "relation_type_distribution": rel_dist,
    }


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def format_file_report(label: str, rows: list[dict]) -> str:
    lines = [f"--- {label} ---"]

    if is_labeled(rows):
        labeled = [r for r in rows if r.get("manual_label", "").strip()]
        summary = compute_summary(labeled)
        total = summary["total"]
        counts = summary["counts"]
        suppress = counts["should_suppress"]
        keep = counts["should_keep"]
        unclear = counts["unclear"]
        lines.append(f"Total reviewed       : {total}")
        lines.append(f"should_suppress      : {suppress:>4d}  ({_pct(suppress, total):5.1f}%)")
        lines.append(f"should_keep          : {keep:>4d}  ({_pct(keep, total):5.1f}%)")
        lines.append(f"unclear              : {unclear:>4d}  ({_pct(unclear, total):5.1f}%)")
        lines.append(
            f"suppression_precision: {summary['suppression_precision']:.3f}"
            f"  (should_suppress / (should_suppress + should_keep))"
        )
    else:
        stats = compute_stats(rows)
        n = stats["total"]
        lines.append(f"Total rows           : {n}")
        if n > 0:
            lines.append(f"avg baseline_conf    : {stats['avg_baseline_confidence']:.6f}")
            lines.append(f"avg learned_conf     : {stats['avg_learned_confidence']:.6f}")
            lines.append(f"avg delta            : {stats['avg_delta']:.6f}")
            lines.append(f"min delta            : {stats['min_delta']:.6f}")
            lines.append(f"max delta            : {stats['max_delta']:.6f}")
            lines.append("relation_type dist   :")
            for rel, count in sorted(stats["relation_type_distribution"].items()):
                lines.append(f"  {rel:<20s}: {count}")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare two suppression audit CSV files."
    )
    ap.add_argument("--old", required=True, help="Old/baseline audit CSV")
    ap.add_argument("--new", required=True, help="New/calibrated audit CSV")
    args = ap.parse_args()

    for path in (args.old, args.new):
        if not os.path.exists(path):
            print(f"File not found: {path}", file=sys.stderr)
            sys.exit(1)

    old_rows = read_all_rows(args.old)
    new_rows = read_all_rows(args.new)

    print(format_file_report(args.old, old_rows))
    print()
    print(format_file_report(args.new, new_rows))


if __name__ == "__main__":
    main()
