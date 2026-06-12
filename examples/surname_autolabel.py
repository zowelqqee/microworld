"""
Auto-label a generated name/surname audit CSV using the quality policy as a
proxy for human judgment.

This is a demonstration tool — real usage requires hand-labelling the
manual_label column.  Auto-labelling lets you exercise the full feedback loop
(generate → label → trust-learn → regenerate → compare) without manual effort.

Labelling heuristic:
  quality_score >= good_threshold             → good
  quality_score < bad_threshold               → bad
  duplicate == "true" and quality < dup_bad   → bad
  otherwise                                   → unclear

Thresholds are tunable via CLI but the defaults are intentionally soft so that
a realistic mix of good / bad / unclear is produced.

Usage:
    python examples/surname_autolabel.py --input data/generated_surnames.csv \\
        --output data/generated_surnames_labeled.csv
    python examples/surname_autolabel.py --input data/generated_surnames.csv \\
        --inplace
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.surname_policy import explain_surname_quality, surname_quality_score

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(
    os.path.join(_HERE, "..", "data", "generated_surnames.csv")
)


def autolabel_row(
    row: dict,
    *,
    good_threshold: float = 0.85,
    bad_threshold: float = 0.60,
    dup_bad_threshold: float = 0.95,
) -> str:
    """Assign a synthetic manual_label to a single audit row.

    Uses quality_score from the row if present (avoids re-scoring); otherwise
    re-scores via the policy.
    """
    try:
        score = float(row.get("quality_score") or 0.0)
    except ValueError:
        score = surname_quality_score(row.get("name", ""))

    is_dup = str(row.get("duplicate", "")).strip().lower() == "true"

    if is_dup and score < dup_bad_threshold:
        return "bad"
    if score >= good_threshold:
        return "good"
    if score < bad_threshold:
        return "bad"
    return "unclear"


def autolabel_file(
    input_path: str,
    output_path: str,
    *,
    good_threshold: float = 0.85,
    bad_threshold: float = 0.60,
    dup_bad_threshold: float = 0.95,
    overwrite_existing: bool = False,
) -> dict:
    """Read *input_path*, assign auto-labels, write to *output_path*.

    Returns a small stats dict.
    """
    with open(input_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    if "manual_label" not in fieldnames:
        fieldnames = list(fieldnames) + ["manual_label"]

    stats = {"good": 0, "bad": 0, "unclear": 0, "skipped": 0, "total": len(rows)}
    for row in rows:
        existing = (row.get("manual_label") or "").strip()
        if existing and not overwrite_existing:
            stats["skipped"] += 1
            continue
        label = autolabel_row(
            row,
            good_threshold=good_threshold,
            bad_threshold=bad_threshold,
            dup_bad_threshold=dup_bad_threshold,
        )
        row["manual_label"] = label
        stats[label] += 1

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return stats


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Auto-label a name/surname audit CSV using the quality policy "
            "(proxy for manual review)."
        )
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Audit CSV to label (default: data/generated_surnames.csv)")
    ap.add_argument("--output", default=None,
                    help="Output path (default: same as input with _labeled suffix)")
    ap.add_argument("--inplace", action="store_true",
                    help="Overwrite the input file in-place")
    ap.add_argument("--overwrite-existing", action="store_true",
                    dest="overwrite_existing",
                    help="Re-label rows that already have a manual_label")
    ap.add_argument("--good-threshold", type=float, default=0.85,
                    dest="good_threshold",
                    help="quality_score >= this → good (default: 0.85)")
    ap.add_argument("--bad-threshold", type=float, default=0.60,
                    dest="bad_threshold",
                    help="quality_score < this → bad (default: 0.60)")
    ap.add_argument("--dup-bad-threshold", type=float, default=0.95,
                    dest="dup_bad_threshold",
                    help="Duplicate names below this score → bad (default: 0.95)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.inplace:
        output_path = args.input
    elif args.output:
        output_path = args.output
    else:
        base, ext = os.path.splitext(args.input)
        output_path = base + "_labeled" + ext

    stats = autolabel_file(
        args.input, output_path,
        good_threshold=args.good_threshold,
        bad_threshold=args.bad_threshold,
        dup_bad_threshold=args.dup_bad_threshold,
        overwrite_existing=args.overwrite_existing,
    )
    print(
        f"Labeled {stats['total']} rows "
        f"(good={stats['good']} bad={stats['bad']} unclear={stats['unclear']}"
        + (f" skipped_existing={stats['skipped']}" if stats["skipped"] else "")
        + f") → {output_path}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
