"""
Summarise a manually-labelled suppression audit CSV.

Valid manual_label values: should_suppress | should_keep | unclear
Rows with an empty manual_label are skipped (not yet reviewed).

Computes:
  suppression_precision = should_suppress / (should_suppress + should_keep)

Usage:
    python examples/suppression_audit_summary.py
    python examples/suppression_audit_summary.py --input data/suppression_audit.csv
"""
import argparse
import csv
import os
import sys

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(
    os.path.join(_HERE, "..", "data", "suppression_audit.csv")
)

VALID_LABELS = {"should_suppress", "should_keep", "unclear"}


def _detect_delimiter(sample: str) -> str:
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_labeled_rows(path: str) -> list[dict]:
    """Return rows whose manual_label is non-empty."""
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        delimiter = _detect_delimiter(sample)
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [row for row in reader if row.get("manual_label", "").strip()]


def compute_summary(rows: list[dict]) -> dict:
    """Return total, per-label counts, and suppression_precision.

    suppression_precision = should_suppress / (should_suppress + should_keep)
    Labels not in VALID_LABELS are counted as 'unclear'.
    """
    counts: dict[str, int] = {"should_suppress": 0, "should_keep": 0, "unclear": 0}
    for row in rows:
        label = row.get("manual_label", "").strip().lower()
        if label not in VALID_LABELS:
            label = "unclear"
        counts[label] += 1

    suppress = counts["should_suppress"]
    keep = counts["should_keep"]
    denom = suppress + keep
    precision = suppress / denom if denom > 0 else 0.0

    return {
        "total": len(rows),
        "counts": counts,
        "suppression_precision": precision,
    }


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def format_summary(summary: dict) -> str:
    total = summary["total"]
    counts = summary["counts"]
    precision = summary["suppression_precision"]
    lines: list[str] = []

    lines.append(f"Total reviewed       : {total}")
    if total == 0:
        lines.append("  (no labeled rows found)")
        return "\n".join(lines)

    suppress = counts.get("should_suppress", 0)
    keep = counts.get("should_keep", 0)
    unclear = counts.get("unclear", 0)

    lines.append(f"should_suppress      : {suppress:>4d}  ({_pct(suppress, total):5.1f}%)")
    lines.append(f"should_keep          : {keep:>4d}  ({_pct(keep,     total):5.1f}%)")
    lines.append(f"unclear              : {unclear:>4d}  ({_pct(unclear,  total):5.1f}%)")
    lines.append(
        f"suppression_precision: {precision:.3f}"
        f"  (should_suppress / (should_suppress + should_keep))"
    )

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarise a manually-labelled suppression audit CSV."
    )
    ap.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help="Path to the labelled suppression audit CSV (default: data/suppression_audit.csv)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows = read_labeled_rows(args.input)
    summary = compute_summary(rows)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
