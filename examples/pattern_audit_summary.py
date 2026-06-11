"""
Summarise a manually-labelled pattern_audit CSV.

Valid manual_label values: correct | plausible | wrong | unclear
Rows with an empty manual_label are skipped (not yet reviewed).

Usage:
    python examples/pattern_audit_summary.py
    python examples/pattern_audit_summary.py --input data/pattern_audit_filtered.csv
"""
import argparse
import csv
import os
import sys

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(
    os.path.join(_HERE, "..", "data", "pattern_audit_filtered.csv")
)

VALID_LABELS = {"correct", "plausible", "wrong", "unclear"}
USEFUL_LABELS = {"correct", "plausible"}

_ALIASES: dict[str, str] = {
    "plusable":  "plausible",
    "plausable": "plausible",
    "posible":   "plausible",
    "true":      "correct",
    "yes":       "correct",
    "false":     "wrong",
    "no":        "wrong",
}


def normalize_label(raw: str) -> str:
    """Normalise a raw manual_label to a canonical VALID_LABELS value.

    Steps: strip whitespace → lowercase → apply alias map → unknown → 'unclear'.
    """
    label = raw.strip().lower()
    label = _ALIASES.get(label, label)
    return label if label in VALID_LABELS else "unclear"


# ── core logic ────────────────────────────────────────────────────────────────

def _detect_delimiter(sample: str) -> str:
    """Return the delimiter detected by csv.Sniffer, falling back to comma."""
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","


def read_labeled_rows(path: str) -> list[dict]:
    """Return rows whose manual_label is non-empty.

    Auto-detects comma or semicolon delimiter via csv.Sniffer; falls back to
    comma if detection fails (e.g. empty file or single-column header).
    """
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
        delimiter = _detect_delimiter(sample)

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        return [row for row in reader if row.get("manual_label", "").strip()]


def compute_summary(rows: list[dict]) -> dict:
    """
    Returns a dict with:
      total, counts (label->int), by_relation (rel->{label->int, total->int})
    """
    counts: dict[str, int] = {label: 0 for label in VALID_LABELS}
    by_relation: dict[str, dict[str, int]] = {}

    for row in rows:
        label = normalize_label(row["manual_label"])
        rel   = row.get("relation_type", "unknown").strip()

        counts[label] = counts.get(label, 0) + 1

        if rel not in by_relation:
            by_relation[rel] = {lbl: 0 for lbl in VALID_LABELS}
            by_relation[rel]["total"] = 0
        by_relation[rel][label] = by_relation[rel].get(label, 0) + 1
        by_relation[rel]["total"] += 1

    return {
        "total":       len(rows),
        "counts":      counts,
        "by_relation": by_relation,
    }


def _pct(n: int, total: int) -> float:
    return 100.0 * n / total if total else 0.0


def format_summary(summary: dict) -> str:
    total  = summary["total"]
    counts = summary["counts"]
    lines: list[str] = []

    lines.append(f"Total reviewed : {total}")
    if total == 0:
        lines.append("  (no labeled rows found)")
        return "\n".join(lines)

    correct  = counts.get("correct",  0)
    plausible= counts.get("plausible",0)
    wrong    = counts.get("wrong",    0)
    unclear  = counts.get("unclear",  0)
    useful   = correct + plausible

    lines.append(f"Correct        : {correct:>4d}  ({_pct(correct,  total):5.1f}%)")
    lines.append(f"Plausible      : {plausible:>4d}  ({_pct(plausible,total):5.1f}%)")
    lines.append(f"Wrong          : {wrong:>4d}  ({_pct(wrong,    total):5.1f}%)")
    lines.append(f"Unclear        : {unclear:>4d}  ({_pct(unclear,  total):5.1f}%)")
    lines.append(f"Useful (c+p)   : {useful:>4d}  ({_pct(useful,   total):5.1f}%)")

    by_rel = summary["by_relation"]
    if by_rel:
        lines.append("")
        lines.append("Per-relation breakdown:")
        # column widths
        w_rel = max(len("relation_type"), max(len(r) for r in by_rel))
        header = (
            f"  {'relation_type':{w_rel}s}  "
            f"{'reviewed':>8}  {'correct':>7}  {'plausible':>9}  "
            f"{'wrong':>5}  {'unclear':>7}  {'useful%':>7}"
        )
        lines.append(header)
        lines.append("  " + "─" * (len(header) - 2))
        for rel in sorted(by_rel):
            rb    = by_rel[rel]
            rtot  = rb["total"]
            rc    = rb.get("correct",  0)
            rp    = rb.get("plausible",0)
            rw    = rb.get("wrong",    0)
            ru    = rb.get("unclear",  0)
            ruseful = rc + rp
            lines.append(
                f"  {rel:{w_rel}s}  "
                f"{rtot:>8d}  {rc:>7d}  {rp:>9d}  "
                f"{rw:>5d}  {ru:>7d}  {_pct(ruseful, rtot):>6.1f}%"
            )

    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarise a manually-labelled pattern audit CSV."
    )
    ap.add_argument(
        "--input", default=_DEFAULT_INPUT,
        help="Path to the labelled audit CSV (default: data/pattern_audit_filtered.csv)"
    )
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    rows    = read_labeled_rows(args.input)
    summary = compute_summary(rows)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
