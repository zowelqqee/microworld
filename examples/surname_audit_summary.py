"""
Summarise a manually-labelled name/surname audit CSV.

Valid manual_label values: good | bad | unclear
Rows with an empty manual_label are skipped (not yet reviewed).

Reports:
  - counts of good / bad / unclear
  - generation_precision = good / (good + bad)   (unclear excluded; 0 if no denom)
  - good_rate / bad_rate / unclear_rate over all reviewed rows
  - average quality_score for good and for bad names
  - the most common quality reasons among bad names

Usage:
    python examples/surname_audit_summary.py --input data/surname_audit.csv
"""
import argparse
import csv
import os
import sys
from collections import Counter

_HERE = os.path.dirname(__file__)
_DEFAULT_INPUT = os.path.normpath(
    os.path.join(_HERE, "..", "data", "surname_audit.csv")
)

VALID_LABELS = ("good", "bad", "unclear")
REQUIRED_COLUMNS = ("name", "manual_label")

_ALIASES: dict[str, str] = {
    "g": "good",
    "ok": "good",
    "yes": "good",
    "b": "bad",
    "no": "bad",
    "junk": "bad",
    "u": "unclear",
    "maybe": "unclear",
    "?": "unclear",
}


def normalize_label(raw: str) -> str:
    """Normalise a raw manual_label to good | bad | unclear (unknown → unclear)."""
    label = raw.strip().lower()
    label = _ALIASES.get(label, label)
    return label if label in VALID_LABELS else "unclear"


def _detect_dialect(sample: str) -> csv.Dialect:
    """Detect comma vs tab audit files, falling back to a simple first-line check."""
    try:
        return csv.Sniffer().sniff(sample, delimiters=",\t")
    except csv.Error:
        first_line = sample.splitlines()[0] if sample.splitlines() else ""
        delimiter = "\t" if first_line.count("\t") > first_line.count(",") else ","
        return type("_Fallback", (csv.excel,), {"delimiter": delimiter})


def _normalise_fieldnames(fieldnames: list[str] | None, path: str) -> list[str]:
    if not fieldnames:
        raise ValueError(f"Missing header row in audit file: {path}")

    normalised = [(name or "").strip() for name in fieldnames]
    missing = [col for col in REQUIRED_COLUMNS if col not in normalised]
    if missing:
        found = ", ".join(name for name in normalised if name) or "none"
        required = ", ".join(REQUIRED_COLUMNS)
        raise ValueError(
            f"Missing required column(s) in audit file {path}: "
            f"{', '.join(missing)}. Required: {required}. Found: {found}."
        )
    return normalised


def read_labeled_rows(path: str) -> list[dict]:
    """Return reviewed audit rows from compact or full CSV/TSV files.

    Required columns are ``name`` and ``manual_label``.  Extra columns from the
    full generated audit CSV are preserved when present.
    """
    with open(path, newline="", encoding="utf-8-sig") as f:
        sample = f.read(4096)
        f.seek(0)
        reader = csv.DictReader(f, dialect=_detect_dialect(sample))
        reader.fieldnames = _normalise_fieldnames(reader.fieldnames, path)

        rows: list[dict] = []
        for row in reader:
            clean = {
                (key or "").strip(): value
                for key, value in row.items()
                if key is not None
            }
            label = (clean.get("manual_label") or "").strip().lower()
            if not label:
                continue
            clean["manual_label"] = label
            rows.append(clean)
        return rows


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def compute_summary(rows: list[dict]) -> dict:
    """Compute counts, precision, average quality, and bad-name reason frequencies.

    ``generation_precision`` = good / (good + bad); unclear is excluded from the
    denominator and the precision is 0.0 when the denominator is zero.
    """
    counts = {label: 0 for label in VALID_LABELS}
    quality = {"good": [], "bad": []}
    bad_reasons: Counter = Counter()

    for row in rows:
        label = normalize_label(row.get("manual_label", ""))
        counts[label] += 1

        score = _to_float(row.get("quality_score", ""))
        if label in quality and score is not None:
            quality[label].append(score)

        if label == "bad":
            raw = (row.get("quality_reasons") or "").strip()
            for reason in (r for r in raw.split("|") if r):
                bad_reasons[reason] += 1

    total = len(rows)
    good, bad = counts["good"], counts["bad"]
    denom = good + bad
    precision = good / denom if denom else 0.0

    def _rate(label: str) -> float:
        return counts[label] / total if total else 0.0

    def _avg(values: list[float]) -> float | None:
        return sum(values) / len(values) if values else None

    return {
        "total": total,
        "counts": counts,
        "generation_precision": precision,
        "good_rate": _rate("good"),
        "bad_rate": _rate("bad"),
        "unclear_rate": _rate("unclear"),
        "avg_quality_good": _avg(quality["good"]),
        "avg_quality_bad": _avg(quality["bad"]),
        "common_bad_reasons": bad_reasons.most_common(),
    }


def _fmt_avg(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def format_summary(summary: dict) -> str:
    counts = summary["counts"]
    lines = [
        f"Total reviewed       : {summary['total']}",
        f"Good                 : {counts['good']}",
        f"Bad                  : {counts['bad']}",
        f"Unclear              : {counts['unclear']}",
        f"Generation precision : {summary['generation_precision']:.4f}"
        "   (good / (good + bad))",
        f"Good rate            : {summary['good_rate']:.4f}",
        f"Bad rate             : {summary['bad_rate']:.4f}",
        f"Unclear rate         : {summary['unclear_rate']:.4f}",
        f"Avg quality (good)   : {_fmt_avg(summary['avg_quality_good'])}",
        f"Avg quality (bad)    : {_fmt_avg(summary['avg_quality_bad'])}",
    ]
    if summary["common_bad_reasons"]:
        lines.append("Common bad reasons   :")
        for reason, n in summary["common_bad_reasons"]:
            lines.append(f"    {n:>4d}  {reason}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Summarise a manually-labelled name/surname audit CSV."
    )
    ap.add_argument("--input", default=_DEFAULT_INPUT,
                    help="Path to the labelled name/surname audit CSV (default: data/surname_audit.csv)")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        print(f"Input not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    try:
        rows = read_labeled_rows(args.input)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    print(format_summary(compute_summary(rows)))


if __name__ == "__main__":
    main()
