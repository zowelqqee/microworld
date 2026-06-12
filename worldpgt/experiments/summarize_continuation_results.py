"""Summarize an audited continuation CSV (must include a `label` column) as JSON metrics.

Usage:
    python -m worldpgt.experiments.summarize_continuation_results \
        --input worldpgt/experiments/microworld_continuation_audited.csv
"""

from __future__ import annotations

import argparse
import csv
import json

from worldpgt.continuation.metrics import summarize_continuation_audit
from worldpgt.continuation.types import ContinuationAuditRow

VALID_LABELS = {"good", "bad", "unclear"}


def load_audited_rows(path: str) -> list[ContinuationAuditRow]:
    rows: list[ContinuationAuditRow] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "label" not in reader.fieldnames:
            raise ValueError(f"Input CSV {path} must contain a 'label' column (good/bad/unclear).")
        for record in reader:
            label = (record.get("label", "") or "").strip().lower()
            if label not in VALID_LABELS:
                label = "unclear"
            rows.append(
                ContinuationAuditRow(
                    prompt=record.get("prompt", ""),
                    continuation=record.get("continuation", ""),
                    ambiguous_term=record.get("ambiguous_term") or None,
                    selected_sense=record.get("selected_sense") or None,
                    expected_sense=record.get("expected_sense") or None,
                    label=label,
                    notes=record.get("notes", "") or "",
                )
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize audited continuation results as JSON metrics.")
    parser.add_argument("--input", required=True, help="Audited CSV with a label column (good/bad/unclear)")
    args = parser.parse_args()

    rows = load_audited_rows(args.input)
    metrics = summarize_continuation_audit(rows)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
