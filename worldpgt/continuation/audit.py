"""CSV read/write helpers for human audit rows. Stdlib csv only."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Union

from worldpgt.continuation.types import ContinuationAuditRow

AUDIT_FIELDS = [
    "prompt",
    "continuation",
    "ambiguous_term",
    "selected_sense",
    "expected_sense",
    "label",
    "notes",
]


def write_audit_rows(path: Union[str, Path], rows: Iterable[ContinuationAuditRow]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "prompt": row.prompt,
                    "continuation": row.continuation,
                    "ambiguous_term": row.ambiguous_term or "",
                    "selected_sense": row.selected_sense or "",
                    "expected_sense": row.expected_sense or "",
                    "label": row.label,
                    "notes": row.notes,
                }
            )


def read_audit_rows(path: Union[str, Path]) -> list[ContinuationAuditRow]:
    rows: list[ContinuationAuditRow] = []
    with open(path, "r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for record in reader:
            rows.append(
                ContinuationAuditRow(
                    prompt=record.get("prompt", ""),
                    continuation=record.get("continuation", ""),
                    ambiguous_term=record.get("ambiguous_term") or None,
                    selected_sense=record.get("selected_sense") or None,
                    expected_sense=record.get("expected_sense") or None,
                    label=record.get("label", "unclear") or "unclear",
                    notes=record.get("notes", "") or "",
                )
            )
    return rows
