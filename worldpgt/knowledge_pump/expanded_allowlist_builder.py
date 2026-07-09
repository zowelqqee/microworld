"""Build the expanded Knowledge Pump allowlist."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from worldpgt.knowledge_pump.title_ranker import normalize_title, rank_titles
from worldpgt.knowledge_pump.types import ExpandedAllowlistEntry, FrontierTitle


FIELDS = [
    "title", "normalized_title", "priority", "reason", "source", "risk_hint",
    "already_fetched", "selected_for_batch", "batch_index",
]


def build_expanded_allowlist(
    frontier: list[FrontierTitle],
    target_total: int,
    batch_size: int,
    already_fetched: set[str] | None = None,
) -> list[ExpandedAllowlistEntry]:
    fetched = {normalize_title(t).casefold() for t in (already_fetched or set())}
    entries: list[ExpandedAllowlistEntry] = []
    for item, priority, risk in rank_titles(frontier):
        if len(entries) >= target_total:
            break
        key = item.title.casefold()
        batch_index = len(entries) // batch_size
        entries.append(
            ExpandedAllowlistEntry(
                title=item.title,
                normalized_title=normalize_title(item.title),
                priority=priority,
                reason=item.reason,
                source=item.source,
                risk_hint=risk,
                already_fetched=key in fetched,
                selected_for_batch=key not in fetched and batch_index == 0,
                batch_index=batch_index,
            )
        )
    return entries


def write_allowlist(entries: list[ExpandedAllowlistEntry], json_path: str | Path, csv_path: str | Path) -> None:
    payload = [entry.to_dict() for entry in entries]
    jp = Path(json_path)
    cp = Path(csv_path)
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with cp.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in payload:
            writer.writerow(row)

