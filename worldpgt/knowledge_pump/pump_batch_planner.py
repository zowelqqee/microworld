"""Batch planning for Knowledge Pump v1."""

from __future__ import annotations

from worldpgt.knowledge_pump.title_ranker import normalize_title
from worldpgt.knowledge_pump.types import ExpandedAllowlistEntry


def plan_batches(
    allowlist: list[ExpandedAllowlistEntry],
    batch_size: int,
    max_batches: int,
    already_fetched_titles: set[str] | None = None,
    start_batch_index: int = 0,
) -> list[list[ExpandedAllowlistEntry]]:
    fetched = {normalize_title(t).casefold() for t in (already_fetched_titles or set())}
    remaining = [
        e for e in allowlist
        if e.normalized_title.casefold() not in fetched and e.batch_index >= start_batch_index
    ]
    batches: list[list[ExpandedAllowlistEntry]] = []
    for idx in range(max_batches):
        batch = remaining[idx * batch_size:(idx + 1) * batch_size]
        if not batch:
            break
        batches.append(batch)
    return batches

