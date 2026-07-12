"""Detect snapshot facts that need freshness recheck."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from worldpgt.knowledge.temporal_classification import classify_temporal_class
from worldpgt.relation_extraction_v2.relation_policy import freshness_window_days


@dataclass
class StaleCandidate:
    subject: str
    predicate: str
    object: str
    as_of: str
    age_days: int
    freshness_window: float
    staleness_ratio: float
    source_page: str = ""
    temporal_class: str = "snapshot"
    is_stale: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "as_of": self.as_of,
            "age_days": self.age_days,
            "freshness_window": self.freshness_window,
            "staleness_ratio": self.staleness_ratio,
            "source_page": self.source_page,
            "temporal_class": self.temporal_class,
            "is_stale": self.is_stale,
        }


def _load_overlay(overlay: str | Path | Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(overlay, (str, Path)):
        rows = json.loads(Path(overlay).read_text(encoding="utf-8"))
        return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
    return [row for row in overlay if isinstance(row, dict)]


def _coerce_date(value: str | date | datetime | None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return datetime.now(timezone.utc).date()
    if len(text) == 7 and text[4] == "-":
        return date(int(text[:4]), int(text[5:7]), 1)
    if len(text) >= 10:
        return date.fromisoformat(text[:10])
    return date.fromisoformat(text)


def _fact_object(item: dict[str, Any]) -> str:
    if item.get("overlay_type") == "overlay_definition":
        return str(item.get("definition", ""))
    return str(item.get("object", ""))


def detect_stale_candidates(
    overlay: str | Path | Iterable[dict[str, Any]],
    current_date: str | date | datetime | None = None,
) -> list[StaleCandidate]:
    """Return snapshot facts with as_of, sorted by staleness ratio descending.

    The returned list includes not-yet-expired snapshot facts too. Their
    ``staleness_ratio`` still drives the recheck priority, while ``is_stale``
    records whether the freshness window has actually been exceeded.
    """
    today = _coerce_date(current_date)
    candidates: list[StaleCandidate] = []

    for item in _load_overlay(overlay):
        predicate = str(item.get("predicate") or ("is_a" if item.get("overlay_type") == "overlay_definition" else ""))
        temporal_class = str(
            item.get("temporal_class")
            or classify_temporal_class(
                predicate,
                item.get("stability"),
                overlay_type=item.get("overlay_type"),
                claim_type=item.get("claim_type"),
            )
            or ""
        )
        if temporal_class != "snapshot":
            continue
        as_of = str(item.get("as_of") or "").strip()
        if not as_of:
            continue
        window = freshness_window_days(predicate, temporal_class)
        if math.isinf(window) or window <= 0:
            continue
        age_days = max(0, (today - _coerce_date(as_of)).days)
        ratio = age_days / float(window)
        candidates.append(
            StaleCandidate(
                subject=str(item.get("subject", "")),
                predicate=predicate,
                object=_fact_object(item),
                as_of=as_of,
                age_days=age_days,
                freshness_window=window,
                staleness_ratio=round(ratio, 4),
                source_page=str(item.get("source_page", "")),
                temporal_class=temporal_class,
                is_stale=ratio >= 1.0,
            )
        )

    return sorted(
        candidates,
        key=lambda c: (-c.staleness_ratio, c.subject.casefold(), c.predicate.casefold()),
    )
