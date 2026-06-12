"""Deterministic summary metrics over audited continuation rows."""

from __future__ import annotations

from typing import Iterable

from worldpgt.continuation.types import ContinuationAuditRow


def summarize_continuation_audit(rows: Iterable[ContinuationAuditRow]) -> dict:
    rows = list(rows)
    total = len(rows)
    good = sum(1 for row in rows if row.label == "good")
    bad = sum(1 for row in rows if row.label == "bad")
    unclear = sum(1 for row in rows if row.label == "unclear")

    good_rate = good / total if total else 0.0
    bad_rate = bad / total if total else 0.0
    unclear_rate = unclear / total if total else 0.0
    precision = good / (good + bad) if (good + bad) else 0.0

    return {
        "total": total,
        "good": good,
        "bad": bad,
        "unclear": unclear,
        "good_rate": good_rate,
        "bad_rate": bad_rate,
        "unclear_rate": unclear_rate,
        "precision": precision,
    }
