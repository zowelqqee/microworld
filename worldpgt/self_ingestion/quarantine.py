"""Quarantine builder for Wikipedia Self-Ingestion v1.

Turns conflict/risk reasons into deterministic quarantine items with a fixed
reason enum, risk level, and suggested action. No ML, no network.
"""

from __future__ import annotations

from worldpgt.self_ingestion.types import (
    ACTION_HUMAN_REVIEW,
    REASON_POLICY,
    REASON_UNKNOWN_FORMAT,
    QuarantineItem,
)


def make_quarantine_item(
    candidate_id: str, text: str, reason: str, source_id: str,
) -> QuarantineItem:
    risk, action = REASON_POLICY.get(reason, ("low", ACTION_HUMAN_REVIEW))
    return QuarantineItem(
        candidate_id=candidate_id,
        text=text,
        reason=reason if reason in REASON_POLICY else REASON_UNKNOWN_FORMAT,
        risk=risk,
        source_id=source_id,
        suggested_action=action,
    )
