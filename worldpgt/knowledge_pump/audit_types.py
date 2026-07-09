"""Types for the structured audit event log and gap analysis report."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AuditLogEntry:
    timestamp: str      # ISO 8601 UTC
    question: str
    entity: str         # extracted entity name, "" if not extracted
    support_kind: str   # raw support_kind from AssistantAnswer, or "multihop_audit"
    reason: str         # human-readable reason text
    source: str         # "assistant_surface" | "multihop"
    temporal_mismatch: bool = False


@dataclass
class EntityGapEntry:
    entity: str
    gap_type: str       # "missing_facts" | "unknown_entity" | "policy_blocked"
    count: int
    top_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "gap_type": self.gap_type,
            "count": self.count,
            "top_reasons": list(self.top_reasons),
        }


@dataclass
class GapReport:
    generated_at: str
    period_days: int
    total_audit_events: int
    acquisition_candidates: list[EntityGapEntry]  # missing_facts + unknown_entity
    policy_blocked: list[EntityGapEntry]          # excluded from frontier
    stale_candidates: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "generated_at": self.generated_at,
            "period_days": self.period_days,
            "total_audit_events": self.total_audit_events,
            "acquisition_candidates": [e.to_dict() for e in self.acquisition_candidates],
            "policy_blocked": [e.to_dict() for e in self.policy_blocked],
            "stale_candidates": [e.to_dict() for e in self.stale_candidates],
        }
