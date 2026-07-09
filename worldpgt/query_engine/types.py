"""QueryPlan and PlanResult dataclasses for the composable query engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from worldpgt.query_engine.primitives import Primitive


@dataclass
class QueryPlan:
    steps: list[Primitive]
    confidence: float
    question_type_hint: str


@dataclass
class PlanResult:
    success: bool
    data: Any
    trace: list[str]
    decision: str        # "answer" | "audit"
    audit_reason: str | None
    answer_text: str = ""
    support_kind: str = "stable_relation"
