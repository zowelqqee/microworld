"""Context-Pack QA Consistency Gate v1 (dry-run only).

Compares EXISTING QA decisions (entity / expansion / adversarial / cross-page
outputs) against the read-only Working Context Pack built for the same question.
It answers: when QA answers, is there explicit stable/semi-stable support? When
QA audits, does the context show blocked / weak-only / missing / current /
volatile / inversion context? Are weak links or volatile facts ever treated as
stable support?

This is strictly a dry-run consistency probe:

- It does NOT change QA behavior and does NOT wire the context pack into planners.
- It does NOT write overlays / accepted memory / promoted overlay.
- It reuses the existing context pack builder + validator (no duplicated logic).
- No neural / GPT / embedding / training / network code.

It may emit passed, warning, and failed rows, and reports them honestly. Safety
is never weakened to make the gate green.
"""

from __future__ import annotations

from .types import (
    ContextConsistencyReport,
    ContextConsistencyResult,
    ContextConsistencySummary,
    LoadedQAResult,
)

__all__ = [
    "ContextConsistencyReport",
    "ContextConsistencyResult",
    "ContextConsistencySummary",
    "LoadedQAResult",
]
