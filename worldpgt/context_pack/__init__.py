"""Working Context Pack v1 (read-only probe layer).

Builds a structured "working context" for a question/claim from the EXPLICIT
wiki memory overlay (accepted or promoted). This is the Microworld analogue of a
context window, but it is structured evidence — not raw text, not token
stuffing, and not a generator.

It is strictly read-only and does NOT change QA behavior:

- No overlay / accepted-memory / promoted-overlay writes.
- No planner / validator / threshold / ingestion / overlay-builder change.
- No activation of proposed analyzer rules or extraction patterns.
- No auto-ingest, auto-promote, or auto-apply of any self-learning artifact.
- No neural / GPT / embedding / training / network code.

Safety invariants surfaced by every pack:

- ``safe_for_general_runtime`` is always False.
- ``safe_for_answering`` may be True only with an explicit stable/semi-stable
  support path; weak links and volatile source-qualified facts never make it
  True.
- Inverted-relation and unsupported-universal claims are blocked/safety-noted.
- Current/live and private/sensitive questions never produce stable context.
"""

from __future__ import annotations

from .types import (
    ContextBlockedPath,
    ContextCandidatePath,
    ContextDefinition,
    ContextMissingKnowledgeHint,
    ContextPackValidationResult,
    ContextRelation,
    ContextSourceFact,
    ContextWeakLink,
    MatchedEntity,
    WorkingContextPack,
)

__all__ = [
    "ContextBlockedPath",
    "ContextCandidatePath",
    "ContextDefinition",
    "ContextMissingKnowledgeHint",
    "ContextPackValidationResult",
    "ContextRelation",
    "ContextSourceFact",
    "ContextWeakLink",
    "MatchedEntity",
    "WorkingContextPack",
]
