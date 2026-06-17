"""Candidate Knowledge Request v1 (proposal/request layer only).

Turns the Self-Learning Loop v1 ``missing_stable_knowledge`` opportunities
(confirmed by the Curriculum Regression Gate v1 to currently audit safely) into
structured *knowledge requests* and *local source document requests*.

This is NOT web search, NOT automatic ingestion, NOT accepted-memory mutation,
and NOT overlay mutation. It only emits request artifacts describing what local
source documents could be authored and ingested LATER, with the safety
constraints that must remain enforced and the audits that must continue until
explicit evidence exists.

Strictly offline and read-only:

- No network / Wikipedia / API calls.
- No neural / GPT / embedding / training / backprop / torch code.
- No generic fallback and no live/current claims.
- No activation of proposed analyzer rules or extraction patterns.
- No auto-create, auto-ingest, auto-promote, or trusted-memory write.
"""

from __future__ import annotations

from .types import (
    ALLOWED_ACTIVATION_STATUSES,
    DANGEROUS_OPPORTUNITY_CATEGORIES,
    REQUEST_CATEGORIES,
    CandidateKnowledgeRequest,
    KnowledgeRequestReport,
    MissingKnowledgeInput,
    SourceDocumentRequest,
)

__all__ = [
    "ALLOWED_ACTIVATION_STATUSES",
    "DANGEROUS_OPPORTUNITY_CATEGORIES",
    "REQUEST_CATEGORIES",
    "CandidateKnowledgeRequest",
    "KnowledgeRequestReport",
    "MissingKnowledgeInput",
    "SourceDocumentRequest",
]
