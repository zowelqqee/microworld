"""Self-Learning Loop v1 (proposal layer only).

This package mines learning opportunities from EXISTING Microworld audit /
QA / quarantine / rejection / promotion / regression artifacts and turns them
into *structured proposals*. It never trains, never mutates trusted memory,
never writes overlays, never activates analyzer rules or extraction patterns,
and never changes runtime behavior.

It is strictly diagnostic and offline:

- No neural weights, GPT renderer, embeddings, training, backprop, or torch.
- No network / Wikipedia / API calls.
- No generic fallback answering and no live/current claims.
- No auto-promotion into accepted memory.
- No auto-application of proposed analyzer rules or extraction patterns.

Everything produced here carries an ``activation_status`` drawn from
``ALLOWED_ACTIVATION_STATUSES`` (which deliberately has no ``auto_apply``).
"""

from __future__ import annotations

from .types import (
    ALLOWED_ACTIVATION_STATUSES,
    OPPORTUNITY_CATEGORIES,
    PROPOSAL_ONLY,
    REQUIRES_HUMAN_REVIEW,
    REQUIRES_INGESTION,
    REQUIRES_REGRESSION_GATE,
    AnalyzerRuleProposal,
    CurriculumRow,
    AdversarialCase,
    LearningOpportunity,
    LearningSignal,
    PatternProposal,
    SelfLearningReport,
)

__all__ = [
    "ALLOWED_ACTIVATION_STATUSES",
    "OPPORTUNITY_CATEGORIES",
    "PROPOSAL_ONLY",
    "REQUIRES_HUMAN_REVIEW",
    "REQUIRES_INGESTION",
    "REQUIRES_REGRESSION_GATE",
    "AnalyzerRuleProposal",
    "CurriculumRow",
    "AdversarialCase",
    "LearningOpportunity",
    "LearningSignal",
    "PatternProposal",
    "SelfLearningReport",
]
