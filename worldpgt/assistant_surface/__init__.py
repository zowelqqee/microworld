"""Microworld Assistant Surface v1 — controlled user-facing assistant entrypoint.

A single deterministic surface over existing explicit memory, overlays, context
packs, and safety gates. It routes a user question to the safest existing QA
path and returns a normal user-facing answer, or audits honestly when no
supported answer exists.

This is NOT GPT. NOT a free-form generator. NOT runtime web search. NOT neural
inference. There is no training, no embeddings, no backprop, no torch, and no
network access. It never writes accepted memory, accepted overlay, promoted
overlay, or the snapshot dry-run overlay, and never changes the runtime
behavior of existing QA planners.

``safe_for_general_runtime`` is always False — this is an experiment surface.
"""

from worldpgt.assistant_surface.types import (
    AssistantAnswer,
    AssistantBenchmarkResult,
    AssistantBenchmarkRow,
    AssistantContextSummary,
    AssistantRequest,
    AssistantRoute,
    AssistantSurfaceSummary,
    AssistantTrace,
)

__all__ = [
    "AssistantAnswer",
    "AssistantBenchmarkResult",
    "AssistantBenchmarkRow",
    "AssistantContextSummary",
    "AssistantRequest",
    "AssistantRoute",
    "AssistantSurfaceSummary",
    "AssistantTrace",
]
