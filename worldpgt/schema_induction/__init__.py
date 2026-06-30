"""Schema induction layer.

Turns an arbitrary corpus of documents into an evidence-bound symbolic
world graph WITHOUT predefining domain predicates:

    documents
      -> sentences
      -> entity mentions
      -> raw claims        (surface relations, not canonical predicates)
      -> argument frames   (generic semantic roles)
      -> relation families (clusters of repeated relation structure)
      -> local types       (induced, not a global enum)
      -> promotion gates    (generated | promoted | rejected)
      -> read-only QA adapter

No LLM inference. No ML training. Fully deterministic. Every artifact keeps
a source trace (doc -> sentence -> claim -> frame -> family -> answer).

This package lives ALONGSIDE the existing curated entity_qa / relation
extraction layers and never mutates accepted/promoted memory.
"""

from __future__ import annotations

from worldpgt.schema_induction.types import (
    ArgumentFrame,
    DocumentRecord,
    EntityMention,
    LocalType,
    PromotionDecision,
    RawClaim,
    RelationFamily,
    SchemaInductionResult,
    SentenceRecord,
)

__all__ = [
    "ArgumentFrame",
    "DocumentRecord",
    "EntityMention",
    "LocalType",
    "PromotionDecision",
    "RawClaim",
    "RelationFamily",
    "SchemaInductionResult",
    "SentenceRecord",
]
