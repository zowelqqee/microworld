"""Knowledge ingestion pipeline v1 — read-only proposal layer.

Raw source text is not memory. This package produces reviewable candidate
memory updates only. No module here writes to sense_memory.py or modifies
any generation behavior.
"""

from worldpgt.knowledge.types import (
    CuratedSnippet,
    FactCandidate,
    NormalizedFact,
    MemoryUpdateProposal,
)
from worldpgt.knowledge.curated_source_reader import CuratedSourceReader
from worldpgt.knowledge.fact_candidate_extractor import FactCandidateExtractor
from worldpgt.knowledge.fact_normalizer import FactNormalizer
from worldpgt.knowledge.memory_update_proposer import MemoryUpdateProposer

__all__ = [
    "CuratedSnippet",
    "FactCandidate",
    "NormalizedFact",
    "MemoryUpdateProposal",
    "CuratedSourceReader",
    "FactCandidateExtractor",
    "FactNormalizer",
    "MemoryUpdateProposer",
]
