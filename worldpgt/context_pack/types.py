"""Deterministic data model for Working Context Pack v1.

Plain dataclasses only. No I/O, no runtime coupling.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

OVERLAY_ACCEPTED = "accepted"
OVERLAY_PROMOTED = "promoted"

# Trust markers that must never be treated as stable factual support.
TRUST_WEAK = "weak_context_only"
STABILITY_VOLATILE = "volatile"
STABLE_STABILITIES = frozenset({"stable", "semi_stable"})


@dataclass
class MatchedEntity:
    surface: str
    name: str
    matched_kind: str  # "entity" | "overlay_term"
    entity_id: Optional[str] = None
    entity_type: Optional[str] = None


@dataclass
class ContextDefinition:
    subject: str
    definition: str
    predicate: str
    stability: str
    evidence_text: str = ""


@dataclass
class ContextRelation:
    subject: str
    predicate: str
    object: str
    stability: str
    risk: str
    trust: str
    evidence_text: str = ""
    source_page: str = ""


@dataclass
class ContextWeakLink:
    source_page: str
    surface: str
    target: str
    relation: str
    strength: str
    trust: str


@dataclass
class ContextSourceFact:
    subject: str
    predicate: str
    object: str
    source_name: str
    as_of: str
    claim_type: str
    stability: str
    risk: str
    trust: str
    requires_recheck: bool = True
    evidence_text: str = ""


@dataclass
class ContextCandidatePath:
    nodes: List[str]
    relations: List[str]
    support: str  # "stable" | "semi_stable"


@dataclass
class ContextBlockedPath:
    nodes: List[str]
    reason: str  # e.g. weak_only_path | missing_explicit_stable_path | relation_inversion
    detail: str


@dataclass
class ContextMissingKnowledgeHint:
    request_id: str
    category: str
    needed_evidence_type: str
    suggested_source_topic: str
    must_remain_audit_until_supported: bool


@dataclass
class WorkingContextPack:
    question: str
    overlay_mode: str
    matched_entities: List[MatchedEntity] = field(default_factory=list)
    definitions: List[ContextDefinition] = field(default_factory=list)
    direct_relations: List[ContextRelation] = field(default_factory=list)
    one_hop_relations: List[ContextRelation] = field(default_factory=list)
    two_hop_relations: List[ContextRelation] = field(default_factory=list)
    weak_links: List[ContextWeakLink] = field(default_factory=list)
    source_facts: List[ContextSourceFact] = field(default_factory=list)
    candidate_paths: List[ContextCandidatePath] = field(default_factory=list)
    blocked_paths: List[ContextBlockedPath] = field(default_factory=list)
    missing_knowledge_hints: List[ContextMissingKnowledgeHint] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    summary: str = ""
    safe_for_answering: bool = False
    safe_for_general_runtime: bool = False


@dataclass
class ContextPackValidationResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    checks: dict = field(default_factory=dict)
