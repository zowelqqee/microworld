"""Core dataclasses for Microworld Assistant Surface v1.

Plain dataclasses only. Deterministic and rule-based. No ML, no embeddings,
no network, no I/O coupling beyond simple serialization helpers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional

# --------------------------------------------------------------------------- #
# Overlay modes.
# --------------------------------------------------------------------------- #
OVERLAY_MODE_ACCEPTED = "accepted"
OVERLAY_MODE_PROMOTED = "promoted"
OVERLAY_MODE_SNAPSHOT_DRY_RUN = "snapshot-dry-run"
OVERLAY_MODE_PUMP_DRY_RUN = "pump-dry-run"
OVERLAY_MODE_CUSTOM_PATH = "custom-overlay-path"

OVERLAY_MODES = (
    OVERLAY_MODE_ACCEPTED,
    OVERLAY_MODE_PROMOTED,
    OVERLAY_MODE_SNAPSHOT_DRY_RUN,
    OVERLAY_MODE_PUMP_DRY_RUN,
)

# --------------------------------------------------------------------------- #
# Router intents (conservative, deterministic).
# --------------------------------------------------------------------------- #
AssistantIntent = Literal[
    "entity_definition",
    "entity_relation",
    "connection_path",
    "source_qualified_fact",
    "weak_link_policy",
    "current_live_request",
    "private_sensitive_request",
    "relation_inversion",
    "unsupported_universal",
    "creative_request",
    "unknown_or_unsupported",
]

AssistantDecision = Literal["answer", "audit", "no"]

# Allowed support kinds.
SupportKind = Literal[
    "stable_definition",
    "stable_relation",
    "semi_stable_relation",
    "explicit_connection_path",
    "explicit_is_a_chain",
    "explicit_type_contradiction",
    "entity_type_mismatch",
    "source_qualified_fact",
    "safe_policy_answer",
    "web_search_result",
    "audit_blocked_context",
    "missing_knowledge",
    "creative_generated",
    "unsupported",
]

ALLOWED_SUPPORT_KINDS = frozenset(
    {
        "stable_definition",
        "stable_relation",
        "semi_stable_relation",
        "explicit_connection_path",
        "explicit_is_a_chain",
        "explicit_type_contradiction",
        "entity_type_mismatch",
        "source_qualified_fact",
        "safe_policy_answer",
        "web_search_result",
        "audit_blocked_context",
        "missing_knowledge",
        "unsupported",
    }
)

# Support kinds that constitute a genuine supported factual answer.
FACTUAL_SUPPORT_KINDS = frozenset(
    {
        "stable_definition",
        "stable_relation",
        "semi_stable_relation",
        "explicit_connection_path",
        "explicit_is_a_chain",
        "explicit_type_contradiction",
        "entity_type_mismatch",
        "source_qualified_fact",
        "web_search_result",
    }
)

# Hard-safety intents that must always audit.
HARD_SAFETY_INTENTS = frozenset(
    {
        "current_live_request",
        "private_sensitive_request",
        "relation_inversion",
        "unsupported_universal",
    }
)


@dataclass
class AssistantRequest:
    question: str
    overlay_mode: str = OVERLAY_MODE_PROMOTED
    json_mode: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "overlay_mode": self.overlay_mode,
            "json_mode": self.json_mode,
        }


@dataclass
class AssistantRoute:
    question: str
    intent: AssistantIntent
    subject: Optional[str] = None
    secondary: Optional[str] = None
    source_hint: Optional[str] = None
    is_hard_safety: bool = False
    risk_flags: List[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "intent": self.intent,
            "subject": self.subject,
            "secondary": self.secondary,
            "source_hint": self.source_hint,
            "is_hard_safety": self.is_hard_safety,
            "risk_flags": list(self.risk_flags),
            "notes": self.notes,
        }


@dataclass
class AssistantContextSummary:
    overlay_mode: str
    pack_valid: bool
    matched_entities: List[str] = field(default_factory=list)
    candidate_paths: List[str] = field(default_factory=list)
    blocked_paths: List[str] = field(default_factory=list)
    weak_links: List[str] = field(default_factory=list)
    source_facts: List[str] = field(default_factory=list)
    definitions: List[str] = field(default_factory=list)
    direct_relations: List[str] = field(default_factory=list)
    missing_hints: List[str] = field(default_factory=list)
    safety_notes: List[str] = field(default_factory=list)
    has_stable_definition: bool = False
    has_stable_relation: bool = False
    has_semi_stable_relation: bool = False
    has_explicit_path: bool = False
    has_source_fact: bool = False
    has_weak_only: bool = False
    safe_for_answering: bool = False

    def to_dict(self) -> dict:
        return {
            "overlay_mode": self.overlay_mode,
            "pack_valid": self.pack_valid,
            "matched_entities": list(self.matched_entities),
            "candidate_paths": list(self.candidate_paths),
            "blocked_paths": list(self.blocked_paths),
            "weak_links": list(self.weak_links),
            "source_facts": list(self.source_facts),
            "definitions": list(self.definitions),
            "direct_relations": list(self.direct_relations),
            "missing_hints": list(self.missing_hints),
            "safety_notes": list(self.safety_notes),
            "has_stable_definition": self.has_stable_definition,
            "has_stable_relation": self.has_stable_relation,
            "has_semi_stable_relation": self.has_semi_stable_relation,
            "has_explicit_path": self.has_explicit_path,
            "has_source_fact": self.has_source_fact,
            "has_weak_only": self.has_weak_only,
            "safe_for_answering": self.safe_for_answering,
        }


@dataclass
class AssistantTrace:
    steps: List[str] = field(default_factory=list)
    route_intent: str = ""
    context_summary: Optional[dict] = None
    cognitive_plan: Optional[dict] = None
    source_system: str = ""
    support_kind: str = "unsupported"
    risk_flags: List[str] = field(default_factory=list)

    def add(self, step: str) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict:
        return {
            "steps": list(self.steps),
            "route_intent": self.route_intent,
            "context_summary": self.context_summary,
            "cognitive_plan": self.cognitive_plan,
            "source_system": self.source_system,
            "support_kind": self.support_kind,
            "risk_flags": list(self.risk_flags),
        }


@dataclass
class AssistantAnswer:
    question: str
    decision: AssistantDecision
    route: str
    answer_text: str
    overlay_mode: str
    supported_by_context: bool
    support_kind: SupportKind
    risk_flags: List[str] = field(default_factory=list)
    source_system: str = ""
    trace: Optional[AssistantTrace] = None
    safe_for_general_runtime: bool = False

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "decision": self.decision,
            "route": self.route,
            "answer_text": self.answer_text,
            "overlay_mode": self.overlay_mode,
            "supported_by_context": self.supported_by_context,
            "support_kind": self.support_kind,
            "risk_flags": list(self.risk_flags),
            "source_system": self.source_system,
            "trace": self.trace.to_dict() if self.trace else None,
            "safe_for_general_runtime": self.safe_for_general_runtime,
        }


@dataclass
class AssistantBenchmarkRow:
    row_id: str
    question: str
    overlay_mode: str
    expected_decision: str = ""
    expected_intent: str = ""
    category: str = ""

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "question": self.question,
            "overlay_mode": self.overlay_mode,
            "expected_decision": self.expected_decision,
            "expected_intent": self.expected_intent,
            "category": self.category,
        }


@dataclass
class AssistantBenchmarkResult:
    row: AssistantBenchmarkRow
    answer: AssistantAnswer

    def to_dict(self) -> dict:
        return {
            "row": self.row.to_dict(),
            "answer": self.answer.to_dict(),
        }


@dataclass
class AssistantSurfaceSummary:
    prompt_total: int = 0
    answer_count: int = 0
    audit_count: int = 0
    supported_answer_count: int = 0
    safe_policy_answer_count: int = 0
    unsupported_audit_count: int = 0
    unsafe_answer_count: int = 0
    answer_without_context_support_count: int = 0
    weak_link_false_support_count: int = 0
    volatile_false_stable_count: int = 0
    current_live_false_support_count: int = 0
    private_false_support_count: int = 0
    inversion_false_support_count: int = 0
    universal_false_support_count: int = 0
    json_mode_supported: bool = True
    overlay_modes_tested: List[str] = field(default_factory=list)
    all_critical_passed: bool = False
    runtime_behavior_modified: bool = False
    trusted_memory_modified: bool = False
    accepted_overlay_modified: bool = False
    promoted_overlay_modified: bool = False
    snapshot_dry_run_overlay_modified: bool = False
    network_calls: bool = False
    safe_for_general_runtime: bool = False

    def to_dict(self) -> dict:
        return {
            "prompt_total": self.prompt_total,
            "answer_count": self.answer_count,
            "audit_count": self.audit_count,
            "supported_answer_count": self.supported_answer_count,
            "safe_policy_answer_count": self.safe_policy_answer_count,
            "unsupported_audit_count": self.unsupported_audit_count,
            "unsafe_answer_count": self.unsafe_answer_count,
            "answer_without_context_support_count": self.answer_without_context_support_count,
            "weak_link_false_support_count": self.weak_link_false_support_count,
            "volatile_false_stable_count": self.volatile_false_stable_count,
            "current_live_false_support_count": self.current_live_false_support_count,
            "private_false_support_count": self.private_false_support_count,
            "inversion_false_support_count": self.inversion_false_support_count,
            "universal_false_support_count": self.universal_false_support_count,
            "json_mode_supported": self.json_mode_supported,
            "overlay_modes_tested": list(self.overlay_modes_tested),
            "all_critical_passed": self.all_critical_passed,
            "runtime_behavior_modified": self.runtime_behavior_modified,
            "trusted_memory_modified": self.trusted_memory_modified,
            "accepted_overlay_modified": self.accepted_overlay_modified,
            "promoted_overlay_modified": self.promoted_overlay_modified,
            "snapshot_dry_run_overlay_modified": self.snapshot_dry_run_overlay_modified,
            "network_calls": self.network_calls,
            "safe_for_general_runtime": self.safe_for_general_runtime,
        }
