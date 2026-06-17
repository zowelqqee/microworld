"""Core dataclasses for Multi-hop QA v1.

Deterministic 2-hop relation chain reasoning over explicit overlay facts.
No ML. No network. All hops must be overlay_relation items.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

MultihopChainType = Literal[
    "connection",   # "How is A connected to C?" / "What links A to C?"
    "through_node", # "Is A connected to C through B?"
    "develops_isa", # "What does A develop that is a C?"
    "who_through",  # "Who is A linked to through B?"
    "direct_only",  # "Who founded A?" — must audit; transitive founder/owner guard
    "unsupported",  # No recognized multi-hop pattern
]

MultihopDecision = Literal["answer", "audit"]


@dataclass
class MultihopQuestion:
    question: str
    chain_type: MultihopChainType
    endpoint_a: Optional[str] = None
    endpoint_c: Optional[str] = None
    through_b: Optional[str] = None
    is_direct_relation: bool = False


@dataclass
class HopEdge:
    subject: str
    predicate: str
    object: str
    overlay_type: str = "overlay_relation"
    trust: str = ""
    stability: str = ""
    risk: str = ""
    source_page: str = ""

    def display(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.object}"

    def to_detail_dict(self) -> dict:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "overlay_type": self.overlay_type,
            "trust": self.trust,
            "stability": self.stability,
            "risk": self.risk,
            "source_page": self.source_page,
        }


@dataclass
class MultihopPath:
    hop1: HopEdge
    hop2: HopEdge
    via_node: str


@dataclass
class MultihopEvidence:
    hops_used: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"hops_used": list(self.hops_used)}


@dataclass
class MultihopPlan:
    question: MultihopQuestion
    decision: MultihopDecision
    audit_reason: Optional[str]
    path: Optional[MultihopPath]
    chain_label: str
    render_args: dict
    evidence: MultihopEvidence = field(default_factory=MultihopEvidence)


@dataclass
class MultihopResult:
    row_id: str
    question: str
    chain_type: str
    decision: MultihopDecision
    answer_text: str
    chain_label: str
    hop1: Optional[str]       # backward-compat display string
    hop2: Optional[str]       # backward-compat display string
    hop1_detail: Optional[dict]  # structured hop metadata
    hop2_detail: Optional[dict]  # structured hop metadata
    audit_reason: Optional[str]
    is_safe: bool
    unsafe_reason: Optional[str]

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "question": self.question,
            "chain_type": self.chain_type,
            "decision": self.decision,
            "answer_text": self.answer_text,
            "chain_label": self.chain_label,
            "hop1": self.hop1,
            "hop2": self.hop2,
            "hop1_detail": self.hop1_detail,
            "hop2_detail": self.hop2_detail,
            "audit_reason": self.audit_reason,
            "is_safe": self.is_safe,
            "unsafe_reason": self.unsafe_reason,
        }


@dataclass
class MultihopSummary:
    total_prompts: int = 0
    answer_count: int = 0
    audit_count: int = 0
    wrong_count: int = 0
    unsupported_answer_count: int = 0
    unsafe_chain_count: int = 0
    missing_hop_audit_count: int = 0
    # Preferred name: questions correctly guarded by direct-relation check.
    direct_only_guard_count: int = 0
    # Backward-compat alias — same value, kept so old artifact consumers don't break.
    direct_relation_leak_count: int = 0
    volatile_hop_audit_count: int = 0
    high_risk_hop_audit_count: int = 0
    current_sensitive_hop_audit_count: int = 0
    transitive_founder_leak_count: int = 0
    transitive_owner_leak_count: int = 0
    all_critical_passed: bool = False

    def to_dict(self) -> dict:
        return {
            "total_prompts": self.total_prompts,
            "answer_count": self.answer_count,
            "audit_count": self.audit_count,
            "wrong_count": self.wrong_count,
            "unsupported_answer_count": self.unsupported_answer_count,
            "unsafe_chain_count": self.unsafe_chain_count,
            "missing_hop_audit_count": self.missing_hop_audit_count,
            "direct_only_guard_count": self.direct_only_guard_count,
            "direct_relation_leak_count": self.direct_relation_leak_count,
            "volatile_hop_audit_count": self.volatile_hop_audit_count,
            "high_risk_hop_audit_count": self.high_risk_hop_audit_count,
            "current_sensitive_hop_audit_count": self.current_sensitive_hop_audit_count,
            "transitive_founder_leak_count": self.transitive_founder_leak_count,
            "transitive_owner_leak_count": self.transitive_owner_leak_count,
            "all_critical_passed": self.all_critical_passed,
        }
