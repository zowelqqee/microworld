"""Dataclasses for the automated knowledge review gate v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

ItemDecision = Literal["accepted_auto", "rejected_auto", "needs_review"]
ProposalDecision = Literal["accepted_auto", "partial_accept", "needs_review", "rejected_auto"]

SAFETY_CHECKS_BEFORE_APPLY: list[str] = [
    "run_trusted_benchmark",
    "wrong_continue_count_remains_0",
    "semantic_quality_flagged_remains_0",
    "true_unsafe_rows_remain_audited",
    "prompt_tail_gate_still_passes",
]


@dataclass
class ReviewedItem:
    """Item-level decision for one value inside a proposal's proposed_update."""

    item_type: str
    value: str
    decision: ItemDecision
    risk_level: Literal["low", "medium", "high"]
    reason: str


@dataclass
class ReviewedUpdate:
    """Accepted / rejected / needs-review split of one proposal's proposed_update."""

    positive_cues: list[str] = field(default_factory=list)
    anti_cues: list[str] = field(default_factory=list)
    typical_actions: list[str] = field(default_factory=list)
    typical_locations: list[str] = field(default_factory=list)
    parts: list[str] = field(default_factory=list)
    objects: list[str] = field(default_factory=list)
    semantic_frame_hints: list[str] = field(default_factory=list)
    phrase_candidate_hints: list[str] = field(default_factory=list)


@dataclass
class ReviewedProposal:
    """Auto-review result for one proposal."""

    proposal_id: str
    term: str
    sense: str
    source_risk_level: str
    proposal_decision: ProposalDecision
    items: list[ReviewedItem]
    accepted_update: ReviewedUpdate
    rejected_items: ReviewedUpdate
    needs_review_items: ReviewedUpdate
    safety_checks_required_before_apply: list[str] = field(
        default_factory=lambda: list(SAFETY_CHECKS_BEFORE_APPLY)
    )


@dataclass
class ReviewPolicy:
    """Immutable policy record: documents what the gate does and does not do."""

    mode: str = "automated_review_only"
    auto_apply: bool = False
    generation_behavior_changed: bool = False
    sense_memory_modified: bool = False
    human_review_required_for_uncertain_items: bool = True


@dataclass
class ReviewSummary:
    """Aggregate statistics across all reviewed proposals."""

    total_proposals: int = 0
    total_items_reviewed: int = 0
    accepted_auto: int = 0
    rejected_auto: int = 0
    needs_review: int = 0
    acceptance_rate: float = 0.0
    rejection_rate: float = 0.0
    needs_review_rate: float = 0.0
    by_term: dict = field(default_factory=dict)
    by_item_type: dict = field(default_factory=dict)
    by_decision: dict = field(default_factory=dict)
    by_risk_level: dict = field(default_factory=dict)


@dataclass
class AutoReviewOutput:
    """Top-level output of the automated review gate."""

    summary: ReviewSummary
    policy: ReviewPolicy
    proposal_reviews: list[ReviewedProposal]
