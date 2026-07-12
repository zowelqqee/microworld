"""Automated knowledge review gate v1.

Reads memory update proposals and classifies every proposed item as:
  accepted_auto  — concrete, narrow, non-conflicting; safe to queue for future apply
  rejected_auto  — broad, conflicting, or generic; discarded
  needs_review   — uncertain; routed to the small human-review bucket

Does NOT modify sense_memory.py, continuation engine, cue memory,
guard rules, renderer, validators, thresholds, or benchmark outputs.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal

from worldpgt.knowledge.auto_review_types import (
    SAFETY_CHECKS_BEFORE_APPLY,
    AutoReviewOutput,
    ItemDecision,
    ReviewedItem,
    ReviewedProposal,
    ReviewedUpdate,
    ReviewPolicy,
    ReviewSummary,
)

# ---------------------------------------------------------------------------
# Decision vocabularies (all lowercase, deterministic)
# ---------------------------------------------------------------------------

# Broad single-word cues — always rejected when fact_type is a cue type.
# Mirrors the normalizer denylist so the gate acts as a second safety net.
_BROAD_DENYLIST: frozenset[str] = frozenset({
    "water", "light", "thing", "object", "place", "time",
    "people", "near", "under", "over", "good", "bad",
    "big", "small", "use", "used", "make", "made",
    "area", "surface", "side", "part", "end", "top",
    "move", "go", "get", "see", "come", "take",
    "long", "large", "high", "low", "new", "old",
})

# Narrow multiword phrases accepted as concrete cues despite containing broad words.
_NARROW_MULTIWORD_EXCEPTIONS: frozenset[str] = frozenset({
    "ocean water",
    "construction site",
    "wax seal",
    "river bank",
    "steel beams",
    "tower crane",
    "home run",
})

# Semantic frame labels already established in the project's frame vocabulary.
# Frames not in this set are routed to needs_review.
_KNOWN_SAFE_FRAMES: frozenset[str] = frozenset({
    "animal_behavior",
    "bird_behavior",
    "document_closure",
    "financial_transaction",
    "natural_geography",
    "sports_equipment_use",
})

# Phrase candidates explicitly confirmed safe for the continuation surface.
_KNOWN_SAFE_PHRASES: frozenset[str] = frozenset({
    "dove below the surface",
    "opened an account",
    "snapped back into place",
    "lifted the load",
    "spread its wings",
    "sealed it shut",
})

# Generic phrases that must never be auto-accepted.
_GENERIC_PHRASES: frozenset[str] = frozenset({
    "did something",
    "moved around",
    "stayed there",
    "was important",
    "continued",
    "did it",
    "went there",
    "was there",
    "happened",
})

# Generic verbs whose presence as the leading verb of a typical_action → rejected.
_GENERIC_ACTION_VERBS: frozenset[str] = frozenset({
    "go", "move", "do", "be", "have", "get", "make", "take", "use",
    "come", "give", "look", "seem", "become", "show", "continue",
    "happen", "exist", "remain", "stay", "keep",
})

# Broad location words that are rejected when used as typical_location values.
_BROAD_LOCATIONS: frozenset[str] = frozenset({
    "place", "area", "region", "location", "environment", "zone",
    "somewhere", "anywhere", "everywhere",
})

# Location values that are not explicitly broad but vague enough to need review.
_NEEDS_REVIEW_LOCATIONS: frozenset[str] = frozenset({
    "outdoor", "outdoors", "inside", "outside",
})

# Item types that correspond to cue-style fields (subject to broad-cue checking).
_CUE_TYPES: frozenset[str] = frozenset({"positive_cue", "anti_cue"})

# Map from proposed_update field name → item_type label used in ReviewedItem.
_FIELD_TO_ITEM_TYPE: dict[str, str] = {
    "positive_cues":        "positive_cue",
    "anti_cues":            "anti_cue",
    "typical_actions":      "typical_action",
    "typical_locations":    "typical_location",
    "parts":                "part",
    "objects":              "object",
    "semantic_frame_hints": "semantic_frame_hint",
    "phrase_candidate_hints": "phrase_candidate_hint",
}

# Inverse map for routing accepted/rejected/needs_review values back into slots.
_ITEM_TYPE_TO_FIELD: dict[str, str] = {v: k for k, v in _FIELD_TO_ITEM_TYPE.items()}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_conflicting_values(conflicting_cues: list[str]) -> frozenset[str]:
    """Extract bare cue values from evidence strings like 'wing (conflicts: machine)'."""
    values: set[str] = set()
    for entry in conflicting_cues:
        bare = entry.split(" (conflicts:")[0].strip()
        values.add(bare)
    return frozenset(values)


def _leading_verb(action: str) -> str:
    return action.split()[0] if action else ""


def _risk_for_decision(decision: ItemDecision, is_broad_conflict: bool = False) -> str:
    if decision == "accepted_auto":
        return "low"
    if decision == "rejected_auto":
        return "high" if is_broad_conflict else "medium"
    return "medium"  # needs_review


# ---------------------------------------------------------------------------
# Item-level decision functions
# ---------------------------------------------------------------------------

def _decide_cue(
    value: str,
    conflicting_values: frozenset[str],
) -> tuple[ItemDecision, str, bool]:
    """Returns (decision, reason, is_broad_or_conflict)."""
    if value in _BROAD_DENYLIST:
        return "rejected_auto", "broad_cue_in_denylist", True
    if value in conflicting_values:
        return "needs_review", "conflicting_cue_requires_human_review", True
    if " " in value:
        if value in _NARROW_MULTIWORD_EXCEPTIONS:
            return "accepted_auto", "narrow_multiword_exception_concrete", False
        return "needs_review", "multiword_cue_needs_guard_rule_review", False
    return "accepted_auto", "concrete_term_specific_non_conflicting_cue", False


def _decide_typical_action(value: str) -> tuple[ItemDecision, str, bool]:
    verb = _leading_verb(value)
    if verb in _GENERIC_ACTION_VERBS:
        return "rejected_auto", "generic_action_verb_too_broad", False
    if value in _GENERIC_PHRASES:
        return "rejected_auto", "generic_phrase_as_action_rejected", False
    return "accepted_auto", "concrete_action_non_generic", False


def _decide_typical_location(value: str) -> tuple[ItemDecision, str, bool]:
    if value in _BROAD_LOCATIONS:
        return "rejected_auto", "broad_location_too_generic", False
    if value in _NEEDS_REVIEW_LOCATIONS:
        return "needs_review", "vague_location_needs_review", False
    return "accepted_auto", "concrete_location", False


def _decide_semantic_frame_hint(value: str) -> tuple[ItemDecision, str, bool]:
    if value in _KNOWN_SAFE_FRAMES:
        return "accepted_auto", "known_compatible_frame_label", False
    return "needs_review", "new_frame_label_requires_project_review", False


def _decide_phrase_candidate(value: str) -> tuple[ItemDecision, str, bool]:
    if value in _GENERIC_PHRASES:
        return "rejected_auto", "generic_phrase_candidate_rejected", False
    if value in _KNOWN_SAFE_PHRASES:
        return "accepted_auto", "known_safe_concrete_phrase_candidate", False
    return "needs_review", "phrase_candidate_needs_prompt_tail_validation", False


def _decide_other(_item_type: str, _value: str) -> tuple[ItemDecision, str, bool]:
    # parts, objects: conservative default
    return "needs_review", "non_standard_item_type_requires_review", False


# ---------------------------------------------------------------------------
# Proposal-level decision
# ---------------------------------------------------------------------------

def _proposal_decision(items: list[ReviewedItem]) -> str:
    if not items:
        return "rejected_auto"
    n_accepted = sum(1 for i in items if i.decision == "accepted_auto")
    n_rejected = sum(1 for i in items if i.decision == "rejected_auto")
    n_review = sum(1 for i in items if i.decision == "needs_review")
    total = len(items)
    if n_accepted == total:
        return "accepted_auto"
    if n_rejected == total:
        return "rejected_auto"
    if n_accepted > 0:
        return "partial_accept"
    if n_review > 0:
        return "needs_review"
    return "rejected_auto"


# ---------------------------------------------------------------------------
# Main reviewer class
# ---------------------------------------------------------------------------

class AutomatedMemoryReviewer:
    """Applies deterministic item-level rules to every proposal.

    Reads proposals as plain dicts (from JSON) so no ingestion pipeline
    re-execution is needed. Produces an AutoReviewOutput with full audit trail.
    """

    def review(self, proposals: list[dict]) -> AutoReviewOutput:
        reviewed: list[ReviewedProposal] = []
        for prop in proposals:
            reviewed.append(self._review_proposal(prop))
        summary = self._build_summary(reviewed)
        policy = ReviewPolicy()
        return AutoReviewOutput(
            summary=summary,
            policy=policy,
            proposal_reviews=reviewed,
        )

    # ------------------------------------------------------------------
    # Proposal processing
    # ------------------------------------------------------------------

    def _review_proposal(self, proposal: dict) -> ReviewedProposal:
        term = proposal["term"]
        sense = proposal["sense"]
        source_risk = proposal.get("risk_level", "medium")
        prop_id = proposal["proposal_id"]
        evidence = proposal.get("evidence", {})
        conflicting_values = _parse_conflicting_values(
            evidence.get("conflicting_cues", [])
        )
        proposed_update = proposal.get("proposed_update", {})

        items: list[ReviewedItem] = []
        accepted = ReviewedUpdate()
        rejected = ReviewedUpdate()
        needs_review = ReviewedUpdate()

        for field_name, item_type in _FIELD_TO_ITEM_TYPE.items():
            values: list[str] = proposed_update.get(field_name, [])
            target_field = field_name  # same field name in ReviewedUpdate
            for value in values:
                decision, reason, is_bc = self._decide_item(
                    item_type, value, conflicting_values
                )
                risk = _risk_for_decision(decision, is_bc)
                items.append(ReviewedItem(
                    item_type=item_type,
                    value=value,
                    decision=decision,
                    risk_level=risk,
                    reason=reason,
                ))
                dest = (
                    accepted if decision == "accepted_auto"
                    else rejected if decision == "rejected_auto"
                    else needs_review
                )
                getattr(dest, target_field).append(value)

        pdecision = _proposal_decision(items)
        return ReviewedProposal(
            proposal_id=prop_id,
            term=term,
            sense=sense,
            source_risk_level=source_risk,
            proposal_decision=pdecision,
            items=items,
            accepted_update=accepted,
            rejected_items=rejected,
            needs_review_items=needs_review,
            safety_checks_required_before_apply=list(SAFETY_CHECKS_BEFORE_APPLY),
        )

    def _decide_item(
        self,
        item_type: str,
        value: str,
        conflicting_values: frozenset[str],
    ) -> tuple[ItemDecision, str, bool]:
        if item_type in _CUE_TYPES:
            return _decide_cue(value, conflicting_values)
        if item_type == "typical_action":
            return _decide_typical_action(value)
        if item_type == "typical_location":
            return _decide_typical_location(value)
        if item_type == "semantic_frame_hint":
            return _decide_semantic_frame_hint(value)
        if item_type == "phrase_candidate_hint":
            return _decide_phrase_candidate(value)
        return _decide_other(item_type, value)

    # ------------------------------------------------------------------
    # Summary construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_summary(reviewed: list[ReviewedProposal]) -> ReviewSummary:
        total_items = sum(len(r.items) for r in reviewed)
        accepted = sum(
            1 for r in reviewed for i in r.items if i.decision == "accepted_auto"
        )
        rejected = sum(
            1 for r in reviewed for i in r.items if i.decision == "rejected_auto"
        )
        needs = sum(
            1 for r in reviewed for i in r.items if i.decision == "needs_review"
        )

        by_term: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_item_type: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        by_decision: dict[str, int] = defaultdict(int)
        by_risk: dict[str, int] = defaultdict(int)

        for r in reviewed:
            for item in r.items:
                by_term[r.term][item.decision] += 1
                by_item_type[item.item_type][item.decision] += 1
                by_decision[item.decision] += 1
                by_risk[item.risk_level] += 1

        denom = total_items if total_items > 0 else 1
        return ReviewSummary(
            total_proposals=len(reviewed),
            total_items_reviewed=total_items,
            accepted_auto=accepted,
            rejected_auto=rejected,
            needs_review=needs,
            acceptance_rate=round(accepted / denom, 4),
            rejection_rate=round(rejected / denom, 4),
            needs_review_rate=round(needs / denom, 4),
            by_term={k: dict(v) for k, v in sorted(by_term.items())},
            by_item_type={k: dict(v) for k, v in sorted(by_item_type.items())},
            by_decision=dict(by_decision),
            by_risk_level=dict(by_risk),
        )
