"""Converts normalized fact candidates into reviewable memory update proposals.

Never writes to sense_memory.py or any live generation pipeline.
Proposals are read-only outputs for human audit.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict

from worldpgt.knowledge.types import (
    MemoryUpdateProposal,
    NormalizedFact,
    ProposalEvidence,
    ProposedUpdate,
    SAFETY_CHECKS,
)


def _stable_proposal_id(term: str, sense: str, source_ids: list[str]) -> str:
    key = f"{term}:{sense}:{','.join(sorted(source_ids))}"
    return "prop_" + hashlib.sha1(key.encode()).hexdigest()[:12]


def _recommended_action(
    risk: str,
    rejected_broad: list[str],
    conflicting: list[str],
) -> str:
    if risk == "high":
        return "human_review"
    if risk == "medium":
        return "human_review"
    # Low risk — but never auto_safe_later if broad/conflicting cues leaked through
    if rejected_broad or conflicting:
        return "human_review"
    return "auto_safe_later"


_FACT_TYPE_TO_FIELD: dict[str, str] = {
    "positive_cue": "positive_cues",
    "anti_cue": "anti_cues",
    "typical_action": "typical_actions",
    "typical_location": "typical_locations",
    "part": "parts",
    "object": "objects",
    "semantic_frame_hint": "semantic_frame_hints",
    "phrase_candidate_hint": "phrase_candidate_hints",
}


class MemoryUpdateProposer:
    """Groups normalized facts by (term, sense) and builds one proposal each."""

    def propose(
        self,
        normalized_facts: list[NormalizedFact],
        snippets_by_source_id: dict[str, object],
    ) -> list[MemoryUpdateProposal]:
        # Group by (term, sense)
        groups: dict[tuple[str, str], list[NormalizedFact]] = defaultdict(list)
        for nf in normalized_facts:
            groups[(nf.term, nf.sense)].append(nf)

        proposals: list[MemoryUpdateProposal] = []
        for (term, sense), facts in sorted(groups.items()):
            proposals.append(
                self._build_proposal(term, sense, facts, snippets_by_source_id)
            )
        return proposals

    def _build_proposal(
        self,
        term: str,
        sense: str,
        facts: list[NormalizedFact],
        snippets_by_source_id: dict[str, object],
    ) -> MemoryUpdateProposal:
        source_ids: list[str] = sorted({f.source_id for f in facts})
        source_titles = [
            getattr(snippets_by_source_id.get(sid), "title", sid)
            for sid in source_ids
        ]

        update = ProposedUpdate()
        rejected_broad: list[str] = []
        conflicting: list[str] = []
        matched_phrases: list[str] = []
        overall_risk = "low"

        for nf in facts:
            if nf.rejected:
                if nf.value not in rejected_broad:
                    rejected_broad.append(nf.value)
                continue
            if nf.conflict_senses:
                label = f"{nf.value} (conflicts: {','.join(nf.conflict_senses)})"
                if label not in conflicting:
                    conflicting.append(label)
                # Conflicting cues still get noted but are not placed into the update
                continue

            # Escalate overall risk
            if nf.risk_level == "high":
                overall_risk = "high"
            elif nf.risk_level == "medium" and overall_risk == "low":
                overall_risk = "medium"

            field = _FACT_TYPE_TO_FIELD.get(nf.fact_type)
            if field is None:
                continue
            target: list[str] = getattr(update, field)
            if nf.value not in target:
                target.append(nf.value)
            if nf.fact_type == "phrase_candidate_hint":
                if nf.value not in matched_phrases:
                    matched_phrases.append(nf.value)

        if rejected_broad or conflicting:
            overall_risk = max(overall_risk, "high", key=lambda x: ["low", "medium", "high"].index(x))

        recommended = _recommended_action(overall_risk, rejected_broad, conflicting)

        # Safety gate: proposals must never suggest threshold/validator changes
        # (enforced by design — proposals only populate the structured update fields)
        assert "threshold" not in str(update), "safety violation"
        assert "validator" not in str(update), "safety violation"

        evidence = ProposalEvidence(
            source_titles=source_titles,
            matched_phrases=matched_phrases,
            rejected_broad_cues=rejected_broad,
            conflicting_cues=conflicting,
        )

        return MemoryUpdateProposal(
            proposal_id=_stable_proposal_id(term, sense, source_ids),
            source_ids=source_ids,
            term=term,
            sense=sense,
            proposal_type="candidate_memory_update",
            risk_level=overall_risk,
            recommended_action=recommended,
            proposed_update=update,
            evidence=evidence,
            safety_checks_required=list(SAFETY_CHECKS),
        )
