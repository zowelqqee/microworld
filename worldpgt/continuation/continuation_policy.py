"""Deterministic decision policy: continue, audit, or suppress."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

from worldpgt.continuation.sense_memory import SenseEvidence
from worldpgt.continuation.types import ContinuationPrompt


@dataclass
class PolicyDecision:
    decision: Literal["continue", "audit", "suppress"]
    selected_sense: Optional[str]
    confidence: float
    reasons: list[str] = field(default_factory=list)


class ContinuationPolicy:
    """Thresholded score/margin policy with banned-pattern suppression."""

    def __init__(
        self,
        min_score: float = 1.0,
        min_margin: float = 1.0,
        banned_patterns: Optional[list[str]] = None,
    ) -> None:
        self.min_score = min_score
        self.min_margin = min_margin
        self.banned_patterns = [p.lower() for p in (banned_patterns or [])]

    def decide(
        self,
        parsed: ContinuationPrompt,
        sense_scores: dict[str, float],
        continuations: Optional[dict[str, list[str]]] = None,
        evidence: Optional[SenseEvidence] = None,
    ) -> PolicyDecision:
        """`continuations` maps sense_id -> allowed continuations, used for banned-pattern checks."""
        if parsed.ambiguous_term is None:
            return PolicyDecision("audit", None, 0.0, ["no_ambiguous_term"])
        if not parsed.candidate_senses:
            return PolicyDecision("audit", None, 0.0, ["no_candidate_senses"])
        if not sense_scores or all(score == 0.0 for score in sense_scores.values()):
            reasons = ["all_sense_scores_zero"]
            if evidence is not None:
                if self._has_anti_cue_conflict(evidence):
                    reasons.append("audit_reason=anti_cue_conflict")
                elif any(evidence.anti_cues.values()):
                    reasons.append("audit_reason=anti_cue")
                if any(evidence.guard_failures.values()):
                    reasons.append("audit_reason=guard_failure")
                if any(evidence.negated_cues.values()):
                    reasons.append("audit_reason=only_negated_evidence")
            return PolicyDecision("audit", None, 0.0, reasons)

        # Deterministic ranking: score descending, then sense_id ascending.
        ranked = sorted(sense_scores.items(), key=lambda kv: (-kv[1], kv[0]))
        top_sense, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = top_score - second_score
        confidence = min(1.0, top_score / (top_score + second_score + 1.0))

        reasons = [
            f"top_sense={top_sense}",
            f"top_score={top_score:g}",
            f"second_score={second_score:g}",
            f"margin={margin:g}",
        ]
        if evidence is not None and evidence.conflict_detected:
            reasons.append("conflict_detected")

        if top_score < self.min_score:
            reasons.append(f"top_score_below_min_score={self.min_score:g}")
            return PolicyDecision("audit", None, confidence, reasons)

        if evidence is not None:
            top_anti_cues = evidence.anti_cues.get(top_sense, [])
            top_guard_failures = evidence.guard_failures.get(top_sense, [])
            if top_anti_cues and evidence.positive_cues.get(top_sense, []):
                reasons.append("audit_reason=anti_cue_conflict")
                return PolicyDecision("audit", None, confidence, reasons)
            if top_anti_cues:
                reasons.append("audit_reason=anti_cue")
                return PolicyDecision("audit", None, confidence, reasons)
            if top_guard_failures:
                reasons.append("audit_reason=guard_failure")
                return PolicyDecision("audit", None, confidence, reasons)
            if any(evidence.guard_failures.values()) and top_score <= self.min_score:
                reasons.append("audit_reason=guard_failure")
                return PolicyDecision("audit", None, confidence, reasons)
            top_negated_cues = evidence.negated_cues.get(top_sense, [])
            if top_negated_cues and top_score <= self.min_score:
                reasons.append("audit_reason=negated_top_sense")
                return PolicyDecision("audit", None, confidence, reasons)
            if evidence.conflict_detected and margin <= self.min_margin:
                reasons.append("audit_reason=low_margin_conflict")
                return PolicyDecision("audit", None, confidence, reasons)
            if evidence.conflict_detected and confidence <= 0.5:
                reasons.append("audit_reason=low_confidence_conflict")
                return PolicyDecision("audit", None, confidence, reasons)

        if margin < self.min_margin:
            reasons.append(f"margin_below_min_margin={self.min_margin:g}")
            return PolicyDecision("audit", None, confidence, reasons)

        if continuations is not None:
            selected_continuations = continuations.get(top_sense, [])
            candidate = selected_continuations[0] if selected_continuations else ""
            for pattern in self.banned_patterns:
                if pattern and pattern in candidate.lower():
                    reasons.append(f"banned_pattern={pattern}")
                    return PolicyDecision("suppress", top_sense, confidence, reasons)

        reasons.append("score_and_margin_above_thresholds")
        return PolicyDecision("continue", top_sense, confidence, reasons)

    def _has_anti_cue_conflict(self, evidence: SenseEvidence) -> bool:
        return any(
            evidence.anti_cues.get(sense_id, []) and evidence.positive_cues.get(sense_id, [])
            for sense_id in evidence.anti_cues
        )
