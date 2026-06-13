"""Deterministic controlled continuation engine.

Pipeline: parse prompt -> score senses -> apply policy -> build semantic frame ->
generate/validate/rank candidate continuations (semantic renderer v2) -> emit the
best safe candidate, with reasons, memory hits, and per-sense scores for auditing.
"""

from __future__ import annotations

from typing import Optional

from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.prompt_parser import parse_continuation_prompt
from worldpgt.continuation.realization import ENDING_NEUTRAL, classify_prompt_ending
from worldpgt.continuation.semantic_frame import build_semantic_frame
from worldpgt.continuation.semantic_renderer import (
    RENDERER_NAME,
    make_frame_candidates,
    rank_frame_candidates,
)
from worldpgt.continuation.prompt_tail_validator import (
    AUDIT_REASON as PROMPT_TAIL_AUDIT_REASON,
    validate_prompt_tail_compatibility,
)
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.continuation.surface_repair import repair_surface_candidate
from worldpgt.continuation.types import ContinuationResult


class ControlledContinuationEngine:
    def __init__(
        self,
        memory: Optional[ExplicitSenseMemory] = None,
        policy: Optional[ContinuationPolicy] = None,
    ) -> None:
        self.memory = memory if memory is not None else ExplicitSenseMemory()
        self.policy = policy if policy is not None else ContinuationPolicy()

    def continue_prompt(self, prompt: str) -> ContinuationResult:
        parsed = parse_continuation_prompt(prompt, self.memory)

        sense_scores: dict[str, float] = {}
        continuations: dict[str, list[str]] = {}
        memory_hits: list[str] = []
        evidence = None

        if parsed.ambiguous_term is not None:
            term = parsed.ambiguous_term
            memory_hits.append(f"term={term}")
            evidence = self.memory.score_senses_with_evidence(prompt, term)
            sense_scores = evidence.adjusted_scores
            for entry in self.memory.get_senses(term):
                continuations[entry.sense_id] = entry.continuations
            for sense_id, cues in evidence.positive_cues.items():
                for cue in cues:
                    memory_hits.append(f"cue={cue} -> {sense_id}")
                    memory_hits.append(f"positive_cue={cue} -> {sense_id}")
            for sense_id, cues in evidence.negated_cues.items():
                for cue in cues:
                    memory_hits.append(f"negated_cue={cue} -> {sense_id}")
            for sense_id, cues in evidence.anti_cues.items():
                for cue in cues:
                    memory_hits.append(f"anti_cue={cue} -> {sense_id}")
            for sense_id, failures in evidence.guard_failures.items():
                for failure in failures:
                    memory_hits.append(f"guard_failure={failure} -> {sense_id}")
            if evidence.conflict_detected:
                memory_hits.append("conflict_detected")

        verdict = self.policy.decide(parsed, sense_scores, continuations, evidence)

        continuation = ""
        if verdict.decision == "continue" and verdict.selected_sense is not None:
            ending_type = classify_prompt_ending(prompt)
            memory_hits.append(f"realization_type={ending_type}")
            selected_entry = next(
                (
                    entry
                    for entry in self.memory.get_senses(parsed.ambiguous_term or "")
                    if entry.sense_id == verdict.selected_sense
                ),
                None,
            )
            if selected_entry is not None:
                continuation = self._render(
                    prompt,
                    parsed.ambiguous_term or "",
                    verdict.selected_sense,
                    selected_entry,
                    ending_type,
                    evidence,
                    verdict,
                    memory_hits,
                )
        # audit and suppress both return an empty continuation: never hallucinate.

        if verdict.selected_sense is not None:
            memory_hits.append(f"selected_sense={verdict.selected_sense}")

        return ContinuationResult(
            prompt=prompt,
            continuation=continuation,
            ambiguous_term=parsed.ambiguous_term,
            selected_sense=verdict.selected_sense,
            confidence=verdict.confidence,
            decision=verdict.decision,
            reasons=list(verdict.reasons),
            memory_hits=memory_hits,
            sense_scores=sense_scores,
        )

    def _render(
        self,
        prompt: str,
        term: str,
        sense_id: str,
        selected_entry,
        ending_type: str,
        evidence,
        verdict,
        memory_hits: list[str],
    ) -> str:
        """Build, validate, and rank semantic candidates; emit the best or audit."""
        templates = selected_entry.continuation_templates
        phrases = (
            templates.get(ending_type)
            or templates.get(ENDING_NEUTRAL)
            or selected_entry.continuations
        )
        frame = build_semantic_frame(
            prompt,
            term,
            sense_id,
            list(evidence.evidence_notes) if evidence is not None else [],
        )
        memory_hits.append(f"semantic_frame_intent={frame.intent}")
        memory_hits.append(f"renderer={RENDERER_NAME}")

        candidates = make_frame_candidates(prompt, frame, list(phrases))
        memory_hits.append(f"candidate_count={len(candidates)}")

        best = rank_frame_candidates(prompt, candidates)
        if best is not None:
            phrase_part = best.text[len(prompt.rstrip()):].strip()
            memory_hits.append(f"selected_candidate={phrase_part}")

            # Surface repair runs after renderer v2 selects a candidate and
            # before final emission. Unsafe repairs route to audit, never emit.
            repair = repair_surface_candidate(prompt, best.text, frame)
            memory_hits.extend(repair.reasons)
            if repair.safe:
                tail = validate_prompt_tail_compatibility(prompt, repair.text, frame)
                if tail.passed:
                    memory_hits.append("prompt_tail_validator=passed")
                    if tail.rule_name is not None:
                        memory_hits.append(f"prompt_tail_rule={tail.rule_name}")
                    for item in tail.evidence:
                        memory_hits.append(f"prompt_tail_evidence={item}")
                    if repair.applied:
                        repaired_part = repair.text[len(prompt.rstrip()):].strip()
                        memory_hits.append(f"repaired_candidate={repaired_part}")
                    if tail.repair_applied:
                        tail_part = tail.text[len(prompt.rstrip()):].strip()
                        memory_hits.append("prompt_tail_repair=applied")
                        memory_hits.append(f"prompt_tail_repaired_candidate={tail_part}")
                    return tail.text

                self._audit_prompt_tail_incompatible(tail, verdict, memory_hits)
                return ""

            self._audit_no_safe_repaired_candidate(repair, verdict, memory_hits)
            return ""

        # No safe candidate: audit, never emit. Preserve surface-risk reporting.
        self._audit_no_safe_candidate(candidates, verdict, memory_hits)
        return ""

    @staticmethod
    def _audit_no_safe_candidate(candidates, verdict, memory_hits: list[str]) -> None:
        verdict.decision = "audit"
        surface_patterns: list[str] = []
        for candidate in candidates:
            for reason in candidate.reasons:
                if reason.startswith("surface_pattern="):
                    pattern = reason.split("=", 1)[1]
                    if pattern not in surface_patterns:
                        surface_patterns.append(pattern)
        if surface_patterns:
            verdict.reasons.append("audit_reason=surface_realization_risk")
            for pattern in surface_patterns:
                verdict.reasons.append(f"surface_pattern={pattern}")
                memory_hits.append(f"surface_risk={pattern}")
        verdict.reasons.append("audit_reason=no_safe_semantic_candidate")

    @staticmethod
    def _audit_no_safe_repaired_candidate(repair, verdict, memory_hits: list[str]) -> None:
        verdict.decision = "audit"
        for reason in repair.reasons:
            verdict.reasons.append(f"surface_repair_reason={reason}")
            memory_hits.append(f"surface_repair_reason={reason}")
        verdict.reasons.append(f"audit_reason={repair.audit_reason}")

    @staticmethod
    def _audit_prompt_tail_incompatible(tail, verdict, memory_hits: list[str]) -> None:
        verdict.decision = "audit"
        verdict.reasons.append("prompt_tail_validator=rejected")
        memory_hits.append("prompt_tail_validator=rejected")
        if tail.rule_name is not None:
            verdict.reasons.append(f"prompt_tail_rule={tail.rule_name}")
            memory_hits.append(f"prompt_tail_rule={tail.rule_name}")
        if tail.rejection_reason is not None:
            verdict.reasons.append(f"prompt_tail_rejection={tail.rejection_reason}")
            memory_hits.append(f"prompt_tail_rejection={tail.rejection_reason}")
        for item in tail.evidence:
            memory_hits.append(f"prompt_tail_evidence={item}")
        verdict.reasons.append(f"audit_reason={PROMPT_TAIL_AUDIT_REASON}")
