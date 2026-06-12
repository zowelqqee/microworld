"""Deterministic controlled continuation engine.

Pipeline: parse prompt -> score senses -> apply policy -> emit constrained continuation
with reasons, memory hits, and per-sense scores for auditing.
"""

from __future__ import annotations

from typing import Optional

from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.prompt_parser import parse_continuation_prompt
from worldpgt.continuation.realization import classify_prompt_ending, compose_continuation
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.continuation.surface_validator import validate_surface_text
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
                continuation = compose_continuation(prompt, selected_entry, ending_type)
                validation = validate_surface_text(prompt, continuation)
                if not validation.ok:
                    verdict.decision = "audit"
                    verdict.reasons.append("audit_reason=surface_realization_risk")
                    verdict.reasons.extend(validation.reasons)
                    for pattern in validation.matched_patterns:
                        memory_hits.append(f"surface_risk={pattern}")
                    continuation = ""
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
