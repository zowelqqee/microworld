"""Bounded deterministic beam search over narrative hypotheses."""

from __future__ import annotations

from dataclasses import dataclass

from poemcore.narrative_reasoning import Hypothesis, HypothesisEvaluation, evaluate_hypothesis
from poemcore.world_state import WorldState


@dataclass(frozen=True)
class ReasoningPlan:
    initial_state: WorldState
    final_state: WorldState
    steps: tuple[Hypothesis, ...]
    evaluations: tuple[HypothesisEvaluation, ...]
    audits: tuple[str, ...] = ()

    @property
    def score(self) -> float:
        return sum(step.score.total for step in self.steps)

    def to_dict(self) -> dict:
        return {
            "initial_state": self.initial_state.to_dict(), "final_state": self.final_state.to_dict(),
            "steps": [step.to_dict() for step in self.steps],
            "score": self.score, "audits": list(self.audits),
        }


@dataclass(frozen=True)
class _BeamItem:
    state: WorldState
    steps: tuple[Hypothesis, ...]
    evaluations: tuple[HypothesisEvaluation, ...]

    @property
    def score(self) -> float:
        continuity = sum(0.5 for left, right in zip(self.steps, self.steps[1:]) if left.involved_entities[:1] == right.involved_entities[:1])
        return sum(step.score.total for step in self.steps) + continuity


def beam_search(
    initial_state: WorldState,
    candidates_by_step: tuple[tuple[Hypothesis, ...], ...],
    *,
    goal_terms: frozenset[str] = frozenset(),
    beam_width: int = 4,
    disabled_scores: frozenset[str] = frozenset(),
) -> ReasoningPlan:
    """Search a fixed candidate lattice with total ordering and no sampling."""
    beam = [_BeamItem(initial_state, (), ())]
    audits: list[str] = []
    for index, candidates in enumerate(candidates_by_step):
        expanded: list[_BeamItem] = []
        for item in beam:
            prior_actions = tuple(step.action for step in item.steps)
            for candidate in sorted(candidates, key=lambda h: h.name):
                evaluation = evaluate_hypothesis(item.state, candidate, goal_terms=goal_terms, prior_actions=prior_actions, disabled_scores=disabled_scores)
                if not evaluation.transition.accepted:
                    continue
                expanded.append(_BeamItem(evaluation.transition.state, item.steps + (evaluation.hypothesis,), item.evaluations + (evaluation,)))
        if not expanded:
            audits.append(f"blocked_no_consistent_continuation_at_{index}")
            break
        beam = sorted(expanded, key=lambda item: (-item.score, tuple(step.name for step in item.steps)))[:max(1, beam_width)]
    best = beam[0]
    audits.extend(
        f"stateless_fallback_at_{index}"
        for index, step in enumerate(best.steps)
        if "stateless_fallback" in step.required_preconditions
    )
    return ReasoningPlan(initial_state, best.state, best.steps, best.evaluations, tuple(audits))
