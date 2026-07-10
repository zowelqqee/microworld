"""Named, auditable hypothesis scoring for narrative state transitions."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from poemcore.world_state import StateDelta, StateFact, TransitionResult, WorldState


class HypothesisStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


@dataclass(frozen=True)
class HypothesisScore:
    causal_consistency: float = 0.0
    temporal_consistency: float = 0.0
    entity_persistence: float = 0.0
    location_consistency: float = 0.0
    state_transition_value: float = 0.0
    contradiction_risk: float = 0.0
    goal_relevance: float = 0.0
    repetition_risk: float = 0.0

    @property
    def total(self) -> float:
        return sum(self.to_dict().values())

    def to_dict(self) -> dict[str, float]:
        return {
            "causal_consistency": self.causal_consistency,
            "temporal_consistency": self.temporal_consistency,
            "entity_persistence": self.entity_persistence,
            "location_consistency": self.location_consistency,
            "state_transition_value": self.state_transition_value,
            "contradiction_risk": self.contradiction_risk,
            "goal_relevance": self.goal_relevance,
            "repetition_risk": self.repetition_risk,
        }

    def ablated(self, disabled: frozenset[str]) -> "HypothesisScore":
        return HypothesisScore(**{key: 0.0 if key in disabled else value for key, value in self.to_dict().items()})


@dataclass(frozen=True)
class Hypothesis:
    name: str
    delta: StateDelta
    action: str
    involved_entities: tuple[str, ...]
    location: str = ""
    temporal_position: int | None = None
    required_preconditions: tuple[str, ...] = ()
    expected_consequences: tuple[str, ...] = ()
    proof_chain: tuple[str, ...] = ()
    score: HypothesisScore = HypothesisScore()
    status: HypothesisStatus = HypothesisStatus.PENDING
    rejection_reason: str = ""
    payload: object | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name, "delta": self.delta.to_dict(), "action": self.action,
            "involved_entities": list(self.involved_entities), "location": self.location,
            "temporal_position": self.temporal_position,
            "required_preconditions": list(self.required_preconditions),
            "expected_consequences": list(self.expected_consequences),
            "proof_chain": list(self.proof_chain), "score": self.score.to_dict(),
            "score_total": self.score.total, "status": self.status.value,
            "rejection_reason": self.rejection_reason,
        }


@dataclass(frozen=True)
class HypothesisEvaluation:
    hypothesis: Hypothesis
    transition: TransitionResult

    def to_dict(self) -> dict:
        return {"hypothesis": self.hypothesis.to_dict(), "transition": self.transition.to_dict()}


def evaluate_hypothesis(
    state: WorldState,
    hypothesis: Hypothesis,
    *,
    goal_terms: frozenset[str] = frozenset(),
    prior_actions: tuple[str, ...] = (),
    disabled_scores: frozenset[str] = frozenset(),
) -> HypothesisEvaluation:
    """Test one candidate without mutating ``state`` and retain every score part."""
    transition = state.apply(hypothesis.delta)
    asserted = transition.asserted
    predicate_names = {fact.predicate for fact in asserted}
    causal = 2.0 if transition.proof_steps else (0.5 if asserted else -1.0)
    temporal = -4.0 * sum(v.code == "unintroduced_entity" for v in transition.violations)
    entity = 1.0 if all(state.is_introduced(entity) or any(f.subject == entity and f.predicate == "introduced" for f in asserted) for entity in hypothesis.involved_entities[:1]) else -3.0
    location = -4.0 * sum(v.code in {"bilocation", "location_change_without_move", "participants_not_colocated"} for v in transition.violations)
    transition_value = 2.0 if asserted or transition.inferred else -2.0
    contradiction = -6.0 * len(transition.violations)
    relevance = 2.0 if goal_terms & {fact.subject for fact in asserted} | goal_terms & {fact.object for fact in asserted} else 0.0
    repetition = -2.0 if hypothesis.action and hypothesis.action in prior_actions else 0.0
    score = HypothesisScore(causal, temporal, entity, location, transition_value, contradiction, relevance, repetition).ablated(disabled_scores)
    if transition.accepted:
        assessed = replace(hypothesis, score=score, status=HypothesisStatus.ACCEPTED, rejection_reason="")
    else:
        reasons = ", ".join(sorted({item.code for item in transition.violations}))
        assessed = replace(hypothesis, score=score, status=HypothesisStatus.REJECTED, rejection_reason=reasons)
    return HypothesisEvaluation(assessed, transition)


def event_hypothesis(
    *,
    name: str,
    subject: str,
    action: str,
    object_: str,
    payload: object | None = None,
) -> Hypothesis:
    """Build one ordinary narrative event candidate from a planned sentence."""
    fact = StateFact(subject, action or "acts", object_ or "scene", t=0)
    return Hypothesis(
        name=name,
        delta=StateDelta(assertions=(fact,), label=name),
        action=action or "acts",
        involved_entities=tuple(item for item in (subject, object_) if item),
        payload=payload,
    )
