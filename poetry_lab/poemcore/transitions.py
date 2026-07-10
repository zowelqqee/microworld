"""Adapters from production rule data to the generalized hypothesis boundary."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

_RUNTIME = Path(__file__).resolve().parents[2]
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from worldpgt.cognition.inference_engine import run_inference  # noqa: E402

from poemcore.narrative_reasoning import Hypothesis, HypothesisEvaluation, event_hypothesis, evaluate_hypothesis
from poemcore.world_state import StateFact, WorldState


def qa_depth_one(overlay_items: list[dict]) -> tuple[HypothesisEvaluation, ...]:
    """Run existing QA rules as a timeless depth-1 configuration.

    Rule execution stays in production's data-driven interpreter.  This thin
    adapter only normalizes its derived facts into the experiment's shared
    hypothesis/test/accept artifact; it never mutates an overlay or QA code.
    """
    labels = sorted({str(item.get(key) or "") for item in overlay_items for key in ("subject", "object") if item.get(key)})
    initial = WorldState.from_initial_facts(tuple(StateFact(label, "introduced", "qa", 0) for label in labels))
    evaluations: list[HypothesisEvaluation] = []
    for inferred in run_inference(overlay_items).facts:
        hypothesis = event_hypothesis(
            name=f"qa:{inferred.rule}:{inferred.subject}:{inferred.predicate}:{inferred.object}",
            subject=inferred.subject, action=inferred.predicate, object_=inferred.object,
        )
        hypothesis = replace(hypothesis, proof_chain=(inferred.rule,), expected_consequences=("answer_support",))
        evaluations.append(evaluate_hypothesis(initial, hypothesis, goal_terms=frozenset({inferred.subject, inferred.object})))
    return tuple(evaluations)
