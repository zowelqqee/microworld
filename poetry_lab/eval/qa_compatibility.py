"""Phase-6 compatibility report; it deliberately does not replace QA."""

from __future__ import annotations

import json
import sys
from pathlib import Path

RUNTIME = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUNTIME))

from worldpgt.cognition.inference_engine import run_inference  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poemcore.transitions import qa_depth_one  # noqa: E402


def main() -> None:
    fixture = [
        {"overlay_type": "overlay_relation", "subject": "A", "predicate": "owned_by", "object": "B"},
        {"overlay_type": "overlay_relation", "subject": "B", "predicate": "owned_by", "object": "C"},
    ]
    production = run_inference(fixture)
    production_facts = [fact.as_dict() for fact in production.facts]
    evaluations = qa_depth_one(fixture)
    generalized_facts = [
        evaluation.hypothesis.delta.assertions[0].triple
        for evaluation in evaluations if evaluation.hypothesis.status.value == "accepted"
    ]
    production_triples = [(fact.subject, fact.predicate, fact.object) for fact in production.facts]
    report = {
        "fixture": "ownership_transitivity_depth_1",
        "production_inferred_facts": production_facts,
        "production_proof_chains": [fact["chain"] for fact in production_facts],
        "narrative_engine_rule_adapter": "production_rule_interpreter_depth_1",
        "accepted_rejected_hypotheses": [evaluation.hypothesis.to_dict() for evaluation in evaluations],
        "answer_supporting_fact_match": generalized_facts == production_triples,
        "compatibility_proof_succeeded": generalized_facts == production_triples and all(e.transition.accepted for e in evaluations),
        "reason": "The adapter executes production JSON rules unchanged, then uses the generalized hypothesis lifecycle. Production QA remains untouched.",
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
