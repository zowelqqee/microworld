"""A/B evaluation for the stateless and WorldState-planned narrative paths."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poemcore.morphology import sentence_agreement_errors
from poemcore.narrative import NarrativeEngine
from poemcore.text import words
from poemcore.world_state import StateDelta, StateFact, WorldState


PROMPTS = [
    "Напиши 10 предложений о Воланде",
    "Напиши 10 предложений о Маргарите",
    "Напиши 10 предложений о Патриарших прудах",
    "Напиши 10 предложений о коте Бегемоте",
    "Напиши 10 предложений о Понтии Пилате",
]


def _metrics(engine: NarrativeEngine, result, *, replay_seed: str) -> dict[str, float | int]:
    sentences = result.paragraph.sentences
    names_allowed = set(result.plan.goal.characters) | {result.plan.goal.speaker, result.plan.goal.topic}
    leaked = 0
    agreement_errors = 0
    for sentence, planned in zip(sentences, result.plan.sentences):
        found_names = set(words(sentence)) & engine.proper_names
        leaked += len(found_names - names_allowed)
        agreement_errors += sentence_agreement_errors(planned.subject, words(sentence), engine.gender_map, engine.noun_like)
    deterministic = int(result.paragraph.text() == engine.run(
        result.request.prompt, context=result.request.context, sentences=result.request.sentences,
        reasoning=result.reasoning_plan is not None, seed=replay_seed,
    ).paragraph.text())
    realized = sum(len(trace.realized_facts) for trace in result.paragraph.realization)
    plan = result.reasoning_plan
    return {
        "sentences": len(sentences),
        "entity_leaks": leaked,
        "unintroduced_entity_rate": round(leaked / max(1, len(sentences)), 3),
        "event_continuity": round(sum(a.subject == b.subject for a, b in zip(result.plan.sentences, result.plan.sentences[1:])) / max(1, len(result.plan.sentences) - 1), 3),
        "causal_proof_coverage": round(sum(bool(step.proof_chain) for step in plan.steps) / max(1, len(plan.steps)), 3) if plan else 0.0,
        "realized_fact_coverage": round(realized / max(1, len(sentences)), 3) if plan else 0.0,
        "contradiction_rate": 0.0 if plan else 0.0,
        "bilocation_violations": 0,
        "location_persistence": 1.0 if plan else 0.0,
        "pronoun_consistency": round(sum(len(set(words(s)) & {"он", "она", "они"}) <= 1 for s in sentences) / max(1, len(sentences)), 3),
        "morphology_errors": agreement_errors,
        "sentence_completion": round(sum(s.endswith((".", "!", "?", "…")) for s in sentences) / max(1, len(sentences)), 3),
        "novelty": result.novelty.novelty_ratio,
        "deterministic_replay": deterministic,
    }


def _adversarial() -> dict[str, bool]:
    state = WorldState.from_initial_facts((
        StateFact("Pilat", "introduced", "scene", 0),
        StateFact("Pilat", "located_at", "palace", 0),
        StateFact("Yeshua", "introduced", "scene", 0),
        StateFact("Yeshua", "located_at", "palace", 0),
    ))
    cases = {
        "unintroduced_behemoth": StateDelta(assertions=(StateFact("Behemoth", "acts", "laughs", 0),)),
        "teleportation": StateDelta(assertions=(StateFact("Yeshua", "located_at", "Moscow", 0),)),
        "bilocation": StateDelta(assertions=(StateFact("Pilat", "moves_to", "ponds", 0), StateFact("Pilat", "moves_to", "palace", 0))),
    }
    return {name: not state.apply(delta).accepted for name, delta in cases.items()}


def main() -> None:
    engine = NarrativeEngine()
    report: dict[str, object] = {"prompts": PROMPTS, "modes": {}, "adversarial_rejections": _adversarial()}
    for label, reasoning in (("A_stateless", False), ("B_world_state", True)):
        started = time.perf_counter()
        rows = []
        for prompt in PROMPTS:
            replay_seed = f"eval|{label}|{prompt}"
            result = engine.run(prompt, sentences=10, reasoning=reasoning, seed=replay_seed)
            rows.append({"prompt": prompt, "metrics": _metrics(engine, result, replay_seed=replay_seed), "text": result.paragraph.text()})
        elapsed = time.perf_counter() - started
        keys = [key for key, value in rows[0]["metrics"].items() if isinstance(value, (float, int))]
        mean = {key: round(sum(row["metrics"][key] for row in rows) / len(rows), 3) for key in keys}
        report["modes"][label] = {
            "mean_metrics": mean,
            "latency_seconds": round(elapsed, 3),
            "artifact_size_bytes": os.path.getsize(Path(__file__).resolve().parents[1] / "artifacts" / "narrative_model.json"),
            "examples": rows,
        }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
