"""Narrative-transfer evaluation over the required prose prompt battery."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poemcore.narrative import NarrativeEngine
from poemcore.text import content_words, words

PROMPTS = [
    ("Continue this dialogue.", "Иван посмотрел на Берлиоза. – Я слушаю вас."),
    ("Describe a room.", ""),
    ("Describe an evening in Moscow.", ""),
    ("Continue a scene.", "Берлиоз остановился у скамейки и посмотрел на пруд."),
    ("Introduce a new character.", ""),
    ("Write one paragraph after this paragraph.", "Вечер опустился на Москву, и улица стала пустой."),
]


def _trigram_coherence(engine: NarrativeEngine, text: str) -> tuple[int, int]:
    tokens = words(text)
    total = max(0, len(tokens) - 2)
    observed = sum(tuple(tokens[i : i + 2]) in engine.phrase.forward2 and tokens[i + 2] in engine.phrase.forward2[tuple(tokens[i : i + 2])]
                   for i in range(total))
    return observed, total


def main() -> None:
    engine = NarrativeEngine()
    sums = {"coherent": 0, "trigrams": 0, "entity_consistent": 0, "entity_total": 0,
            "pronoun_consistent": 0, "sentences": 0, "topic_on_field": 0, "novel": 0}
    elapsed = []
    examples = []
    third_person = {"он", "она", "они"}
    for index, (prompt, context) in enumerate(PROMPTS):
        started = time.perf_counter()
        result = engine.run(prompt, context=context, sentences=4, seed=f"eval|{index}|{prompt}|{context}")
        elapsed.append(time.perf_counter() - started)
        field_ = engine.graph.activate(list(result.plan.goal.seeds))
        for sentence in result.paragraph.sentences:
            token_set = set(words(sentence))
            cwords = content_words(sentence)
            observed, total = _trigram_coherence(engine, sentence)
            sums["coherent"] += observed
            sums["trigrams"] += total
            names = token_set & engine.proper_names
            sums["entity_consistent"] += int(names <= set(result.plan.goal.characters) | {result.plan.goal.location})
            sums["entity_total"] += 1
            sums["pronoun_consistent"] += int(len(token_set & third_person) <= 1)
            sums["sentences"] += 1
            sums["topic_on_field"] += int(any(field_.get(word, 0.0) > 0.0 for word in cwords))
            sums["novel"] += int(not engine.phrase.contains_seen_4gram(words(sentence)))
        examples.append({"prompt": prompt, "paragraph": result.paragraph.text()})
    n = sums["sentences"] or 1
    report = {
        "prompts": len(PROMPTS),
        "sentence_coherence": round(sums["coherent"] / max(1, sums["trigrams"]), 3),
        "entity_consistency": round(sums["entity_consistent"] / max(1, sums["entity_total"]), 3),
        "pronoun_consistency": round(sums["pronoun_consistent"] / n, 3),
        "topic_drift": round(1 - sums["topic_on_field"] / n, 3),
        "novelty": round(sums["novel"] / n, 3),
        "generation_speed": {"mean_seconds_per_paragraph": round(sum(elapsed) / len(elapsed), 3)},
        "artifact_size_bytes": os.path.getsize("artifacts/narrative_model.json"),
        "examples": examples,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
