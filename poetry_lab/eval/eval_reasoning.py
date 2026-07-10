"""Reasoning-transfer measurement: do explicit line decisions reach output?"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import PROMPTS, generated_lines  # noqa: E402
from poemcore.engine import PoetryEngine  # noqa: E402
from poemcore.text import content_words  # noqa: E402


def _measure(engine: PoetryEngine, *, use_reasoning: bool) -> dict[str, float]:
    planned = realized = 0
    action_planned = action_realized = 0
    for prompt, verse in PROMPTS:
        result = engine.run(prompt, given_verse=verse, use_reasoning=use_reasoning)
        for intent, line in zip(result.line_intents, generated_lines(result)):
            actual = set(content_words(line))
            tokens = {intent.subject, intent.action, intent.object} - {"", result.poem_intent.speaker}
            planned += len(tokens)
            realized += sum(token in actual for token in tokens)
            if intent.action:
                action_planned += 1
                action_realized += int(intent.action in actual)
    return {
        "decision_realization": realized / planned if planned else 0.0,
        "action_realization": action_realized / action_planned if action_planned else 0.0,
    }


def main() -> None:
    engine = PoetryEngine()
    baseline = _measure(engine, use_reasoning=False)
    reasoning = _measure(engine, use_reasoning=True)
    print("=== explicit reasoning (A/B: generic line intent vs reasoning decisions) ===")
    print(f"{'metric':<24}{'OFF':>8}{'ON':>8}{'delta':>9}")
    for key, label in (
        ("decision_realization", "decision realization"),
        ("action_realization", "action realization"),
    ):
        off = baseline[key]
        on = reasoning[key]
        print(f"{label:<24}{off:>8.3f}{on:>8.3f}{on - off:>+9.3f}")


if __name__ == "__main__":
    main()
