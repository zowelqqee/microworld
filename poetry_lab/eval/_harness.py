"""Shared evaluation harness — a fixed prompt battery reused by every metric."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poemcore.engine import PoetryEngine, Result  # noqa: E402

# The battery mixes the four required example prompts with a few style/theme
# variants so a metric sees breadth, not a single lucky sample.
PROMPTS: list[tuple[str, str]] = [
    ("write a poem about autumn", ""),
    ("write in the style of Pushkin", ""),
    ("write a poem about space using classical Russian imagery", ""),
    ("continue this verse", "Белеет парус одинокой\nВ тумане моря голубом!.."),
    ("напиши стихотворение о ночи", ""),
    ("write a poem about the sea", ""),
    ("write in the style of Akhmatova", ""),
    ("write in the style of Blok about night", ""),
    ("write a poem about winter", ""),
    ("write a poem about love", ""),
]


def run_battery(stanzas: int = 2) -> list[Result]:
    engine = PoetryEngine()
    results = []
    for prompt, verse in PROMPTS:
        results.append(engine.run(prompt, given_verse=verse, stanzas=stanzas))
    return results


def generated_lines(result: Result) -> list[str]:
    """Only the model-produced lines (drops any echoed prompt verse and the
    separator inserted in continuation mode)."""

    given = set(result.request.given_verse.splitlines())
    out = []
    for line in result.poem.lines:
        s = line.strip()
        if not s or s in given or set(s) <= {"—", " ", "-"}:
            continue
        out.append(line)
    return out
