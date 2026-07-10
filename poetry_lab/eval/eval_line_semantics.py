"""Line-semantic metrics — does the LinePlan layer give lines an intent?

A/B: discourse ranking alone (use_line_plan=False) vs. discourse ranking plus
the line-intent score (use_line_plan=True). Everything else identical. Reports:

  * subject/action presence — share of lines with both a subject and a verb
  * entity consistency      — share of content words linked to the poem's core
                              images (2-hop) — low = scattered entities
  * proper-name rate        — random named entities per line (lower is better)
  * plan satisfaction       — share of planned subject/object/setting concepts
                              actually realized in their line
  * semantic continuity     — adjacent-line content-word graph link share
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import PROMPTS, generated_lines  # noqa: E402
from poemcore.engine import PoetryEngine  # noqa: E402
from poemcore.line_plan import has_subject, has_verb  # noqa: E402
from poemcore.text import content_words, words  # noqa: E402


def _core_field(engine, result):
    seeds = set(result.request.seed_concepts)
    field_ = set(seeds)
    for s in seeds:
        for nb, _w in engine.graph.neighbors(s):
            field_.add(nb)
            for nb2, _w2 in engine.graph.neighbors(nb)[:8]:
                field_.add(nb2)
    return field_


def _measure(engine: PoetryEngine, *, use_line_plan: bool) -> dict:
    lines_with_sa = lines_total = 0
    linked = words_total = 0
    proper_hits = 0
    plan_hits = plan_total = 0
    cont_shares: list[float] = []

    for prompt, verse in PROMPTS:
        result = engine.run(prompt, given_verse=verse, use_line_plan=use_line_plan)
        core = _core_field(engine, result)
        gen = generated_lines(result)

        # subject/action + entity consistency + proper names
        for line in gen:
            toks = words(line)
            cw = content_words(line)
            lines_total += 1
            if has_subject(toks, cw) and has_verb(toks):
                lines_with_sa += 1
            for w in cw:
                words_total += 1
                if w in core:
                    linked += 1
                if w in engine.proper_names:
                    proper_hits += 1

        # plan satisfaction: did each line realize its planned concepts?
        intents = {li.index: li for li in result.line_intents}
        gen_by_index = generated_lines(result)
        for i, line in enumerate(gen_by_index):
            li = intents.get(i)
            if li is None:
                continue
            cw = set(content_words(line))
            planned = {li.subject, li.object, li.modifier} - {"", result.poem_intent.speaker}
            for c in planned:
                plan_total += 1
                if c in cw:
                    plan_hits += 1

        # semantic continuity (adjacent lines)
        wl = [set(content_words(l)) for l in gen]
        for a, b in zip(wl, wl[1:]):
            if not b:
                continue
            hit = sum(
                1 for w in b
                if w in a or any(nb in a for nb, _ in engine.graph.neighbors(w))
            )
            cont_shares.append(hit / len(b))

    return {
        "subject_action": lines_with_sa / lines_total if lines_total else 0.0,
        "entity_consistency": linked / words_total if words_total else 0.0,
        "proper_name_rate": proper_hits / lines_total if lines_total else 0.0,
        "plan_satisfaction": plan_hits / plan_total if plan_total else 0.0,
        "semantic_continuity": sum(cont_shares) / len(cont_shares) if cont_shares else 0.0,
    }


def main() -> None:
    engine = PoetryEngine()
    off = _measure(engine, use_line_plan=False)
    on = _measure(engine, use_line_plan=True)
    print("=== line semantics (A/B: line_plan off vs on) ===")
    rows = [
        ("subject/action presence", "subject_action", "higher better"),
        ("entity consistency", "entity_consistency", "higher better"),
        ("proper-name rate/line", "proper_name_rate", "lower better"),
        ("plan satisfaction", "plan_satisfaction", "higher better"),
        ("semantic continuity", "semantic_continuity", "higher better"),
    ]
    print(f"{'metric':<26}{'OFF':>8}{'ON':>8}{'delta':>9}   note")
    for label, key, note in rows:
        d = on[key] - off[key]
        print(f"{label:<26}{off[key]:>8.3f}{on[key]:>8.3f}{d:>+9.3f}   {note}")


if __name__ == "__main__":
    main()
