"""A/B for intent-seeded generation: LineIntent ranking only vs. ranking +
seeded generation. Everything else (trigram, corpus, gates) identical.

Reports the full metric set requested for this experiment so the trade-offs are
visible in one place: the target (plan satisfaction) alongside every metric
seed-forcing might damage (grammaticality, rhyme, meter, novelty).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import PROMPTS, generated_lines  # noqa: E402
from poemcore.engine import PoetryEngine  # noqa: E402
from poemcore.line_plan import has_subject, has_verb  # noqa: E402
from poemcore.novelty import check_line  # noqa: E402
from poemcore.text import content_words, last_word, rhyme_key, syllables, words  # noqa: E402


def _core_field(engine, result):
    seeds = set(result.request.seed_concepts)
    field_ = set(seeds)
    for s in seeds:
        for nb, _w in engine.graph.neighbors(s):
            field_.add(nb)
            for nb2, _w2 in engine.graph.neighbors(nb)[:8]:
                field_.add(nb2)
    return field_


def _measure(engine: PoetryEngine, *, seeded: bool) -> dict:
    sa = sa_total = 0
    real3 = tot3 = 0
    on_theme = words_total = 0
    proper = 0
    plan_hits = plan_total = 0
    meter_ok = meter_total = 0
    novel = novel_total = 0
    rhyme_ok = rhyme_total = 0
    cont: list[float] = []

    for prompt, verse in PROMPTS:
        result = engine.run(prompt, given_verse=verse, use_seeded_generation=seeded)
        core = _core_field(engine, result)
        gen = generated_lines(result)
        target = result.plan.target_syllables
        intents = {li.index: li for li in result.line_intents}

        wl = []
        for i, line in enumerate(gen):
            toks = words(line)
            cw = content_words(line)
            wl.append(set(cw))

            sa_total += 1
            if has_subject(toks, cw) and has_verb(toks):
                sa += 1

            for a, b, c in zip(toks, toks[1:], toks[2:]):
                tot3 += 1
                if c in engine.phrase.forward2.get((a, b), {}):
                    real3 += 1

            for w in cw:
                words_total += 1
                if w in core:
                    on_theme += 1
                if w in engine.proper_names:
                    proper += 1

            meter_total += 1
            if abs(syllables(line) - target) <= 1:
                meter_ok += 1

            novel_total += 1
            if check_line(engine.phrase, line):
                novel += 1

            li = intents.get(i)
            if li is not None:
                planned = {li.subject, li.object, li.modifier} - {"", result.poem_intent.speaker}
                for pc in planned:
                    plan_total += 1
                    if pc in set(cw):
                        plan_hits += 1

        for a, b in zip(wl, wl[1:]):
            if not b:
                continue
            hit = sum(1 for w in b if w in a or any(nb in a for nb, _ in engine.graph.neighbors(w)))
            cont.append(hit / len(b))

        # rhyme: per-stanza label pairs share a rhyme key
        scheme = result.plan.rhyme_scheme
        per = len(scheme)
        given = set(result.request.given_verse.splitlines())
        content = [l for l in result.poem.lines if l.strip() and set(l.strip()) - {"—", "-", " "} and l not in given]
        for s in range(0, len(content) - per + 1, per):
            stanza = content[s:s + per]
            by_label = defaultdict(list)
            for i, line in enumerate(stanza):
                by_label[scheme[i % len(scheme)]].append(line)
            for group in by_label.values():
                if len(group) < 2:
                    continue
                rhyme_total += 1
                if len({rhyme_key(last_word(g)) for g in group}) == 1:
                    rhyme_ok += 1

    return {
        "plan_satisfaction": plan_hits / plan_total if plan_total else 0.0,
        "subject_action": sa / sa_total if sa_total else 0.0,
        "inter_line_continuity": sum(cont) / len(cont) if cont else 0.0,
        "thematic_coherence": on_theme / words_total if words_total else 0.0,
        "local_grammaticality": real3 / tot3 if tot3 else 0.0,
        "meter_within1": meter_ok / meter_total if meter_total else 0.0,
        "rhyme": rhyme_ok / rhyme_total if rhyme_total else 0.0,
        "novelty": novel / novel_total if novel_total else 0.0,
        "proper_name_rate": proper / sa_total if sa_total else 0.0,
    }


def main() -> None:
    engine = PoetryEngine()
    off = _measure(engine, seeded=False)
    on = _measure(engine, seeded=True)
    print("=== intent-seeded generation (A/B: ranking only vs ranking + seeding) ===")
    rows = [
        ("plan satisfaction", "plan_satisfaction", "TARGET ↑"),
        ("subject/action presence", "subject_action", "↑"),
        ("inter-line continuity", "inter_line_continuity", "↑"),
        ("thematic coherence", "thematic_coherence", "↑"),
        ("local grammaticality", "local_grammaticality", "↑ (risk)"),
        ("meter within ±1", "meter_within1", "↑"),
        ("rhyme", "rhyme", "↑ (risk)"),
        ("novelty", "novelty", "↑"),
        ("proper-name rate/line", "proper_name_rate", "↓"),
    ]
    print(f"{'metric':<26}{'OFF':>8}{'ON':>8}{'delta':>9}   note")
    for label, key, note in rows:
        d = on[key] - off[key]
        print(f"{label:<26}{off[key]:>8.3f}{on[key]:>8.3f}{d:>+9.3f}   {note}")


if __name__ == "__main__":
    main()
