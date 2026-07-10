"""Coherence — measured two different ways, because they diverge.

1. THEMATIC coherence (graph on-theme share): the share of a poem's content
   words that are connected in the concept graph to the prompt's seed concepts
   (within two hops). Measures whether the *reasoning* stage (spreading
   activation) steered word choice toward the right image cluster.

2. LOCAL grammaticality (real-trigram share): the share of generated 3-word
   windows that actually occur in the corpus. Measures whether the *language*
   stage produced grammatical spans rather than word salad. A pure bigram walk
   scores low here (it only ever guarantees 2-word spans); the order-2 phrase
   model — the port of QA's multi-word-fragment context — is what raises it.

These pull in slightly different directions: order-2 context improves local
grammaticality but can spend a word on a grammatical connective instead of an
on-theme image, nudging thematic share down. Reporting both keeps that
trade-off visible instead of hiding it behind one number.
"""

from __future__ import annotations

from _harness import generated_lines, run_battery
from poemcore.text import content_words, words


def main() -> None:
    from poemcore.engine import PoetryEngine

    engine = PoetryEngine()
    graph = engine.graph

    overall_hits, overall_words = 0, 0
    per_prompt: list[tuple[str, float]] = []
    for result in run_battery_with(engine):
        seeds = set(result.request.seed_concepts)
        # two-hop neighbourhood of the seeds
        reachable = set(seeds)
        for s in seeds:
            for nb, _w in graph.neighbors(s):
                reachable.add(nb)
                for nb2, _w2 in graph.neighbors(nb)[:12]:
                    reachable.add(nb2)
        hits, words_ct = 0, 0
        for line in generated_lines(result):
            for w in content_words(line):
                words_ct += 1
                if w in reachable:
                    hits += 1
        overall_hits += hits
        overall_words += words_ct
        share = hits / words_ct if words_ct else 0.0
        per_prompt.append((result.request.theme or result.request.style_author, share))

    print("=== coherence (graph on-theme share) ===")
    for theme, share in per_prompt:
        print(f"  {share:5.2f}  {theme}")
    overall = overall_hits / overall_words if overall_words else 0.0
    print(f"overall on-theme content-word share : {overall:.3f}")

    # local grammaticality: share of generated trigrams attested in the corpus
    real3, total3 = 0, 0
    for result in run_battery_with(engine):
        for line in generated_lines(result):
            toks = words(line)
            for a, b, c in zip(toks, toks[1:], toks[2:]):
                total3 += 1
                if c in engine.phrase.forward2.get((a, b), {}):
                    real3 += 1
    gram = real3 / total3 if total3 else 0.0
    print(f"local grammaticality (real-trigram share) : {gram:.3f}  "
          f"({real3}/{total3} generated 3-word windows seen in corpus)")


def run_battery_with(engine):
    from _harness import PROMPTS
    return [engine.run(p, given_verse=v) for p, v in PROMPTS]


if __name__ == "__main__":
    main()
