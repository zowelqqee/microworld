"""Narrative quality baseline for the three-layer description/scene generator.

Every metric here is *corpus-grounded and interpretable* — the point that makes
this an honest alternative to an LLM: each number is a share of the output that
is provably backed by the ingested data, not a vibe.

Per prompt and in aggregate:

  - grammaticality   share of generated 3-word windows that actually occur in
                     the corpus (order-2 phrase model). A word-salad detector:
                     1.0 = every trigram is real corpus language, 0.0 = invented.
                     (same measure as eval_coherence's "local grammaticality" /
                     eval_narrative's trigram coherence.)
  - thematic         share of output content words that are within two concept-
                     graph hops of the prompt's seed concepts. Did the reasoning
                     stage keep word choice on-topic?
  - continuity       avg concept-graph link share between *consecutive*
                     sentences — the metric the coherence work (#1) targets:
                     does each sentence develop images already in play, or is
                     the paragraph a bag of disconnected facets? (same measure
                     as eval_continuity, applied to the narrative engine.)
  - novelty          the inverted-gate score: 1.0 = recites no corpus 4-gram.
  - grounded         share of prompts that produced a non-empty passage (an
                     off-corpus topic honestly produces nothing rather than
                     hallucinating — that is a feature, and this quantifies it).

Corpus-agnostic: point it at any artifact with --artifact. Deterministic per
seed. Run before/after an architecture change to see which number moved.

    python3 eval/bench_narrative_quality.py
    python3 eval/bench_narrative_quality.py --artifact artifacts/narrative_model.json --json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poemcore.narrative import NarrativeEngine  # noqa: E402
from poemcore.text import content_words, words  # noqa: E402

# On-corpus topics (should generate richly) mixed with a couple of anachronisms
# (should honestly produce little/nothing), and both describe- and scene-framed
# prompts so the schema pool and the scene bias are both exercised.
BATTERY: tuple[tuple[str, int], ...] = (
    ("Describe love", 1),
    ("Describe the sea", 1),
    ("Describe the night", 1),
    ("Describe death", 1),
    ("Describe a king", 1),
    ("Describe fear", 1),
    ("Write a scene about a murder", 0),
    ("Write a scene about a storm", 0),
    ("Write a scene about a battle", 0),
    ("Write a story about the moon", 0),
    ("Write a story about a monster", 0),
    ("Write a scene about a ship", 0),
    ("Describe a rocket", 1),        # anachronism — expect empty/thin, honestly
    ("Write a scene about a robot", 0),  # anachronism
)

SENTENCES = 6
SEEDS = ("q1", "q2", "q3")  # average over a few re-rolls (creative is non-deterministic)


def _grammaticality(engine: NarrativeEngine, text: str) -> tuple[int, int]:
    """Observed corpus trigrams / total generated trigrams."""
    toks = words(text)
    total = max(0, len(toks) - 2)
    observed = 0
    for i in range(total):
        ctx = tuple(toks[i : i + 2])
        nxts = engine.phrase.forward2.get(ctx)
        if nxts and toks[i + 2] in nxts:
            observed += 1
    return observed, total


def _two_hop(graph, seeds: tuple[str, ...]) -> set[str]:
    field = set(seeds)
    for s in seeds:
        for nb, _w in graph.neighbors(s):
            field.add(nb)
            for nb2, _w2 in graph.neighbors(nb):
                field.add(nb2)
    return field


def _thematic(engine: NarrativeEngine, text: str, seeds: tuple[str, ...]) -> tuple[int, int]:
    field = _two_hop(engine.graph, seeds)
    cw = content_words(text)
    on = sum(1 for w in cw if w in field)
    return on, len(cw)


def _continuity(engine: NarrativeEngine, sentences: list[str], topic: str = "") -> list[float]:
    """Inter-sentence link share. ``topic`` (the repeated subject word) is
    excluded from both sides: otherwise a paragraph that restates the same
    subject every sentence scores a trivial 1.0 without any real discourse
    flow. Excluding it measures whether the *rest* of each sentence develops
    images already in play — the property real narrative coherence needs."""
    shares: list[float] = []
    lines = [[w for w in content_words(s) if w != topic] for s in sentences]
    for prev, cur in zip(lines, lines[1:]):
        if not cur:
            continue
        prev_set = set(prev)
        linked = 0
        for w in cur:
            if w in prev_set or any(nb in prev_set for nb, _wt in engine.graph.neighbors(w)):
                linked += 1
        shares.append(linked / len(cur))
    return shares


def _pct(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default=str(Path(__file__).resolve().parents[1] / "artifacts" / "narrative_model.json"))
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    engine = NarrativeEngine(Path(args.artifact))
    load_s = time.perf_counter() - t0

    rows = []
    for prompt, _bias in BATTERY:
        gram_obs = gram_tot = them_on = them_tot = 0
        cont: list[float] = []
        novelties: list[float] = []
        sent_counts: list[int] = []
        word_counts: list[int] = []
        latencies: list[float] = []
        nonempty = 0
        for seed in SEEDS:
            t = time.perf_counter()
            result = engine.run(prompt, sentences=SENTENCES, seed=seed)
            latencies.append((time.perf_counter() - t) * 1000)
            paragraph = result.paragraph
            text = paragraph.text().strip()
            if not text:
                continue
            nonempty += 1
            seeds = tuple(result.plan.goal.seeds)
            go, gt = _grammaticality(engine, text)
            to, tt = _thematic(engine, text, seeds)
            gram_obs += go; gram_tot += gt
            them_on += to; them_tot += tt
            cont.extend(_continuity(engine, paragraph.sentences, topic=result.plan.goal.topic))
            novelties.append(result.novelty.novelty_ratio)
            sent_counts.append(len(paragraph.sentences))
            word_counts.append(len(words(text)))
        rows.append({
            "prompt": prompt,
            "grounded": round(nonempty / len(SEEDS), 2),
            "grammaticality": round(_pct(gram_obs, gram_tot), 3),
            "thematic": round(_pct(them_on, them_tot), 3),
            "continuity": round(statistics.mean(cont), 3) if cont else 0.0,
            "novelty": round(statistics.mean(novelties), 3) if novelties else 0.0,
            "sentences": round(statistics.mean(sent_counts), 1) if sent_counts else 0.0,
            "words": round(statistics.mean(word_counts), 1) if word_counts else 0.0,
            "ms": round(statistics.mean(latencies), 1),
            "sample": engine.run(prompt, sentences=SENTENCES, seed="sample").paragraph.text().strip(),
        })

    def agg(key: str) -> float:
        vals = [r[key] for r in rows if r["grounded"] > 0]
        return round(statistics.mean(vals), 3) if vals else 0.0

    summary = {
        "artifact": Path(args.artifact).name,
        "load_seconds": round(load_s, 2),
        "prompts": len(BATTERY),
        "aggregate": {
            "grounded": round(statistics.mean(r["grounded"] for r in rows), 2),
            "grammaticality": agg("grammaticality"),
            "thematic": agg("thematic"),
            "continuity": agg("continuity"),
            "novelty": agg("novelty"),
        },
    }

    if args.json:
        print(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2))
        return

    print(f"\nartifact={summary['artifact']}  load={summary['load_seconds']}s  "
          f"prompts={summary['prompts']}  seeds/prompt={len(SEEDS)}\n")
    print(f"{'prompt':<32} {'grnd':>5} {'gram':>5} {'them':>5} {'cont':>5} {'novl':>5} {'sent':>5} {'ms':>6}")
    print("-" * 80)
    for r in rows:
        print(f"{r['prompt']:<32} {r['grounded']:>5} {r['grammaticality']:>5} "
              f"{r['thematic']:>5} {r['continuity']:>5} {r['novelty']:>5} {r['sentences']:>5} {r['ms']:>6}")
    a = summary["aggregate"]
    print("-" * 80)
    print(f"{'AGGREGATE (grounded prompts)':<32} {a['grounded']:>5} {a['grammaticality']:>5} "
          f"{a['thematic']:>5} {a['continuity']:>5} {a['novelty']:>5}")
    print("\nlegend: grnd=grounded(non-empty share)  gram=grammaticality(corpus trigram share)  "
          "them=thematic(on-graph share)\n        cont=continuity(inter-sentence link share)  "
          "novl=novelty(no corpus 4-gram)\n")
    print("samples:")
    for r in rows:
        print(f"  [{r['prompt']}]  {r['sample'] or '(empty — off-corpus)'}")
    print()


if __name__ == "__main__":
    main()
