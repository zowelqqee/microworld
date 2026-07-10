"""Inter-line continuity — the metric the discourse-state transfer targets.

Local grammaticality (eval_coherence) measures *within-line* spans and does not
move when we change line *selection*. Sentence-level coherence is a cross-line
property: does each line develop images already in play? This measures it as the
average concept-graph connectedness between consecutive lines — for each
adjacent content-line pair, the share of the later line's content words that are
graph-adjacent (or equal) to some content word in the earlier line.

It runs the battery twice — with discourse-salience ranking on and off — so the
effect of the ported mechanism is isolated. Meter/rhyme/novelty are gated before
ranking, so only this number should move.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _harness import PROMPTS, generated_lines  # noqa: E402
from poemcore.engine import PoetryEngine  # noqa: E402
from poemcore.text import content_words  # noqa: E402


def _line_link_share(prev_words: list[str], cur_words: list[str], neighbours) -> float:
    if not cur_words:
        return 0.0
    prev_set = set(prev_words)
    linked = 0
    for w in cur_words:
        if w in prev_set:
            linked += 1
            continue
        if any(nb in prev_set for nb, _wt in neighbours(w)):
            linked += 1
    return linked / len(cur_words)


def _continuity(engine: PoetryEngine, *, rank: bool) -> float:
    graph = engine.graph
    shares: list[float] = []
    for prompt, verse in PROMPTS:
        result = engine.run(prompt, given_verse=verse, rank=rank)
        lines = [content_words(ln) for ln in generated_lines(result)]
        for prev, cur in zip(lines, lines[1:]):
            shares.append(_line_link_share(prev, cur, graph.neighbors))
    return sum(shares) / len(shares) if shares else 0.0


def main() -> None:
    engine = PoetryEngine()
    off = _continuity(engine, rank=False)
    on = _continuity(engine, rank=True)
    print("=== inter-line continuity (graph link share, adjacent lines) ===")
    print(f"ranking OFF (accept first valid) : {off:.3f}")
    print(f"ranking ON  (discourse salience) : {on:.3f}")
    delta = on - off
    print(f"delta                            : {delta:+.3f} "
          f"({'salience selection improves cross-line coherence' if delta > 0 else 'no gain'})")


if __name__ == "__main__":
    main()
