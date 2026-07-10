"""Novelty metric — is the system recombining or memorising?

The central risk the brief calls out: the system must NOT reproduce fixed
responses. This measures the fraction of generated lines that contain no corpus
4-gram (higher = more original recombination). It is the quantitative form of
the inverted support gate in ``poemcore/novelty.py``.
"""

from __future__ import annotations

from _harness import generated_lines, run_battery
from poemcore.novelty import check_line
from poemcore.text import words


def main() -> None:
    engine_results = run_battery()
    phrase = None
    total, novel = 0, 0
    worst: list[tuple[float, str]] = []
    for result in engine_results:
        from poemcore.engine import PoetryEngine  # local import keeps harness lean
        phrase = phrase or PoetryEngine().phrase
        lines = generated_lines(result)
        for line in lines:
            total += 1
            if check_line(phrase, line):
                novel += 1
            else:
                worst.append((0.0, line))
    ratio = novel / total if total else 0.0
    print("=== novelty ===")
    print(f"generated lines : {total}")
    print(f"novel lines     : {novel}")
    print(f"novelty ratio   : {ratio:.3f}  (1.0 = no memorised 4-grams)")
    # Bigram REUSE is expected to be ~1.0 by construction: the generator only
    # ever walks word-transitions the corpus taught it, so every adjacent pair
    # is corpus-derived. Novelty therefore lives at the 3–4-gram level — new
    # *combinations* of learned transitions — which is exactly what the line
    # metric above measures. Reporting the bigram figure makes that explicit
    # rather than hiding it.
    corpus_bigrams = _corpus_bigrams(phrase)
    reuse = 1.0 - _bigram_novelty(engine_results, corpus_bigrams)
    print(f"bigram reuse    : {reuse:.3f}  (expected ~1.0 — every step is a learned transition)")
    if worst:
        print("\nechoed lines (memorised 4-gram present):")
        for _s, line in worst[:8]:
            print(f"  · {line}")


def _corpus_bigrams(phrase) -> set:
    bigrams = set()
    for a, counter in phrase.forward.items():
        for b in counter:
            bigrams.add((a, b))
    return bigrams


def _bigram_novelty(results, corpus_bigrams) -> float:
    seen, novel = 0, 0
    for result in results:
        for line in generated_lines(result):
            toks = words(line)
            for a, b in zip(toks, toks[1:]):
                seen += 1
                if (a, b) not in corpus_bigrams:
                    novel += 1
    return novel / seen if seen else 0.0


if __name__ == "__main__":
    main()
