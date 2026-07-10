"""Style metric — does "in the style of X" actually move toward X?

For each author, generate a style-conditioned poem and measure how much of its
vocabulary overlaps that author's signature vocabulary versus the other
authors'. A positive gap (own > others) means the style profile learned at
ingest time is steering generation, which is what "keep the language layer
intact but swap the source" is supposed to preserve.
"""

from __future__ import annotations

from poemcore.engine import PoetryEngine
from poemcore.text import content_words


def main() -> None:
    engine = PoetryEngine()
    authors = engine.meta["authors"]
    signatures = {
        a: set(engine.style_profiles[a].get("signature_vocab", [])) for a in authors
    }

    print("=== style separation ===")
    own_scores, other_scores = [], []
    for author in authors:
        result = engine.run(f"write in the style of {author}", stanzas=2)
        vocab = set()
        for line in result.poem.lines:
            vocab.update(content_words(line))
        own = _overlap(vocab, signatures[author])
        others = [
            _overlap(vocab, signatures[b]) for b in authors if b != author
        ]
        avg_other = sum(others) / len(others) if others else 0.0
        own_scores.append(own)
        other_scores.append(avg_other)
        gap = own - avg_other
        mark = "✓" if gap > 0 else "·"
        print(f"  {mark} {author:<10} own={own:.2f}  others_avg={avg_other:.2f}  gap={gap:+.2f}")

    mean_own = sum(own_scores) / len(own_scores)
    mean_other = sum(other_scores) / len(other_scores)
    print(f"mean own-signature overlap   : {mean_own:.3f}")
    print(f"mean other-signature overlap : {mean_other:.3f}")
    print(f"separation                   : {mean_own - mean_other:+.3f} "
          f"({'style steers generation' if mean_own > mean_other else 'no separation'})")


def _overlap(vocab: set, signature: set) -> float:
    if not signature:
        return 0.0
    return len(vocab & signature) / len(signature)


if __name__ == "__main__":
    main()
