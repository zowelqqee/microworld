"""Novelty guard — the inverted support gate.

In production, ``worldpgt/cognition/support_guard.py`` enforces the QA rule:
an output is allowed only if every claim is *supported* by accepted memory;
unsupported content is rejected ("I don't know"). That gate is the single most
QA-specific assumption in the system, and this experiment inverts it rather than
deletes it — the architectural slot (a gate between reasoning and final output)
is kept, its polarity flipped:

    QA gate      : reject output NOT grounded in the corpus.
    poetry gate  : reject output that merely REPRODUCES the corpus.

So creative combination of learned images is allowed (the point of the
experiment), while verbatim memorisation of a training 4-gram is blocked. The
system must recombine, not recite.
"""

from __future__ import annotations

from dataclasses import dataclass

from poemcore.phrase_model import PhraseModel
from poemcore.text import words


@dataclass(frozen=True)
class NoveltyReport:
    total_lines: int
    echoed_lines: int
    novelty_ratio: float
    flagged: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return self.echoed_lines == 0


def check_line(model: PhraseModel, line: str) -> bool:
    """True when the line does NOT reproduce any corpus 4-gram."""

    return not model.contains_seen_4gram(words(line))


def assess_poem(model: PhraseModel, lines: list[str]) -> NoveltyReport:
    flagged = [ln for ln in lines if ln.strip() and not check_line(model, ln)]
    total = sum(1 for ln in lines if ln.strip())
    echoed = len(flagged)
    ratio = 1.0 - (echoed / total) if total else 0.0
    return NoveltyReport(
        total_lines=total,
        echoed_lines=echoed,
        novelty_ratio=round(ratio, 3),
        flagged=tuple(flagged),
    )
