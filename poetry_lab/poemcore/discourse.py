"""Discourse state + line salience — port of MicroWorld's dialogue selection.

This is the transfer chosen in ``DESIGN_next_experiment.md``. Two production
mechanisms are ported, unchanged in shape, only re-domained:

  * ``worldpgt/dialogue/state.py::EntityActivation`` — a per-conversation record
    of which entities are active, how recently, how often. Here it becomes a
    per-*poem* record of which images are active across lines: ``DiscourseState``.

  * ``worldpgt/dialogue/salience.py::base_salience`` — scores a candidate
    against that state with an integer breakdown, every point named, so the
    trace is a proof of the choice. Here ``line_salience`` scores a candidate
    *line* against the poem's active images the same way.

The generator already produces several candidate lines per slot (its retry
loop) and keeps only the first valid one. Feeding those candidates through this
scorer — and keeping the one that best continues the images already in play —
is what turns locally-grammatical lines into a poem that develops a subject
across lines. Meter/rhyme/novelty are gated *before* scoring, so they cannot
regress; this layer only decides *between* already-valid lines. No neural model,
no language model — integer/float arithmetic over the existing concept graph,
exactly as in ``salience.py``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from poemcore.concept_graph import ConceptGraph
from poemcore.text import content_words

# Scoring constants, in the spirit of ``worldpgt/dialogue/constants.py``: small
# integers, each a named reason that shows up in the breakdown.
ACTIVE_SELF = 4      # a candidate word *is* a currently-active image (direct continuity)
ACTIVE_LINK = 3      # a candidate word is graph-adjacent to an active image
FOCUS_MATCH = 4      # the candidate realizes this line's planned focus concept
REPETITION = -2      # per prior use of a word already placed in the poem
REPETITION_CAP = 3   # ... capped, like salience.py caps its mention bonus
DECAY = 0.55         # per-line decay of active salience == the recency penalty
_NEIGHBOUR_SPREAD = 0.5   # weight carried one hop out from an active image
_NEIGHBOUR_LIMIT = 12     # same fan-out cap the coherence eval uses


@dataclass
class DiscourseState:
    """Which images are active across the poem so far, how recently, how often.

    Structural port of ``EntityActivation`` scoped to one poem instead of one
    conversation. ``active`` is the decaying salience map (recency); ``history``
    is the mention counter (repetition control)."""

    active: dict[str, float] = field(default_factory=dict)
    history: Counter = field(default_factory=Counter)
    lines_done: int = 0

    @classmethod
    def seeded(cls, concepts: list[str]) -> "DiscourseState":
        """Start the poem already 'thinking about' its seed images, so the very
        first line has a topic to continue (and continuation mode coheres with
        the given verse, whose words are the seeds)."""

        state = cls()
        for c in concepts:
            state.active[c] = 1.0
        return state

    def salience_field(self, graph: ConceptGraph) -> dict[str, float]:
        """The active images plus their one-hop neighbourhood, weighted — the
        spread-activation field a candidate line is scored against. Reuses
        ``concept_graph.neighbors`` (spreading activation) rather than adding
        any new traversal."""

        field_: dict[str, float] = {}
        for concept, weight in self.active.items():
            if weight > field_.get(concept, 0.0):
                field_[concept] = weight
            for nb, _w in graph.neighbors(concept)[:_NEIGHBOUR_LIMIT]:
                spread = weight * _NEIGHBOUR_SPREAD
                if spread > field_.get(nb, 0.0):
                    field_[nb] = spread
        return field_

    def update(self, line: str) -> None:
        """Decay every active image (recency), then refresh with this line's
        images and bump their mention counts."""

        self.active = {c: w * DECAY for c, w in self.active.items() if w * DECAY > 0.05}
        for w in content_words(line):
            self.active[w] = 1.0
            self.history[w] += 1
        self.lines_done += 1


def line_salience(
    line: str,
    state: DiscourseState,
    graph: ConceptGraph,
    focus: str,
    *,
    field_: dict[str, float] | None = None,
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Score a candidate line for how well it continues the poem so far.

    Port of ``salience.base_salience``: returns ``(score, breakdown)`` where the
    breakdown is the list of named point contributions — the proof of the
    score. ``field_`` may be precomputed once per line and shared across the
    line's candidates (the active field does not change between candidates).
    """

    if field_ is None:
        field_ = state.salience_field(graph)
    parts: list[tuple[str, float]] = []
    cwords = content_words(line)

    for w in cwords:
        if w in state.active:
            parts.append(("active_self", ACTIVE_SELF))
        elif field_.get(w, 0.0) > 0.0:
            parts.append(("active_link", round(ACTIVE_LINK * field_[w], 3)))

    if focus and focus in cwords:
        parts.append(("focus_match", FOCUS_MATCH))

    for w in cwords:
        prior = state.history.get(w, 0)
        if prior:
            parts.append(("repetition", REPETITION * min(prior, REPETITION_CAP)))

    # Normalize by content-word count: score is continuity *density*, not raw
    # count. Without this, a line scores higher simply by having more words,
    # which biases selection toward longer (off-meter) lines. Meter is already
    # gated to a window by the caller; this removes the residual length reward.
    raw = sum(points for _name, points in parts)
    score = raw / max(1, len(cwords))
    return score, tuple(parts)
