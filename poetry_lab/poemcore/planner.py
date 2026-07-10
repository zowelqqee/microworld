"""Poetic reasoning layer — plan a poem as an explicit, inspectable artifact.

This is the reasoning stage, kept structurally identical to production. In
``worldpgt/cognition/semantic_thought_graph.py`` the system spreads activation
over a graph and then *selects a sequence of moves* (decompose, ground, compare,
turn...) that will structure the answer. Here we do the same: activate the
concept graph from the prompt, then select a sequence of *poetic* moves
(establish → develop → leap → turn → closure) and bind each planned line to one
activated concept, a rhyme label, and a syllable target.

The output ``PoemPlan`` is a typed artifact, exactly like ``ReasoningTrace``:
the language layer renders it but never re-derives it. Reasoning here is
deterministic combination over the graph (activation + move selection), not
free generation — that is the whole claim being tested.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from poemcore.concept_graph import ConceptGraph
from poemcore.text import STOPWORDS

# Poetic moves — the creative-writing analogue of the production ThoughtMoveKind
# enum. Each move says how a line should relate to the images already in play.
MOVES = ("establish", "develop", "leap", "turn", "closure")


@dataclass(frozen=True)
class LinePlan:
    index: int
    move: str
    focus: str            # activated concept this line is built around
    rhyme_label: str      # A/B/... lines sharing a label must rhyme
    target_syllables: int
    prefer_forward: bool  # opener-seeded (True) vs rhyme-seeded (False) growth

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "move": self.move,
            "focus": self.focus,
            "rhyme_label": self.rhyme_label,
            "target_syllables": self.target_syllables,
        }


@dataclass
class PoemPlan:
    theme: str
    style_author: str
    rhyme_scheme: str
    target_syllables: int
    lines: list[LinePlan] = field(default_factory=list)
    activated: list[tuple[str, float]] = field(default_factory=list)
    seed_concepts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "theme": self.theme,
            "style_author": self.style_author,
            "rhyme_scheme": self.rhyme_scheme,
            "target_syllables": self.target_syllables,
            "seed_concepts": self.seed_concepts,
            "activated": [{"concept": c, "activation": round(a, 3)} for c, a in self.activated],
            "lines": [ln.to_dict() for ln in self.lines],
        }


def plan_poem(
    graph: ConceptGraph,
    seed_concepts: list[str],
    *,
    theme: str,
    style_author: str,
    stanzas: int = 2,
    lines_per_stanza: int = 4,
    rhyme_scheme: str = "ABAB",
    target_syllables: int = 8,
    continuation: bool = False,
) -> PoemPlan:
    """Build a line-by-line plan by spreading activation, then assigning the
    ranked concepts to a move sequence across the requested stanza structure."""

    n_lines = stanzas * lines_per_stanza
    # ``ConceptGraph.activate`` retains every node's corpus base weight.  Its
    # global top-k can therefore be dominated by high-frequency words which
    # have no path from the prompt.  A reasoning plan must only choose from the
    # seed-connected field; otherwise it makes an explicit but arbitrary
    # decision such as subject="кто" for an autumn request.
    activated = _local_activation(graph, seed_concepts, k=max(12, n_lines * 2))
    # Prompt seeds are commitments, not merely hints.  Schedule them before
    # looser graph associations so the line decisions stay about the requested
    # subject matter; associations extend that thought only after its core
    # images have been stated.
    pool = list(dict.fromkeys(seed_concepts))
    pool.extend(c for c, _ in activated if c not in pool)
    if not pool:
        pool = seed_concepts or ["тишина"]

    lines: list[LinePlan] = []
    for i in range(n_lines):
        pos_in_stanza = i % lines_per_stanza
        move = _move_for_position(pos_in_stanza, lines_per_stanza, i, n_lines, continuation)
        focus = _focus_for_move(move, i, seed_concepts, pool)
        rhyme_label = rhyme_scheme[pos_in_stanza % len(rhyme_scheme)]
        # Odd-positioned rhyme labels grown from the rhyme word backward;
        # scene-establishing openers grown forward from a learned opener.
        prefer_forward = move in ("establish",) and pos_in_stanza == 0
        lines.append(
            LinePlan(
                index=i,
                move=move,
                focus=focus,
                rhyme_label=rhyme_label,
                target_syllables=target_syllables,
                prefer_forward=prefer_forward,
            )
        )

    return PoemPlan(
        theme=theme,
        style_author=style_author,
        rhyme_scheme=rhyme_scheme,
        target_syllables=target_syllables,
        lines=lines,
        activated=activated,
        seed_concepts=seed_concepts,
    )


def _move_for_position(
    pos: int, per_stanza: int, global_i: int, n_lines: int, continuation: bool
) -> str:
    if global_i == 0:
        return "develop" if continuation else "establish"
    if global_i == n_lines - 1:
        return "closure"
    # a volta near the final stanza boundary
    if pos == 0 and global_i >= n_lines - per_stanza:
        return "turn"
    if pos == 0:
        return "develop"
    if pos % 2 == 1:
        return "leap"
    return "develop"


def _focus_for_move(move: str, i: int, seeds: list[str], pool: list[str]) -> str:
    if move == "establish":
        return seeds[0] if seeds else pool[0]
    if move == "closure":
        return seeds[0] if seeds else pool[i % len(pool)]
    # develop/leap/turn draw from the activated associations, rotating so the
    # poem moves through several images rather than repeating one
    return pool[i % len(pool)]


def _local_activation(
    graph: ConceptGraph, seeds: list[str], *, k: int
) -> list[tuple[str, float]]:
    """Rank only direct concept associations of the prompt seeds.

    Two hops were enough for high-degree corpus hubs to reintroduce unrelated
    function-like nodes. Direct co-occurrence stays a meaningful association
    while grounding the decision in the prompted imagery.
    """

    activation = graph.activate(seeds)
    local: set[str] = {seed for seed in seeds if _plan_concept(seed)}
    for source in tuple(local):
        for target, _weight in graph.neighbors(source):
            if _plan_concept(target):
                local.add(target)
    ranked = sorted(((word, activation.get(word, 0.0)) for word in local), key=lambda item: (-item[1], item[0]))
    return ranked[:k]


def _plan_concept(word: str) -> bool:
    return len(word) > 2 and word not in STOPWORDS
