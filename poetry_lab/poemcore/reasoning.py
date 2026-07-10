"""Explicit thought planning for the poetry transfer experiment.

This module is deliberately smaller than a language generator.  It commits to
what each stanza and line is meant to express before the phrase graph starts a
walk.  The decisions are drawn only from the existing prompt seeds, concept
graph, phrase model, and structural ``PoemPlan``; no corpus or learned model is
added here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from poemcore.concept_graph import ConceptGraph
from poemcore.line_plan import MOOD_BY_THEME, _ADJ_END, _VERB_END
from poemcore.phrase_model import PhraseModel, seeded_weighted_pick
from poemcore.planner import PoemPlan
from poemcore.text import STOPWORDS


@dataclass(frozen=True)
class PoemGoal:
    theme: str
    speaker: str
    mood: str
    setting: str
    core_images: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "theme": self.theme,
            "speaker": self.speaker,
            "mood": self.mood,
            "setting": self.setting,
            "core_images": list(self.core_images),
        }


@dataclass(frozen=True)
class StanzaPlan:
    index: int
    purpose: str
    anchor: str
    progression: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "purpose": self.purpose,
            "anchor": self.anchor,
            "progression": list(self.progression),
        }


@dataclass(frozen=True)
class LineDecision:
    index: int
    stanza_index: int
    purpose: str
    relation: str
    subject: str
    action: str
    object: str
    modifier: str

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "stanza_index": self.stanza_index,
            "purpose": self.purpose,
            "relation": self.relation,
            "subject": self.subject,
            "action": self.action,
            "object": self.object,
            "modifier": self.modifier,
        }


@dataclass
class PoemReasoning:
    """The inspectable reasoning boundary between structural plan and render."""

    goal: PoemGoal
    stanzas: list[StanzaPlan] = field(default_factory=list)
    lines: list[LineDecision] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "goal": self.goal.to_dict(),
            "stanzas": [stanza.to_dict() for stanza in self.stanzas],
            "line_decisions": [line.to_dict() for line in self.lines],
        }


_RELATION_BY_MOVE = {
    "establish": "introduce",
    "develop": "develop",
    "leap": "associate",
    "turn": "reconsider",
    "closure": "resolve",
}

# The shared lightweight verb-ending heuristic is intentionally permissive for
# candidate scoring.  A thought trace needs a stricter bar: these frequent
# noun forms happen to end in ``-ет`` but cannot be the selected action.
_NON_ACTION_WORDS = frozenset({
    "свет", "совет", "ответ", "привет", "цвет", "завет", "предмет", "портрет", "поэт",
    "будет", "будут",
})

# A higher-precision subset of the permissive scorer heuristic.  The generic
# ``_VERB_END`` deliberately treats many endings as possible verbs; a thought
# planner needs a stricter bar because a false action becomes an explicit claim.
_ACTION_FORM = re.compile(
    r"(?:ться|тся|[аяеои]л(?:а|о|и)?|лся|лась|лось|лись|"
    r"ают|яют|уют|ат|ят|ает|яет|ует|ит|ёт|ет|"
    r"аешь|яешь|уешь|ишь|ешь|аете|яете|уете|ите|"
    r"аем|яем|уем|аемся|яемся|уются|аются|яются|"
    r"аю|яю|ую|[аяеёиоуыэюя]ть)$"
)


def reason_poem(
    graph: ConceptGraph,
    phrase: PhraseModel,
    plan: PoemPlan,
    *,
    theme: str,
    speaker: str = "я",
) -> PoemReasoning:
    """Make deterministic semantic decisions for an already-structured poem.

    Structural planning still owns meter, rhyme, and move positions.  This
    layer supplies the missing propositional content: a stable goal, a purpose
    for each stanza, and an SVO-like decision for every line.  The renderer may
    fail to realize a decision when no valid trigram path contains its token;
    that failure remains visible in the printed artifact instead of being
    silently converted into a different thought.
    """

    core = tuple(dict.fromkeys(plan.seed_concepts)) or ("тишина",)
    goal = PoemGoal(
        theme=theme,
        speaker=speaker,
        mood=MOOD_BY_THEME.get(theme.lower().strip(), "neutral"),
        setting=core[0],
        core_images=core,
    )
    per_stanza = max(1, len(plan.rhyme_scheme))
    grouped = [plan.lines[i : i + per_stanza] for i in range(0, len(plan.lines), per_stanza)]
    stanzas = [
        StanzaPlan(
            index=index,
            purpose=_stanza_purpose(index, len(grouped)),
            anchor=_anchor(lines, goal),
            progression=tuple(dict.fromkeys(line.focus for line in lines if line.focus)),
        )
        for index, lines in enumerate(grouped)
    ]

    decisions: list[LineDecision] = []
    for line in plan.lines:
        stanza = stanzas[line.index // per_stanza]
        relation = _RELATION_BY_MOVE.get(line.move, "develop")
        subject, object_ = _roles(line.move, line.focus, stanza.anchor, goal)
        # Decisions are chosen before traversal, but planning may consult the
        # phrase graph to select a *realizable* SVO shape.  A clause is two
        # overlapping learned trigrams (subject, bridge, action) and
        # (bridge, action, object), never a learned 4-gram or a new corpus.
        surface_roles = _choose_surface_roles(
            graph, phrase, (subject, object_, goal.setting), f"{line.index}|{relation}"
        )
        if surface_roles is not None:
            subject, action, object_ = surface_roles
        else:
            action = _choose_action(graph, phrase, subject, object_, f"{line.index}|{relation}")
        decisions.append(
            LineDecision(
                index=line.index,
                stanza_index=stanza.index,
                purpose=stanza.purpose,
                relation=relation,
                subject=subject,
                action=action,
                object=object_,
                modifier=goal.setting,
            )
        )
    return PoemReasoning(goal=goal, stanzas=stanzas, lines=decisions)


def _stanza_purpose(index: int, total: int) -> str:
    if total <= 1:
        return "develop_and_resolve"
    if index == 0:
        return "introduce"
    if index == total - 1:
        return "turn_and_resolve"
    return "develop"


def _anchor(lines, goal: PoemGoal) -> str:
    for line in lines:
        if line.focus and line.focus != goal.setting:
            return line.focus
    return goal.setting


def _roles(move: str, focus: str, anchor: str, goal: PoemGoal) -> tuple[str, str]:
    if move == "establish":
        return focus or goal.setting, goal.setting
    if move in ("turn", "closure"):
        return goal.speaker, focus or anchor
    if move == "leap":
        return focus or anchor, goal.setting
    return focus or anchor, anchor


def _choose_action(
    graph: ConceptGraph, phrase: PhraseModel, subject: str, object_: str, seed: str
) -> str:
    """Choose a corpus-observed action associated with the decided images."""

    candidates: dict[str, int] = {}
    for concept in (subject, object_):
        for word, edge_weight in graph.neighbors(concept):
            if _is_action(word, phrase):
                candidates[word] = max(candidates.get(word, 0), int(edge_weight * 100) + phrase.unigram[word])
    if not candidates:
        for word, count in phrase.unigram.items():
            if _is_action(word, phrase):
                candidates[word] = count
    if not candidates:
        return ""
    # A line decision is only useful when the language graph has a plausible
    # way to express it.  Prefer verbs reachable from the subject or leading
    # toward the object, while retaining graph association as the semantic
    # source of the candidate set.  ``link_score`` walks the existing 1/2-hop
    # transition tables; it does not introduce a larger language model.
    connected: list[tuple[str, int]] = []
    for word, semantic_weight in candidates.items():
        surface_weight = phrase.slot_link_score(subject, word) + phrase.slot_link_score(word, object_)
        if surface_weight:
            connected.append((word, semantic_weight + surface_weight * 100))
    # A line without an available verbal route must remain visibly incomplete;
    # selecting a merely co-occurring verb would create an arbitrary thought.
    return seeded_weighted_pick(seed, "line_action", connected) if connected else ""


def _choose_surface_roles(
    graph: ConceptGraph, phrase: PhraseModel, concepts: tuple[str, ...], seed: str
) -> tuple[str, str, str] | None:
    """Select an SVO-shaped decision supported by two overlapping trigrams.

    The input concepts are the semantic commitments from the stanza/line plan.
    We inspect only those words and their direct graph neighbours, then seek a
    short phrase path that can carry one of them as subject or object.  This is
    the planning analogue of QA choosing a relation it can actually traverse:
    a thought stays thematic, while its surface slots become attainable.
    """

    anchors: list[str] = []
    for concept in concepts:
        if concept and concept in phrase.unigram and concept not in anchors:
            anchors.append(concept)
        for neighbour, _weight in graph.neighbors(concept)[:8]:
            if neighbour in phrase.unigram and neighbour not in anchors and not _is_action(neighbour, phrase):
                anchors.append(neighbour)

    preferred = set(concepts)
    candidates: dict[tuple[str, str, str], int] = {}
    for subject in anchors:
        bridges = sorted(phrase.forward.get(subject, {}).items(), key=lambda item: (-item[1], item[0]))[:32]
        for bridge, bridge_count in bridges:
            actions = phrase.forward2.get((subject, bridge), {})
            for action, action_count in actions.items():
                if not _is_action(action, phrase):
                    continue
                objects = phrase.forward2.get((bridge, action), {})
                for object_, object_count in objects.items():
                    if not _surface_object(object_, phrase):
                        continue
                    relevance = 0
                    if subject in preferred:
                        relevance += 8
                    if object_ in preferred:
                        relevance += 8
                    if not relevance:
                        continue
                    weight = relevance * 1000 + min(bridge_count, 20) + min(action_count, 20) + min(object_count, 20)
                    key = (subject, action, object_)
                    candidates[key] = max(candidates.get(key, 0), weight)
    # The thematic concept is often an object ("... leaves ..."), particularly
    # in inflected Russian prompts.  Recover the same two-trigram clause from
    # its object end so the semantic anchor need not be forced into nominative
    # subject position.
    for object_ in anchors:
        actions = phrase.backward.get(object_, {})
        for action, action_count in actions.items():
            if not _is_action(action, phrase):
                continue
            bridges = phrase.backward2.get((action, object_), {})
            for bridge, bridge_count in bridges.items():
                subjects = phrase.backward2.get((bridge, action), {})
                for subject, subject_count in subjects.items():
                    if not _surface_object(subject, phrase):
                        continue
                    relevance = 8 if object_ in preferred else 0
                    if subject in preferred:
                        relevance += 8
                    if not relevance:
                        continue
                    weight = relevance * 1000 + min(action_count, 20) + min(bridge_count, 20) + min(subject_count, 20)
                    key = (subject, action, object_)
                    candidates[key] = max(candidates.get(key, 0), weight)
    if not candidates:
        return None
    encoded = [("\t".join(roles), weight) for roles, weight in candidates.items()]
    return tuple(seeded_weighted_pick(seed, "surface_roles", encoded).split("\t"))  # type: ignore[return-value]


def _surface_object(word: str, phrase: PhraseModel) -> bool:
    return (
        len(word) >= 2
        and word not in STOPWORDS
        and not _is_action(word, phrase)
        and word in phrase.unigram
    )


def _is_action(word: str, phrase: PhraseModel) -> bool:
    return (
        len(word) >= 4
        and word not in _NON_ACTION_WORDS
        and word not in STOPWORDS
        and not _ADJ_END.search(word)
        and bool(_VERB_END.search(word))
        and bool(_ACTION_FORM.search(word))
        and word in phrase.unigram
    )
