"""Line-level semantic intent — a minimal plan each line is scored against.

The discourse state kept images *active* across lines, but never said what a
given line should be *about*. This layer adds that: a poem-level intent (theme,
speaker, setting, mood, core images) and a per-line intent (subject, action,
object, modifier, mood, relation-to-previous). Candidate lines are then scored
for how well they satisfy the current intent, on top of meter/rhyme/salience.

Deliberately not a parser and not an LLM: subject/action/proper-noun detection
is light Russian morphology (verb endings, a closed pronoun set) plus the
corpus-derived proper-name lexicon from ``ingest.py``. Mis-detections only nudge
a soft score, exactly the tolerance the epithet and generic-verb heuristics
already run at elsewhere in the project. Nothing here changes the n-gram order,
the corpus, or the generator's traversal — it only adds one scoring term.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from poemcore.concept_graph import ConceptGraph
from poemcore.text import content_words, words

# -- light Russian morphology (no POS model) --------------------------------- #

# Verb surface endings. Past tense (-л/-ла/-ло/-ли, reflexive -лся/-лась/…) is
# very reliable in Russian; present/future personal endings less so but still
# a useful soft signal. Only used to answer "does this line contain an action?"
_VERB_END = re.compile(
    r"(ть|ться|тся|ти|"
    r"л|ла|ло|ли|лся|лась|лось|лись|"
    r"ет|ёт|ит|ут|ют|ат|ят|ешь|ишь|ете|ите|ем|им|аю|яю|ую|ею|еет|"
    r"ешься|ишься|етесь|итесь|емся|имся|ется|ются|атся|ятся)$"
)

_PRONOUNS = frozenset({
    "я", "ты", "он", "она", "оно", "мы", "вы", "они",
    "кто", "все", "всё", "никто", "себя", "мне", "тебя", "меня",
})

_ADJ_END = re.compile(r"(ый|ий|ой|ая|яя|ое|ее|ые|ие|ым|им|ом|ою|ую|юю)$")


def has_verb(toks: list[str]) -> bool:
    return any(len(t) >= 4 and _VERB_END.search(t) for t in toks)


def has_subject(toks: list[str], cwords: list[str]) -> bool:
    if any(t in _PRONOUNS for t in toks):
        return True
    # any content word that is not verb-like and not adjective-like ≈ a noun
    for w in cwords:
        if not _VERB_END.search(w) and not _ADJ_END.search(w):
            return True
    return False


# -- theme → mood → mood vocabulary ------------------------------------------ #

MOOD_BY_THEME: dict[str, str] = {
    "autumn": "sadness", "осень": "sadness", "осени": "sadness",
    "winter": "cold", "зима": "cold",
    "spring": "hope", "весна": "hope",
    "space": "awe", "космос": "awe",
    "sea": "longing", "море": "longing",
    "love": "tenderness", "любовь": "tenderness",
    "night": "solitude", "ночь": "solitude", "ночи": "solitude",
}

_MOOD_SEEDS: dict[str, list[str]] = {
    "sadness": ["печаль", "грусть", "унылый", "увяданье", "слеза", "тоска", "туча"],
    "cold": ["мороз", "снег", "холод", "вьюга", "лёд", "иней"],
    "hope": ["заря", "свет", "весна", "цвет", "лазурь"],
    "awe": ["звезда", "небо", "бездна", "вечность", "тьма", "луна"],
    "longing": ["даль", "волна", "парус", "берег", "тоска", "море"],
    "tenderness": ["нежность", "душа", "сердце", "милый", "любовь", "уста"],
    "solitude": ["один", "тишина", "тень", "молчанье", "ночь", "сон"],
    "neutral": [],
}


def _mood_field(mood: str, graph: ConceptGraph) -> frozenset[str]:
    """Mood seeds present in the corpus plus their one-hop neighbours — the
    vocabulary that counts as 'matching the mood'. Reuses the concept graph,
    adds no new data."""

    field_: set[str] = set()
    for seed in _MOOD_SEEDS.get(mood, ()):
        if seed in graph.weight:
            field_.add(seed)
            for nb, _w in graph.neighbors(seed)[:8]:
                field_.add(nb)
    return frozenset(field_)


# -- intents ----------------------------------------------------------------- #

_RELATION_BY_MOVE = {
    "establish": "open",
    "develop": "continue",
    "leap": "associate",
    "turn": "contrast",
    "closure": "resolve",
}


@dataclass(frozen=True)
class PoemIntent:
    theme: str
    speaker: str            # the lyric subject available to any line ("я" by default)
    setting: str            # a persistent setting/place concept
    mood: str
    core_images: tuple[str, ...]
    mood_field: frozenset[str]
    allow_names: bool = False

    def to_dict(self) -> dict:
        return {
            "theme": self.theme,
            "speaker": self.speaker,
            "setting": self.setting,
            "mood": self.mood,
            "core_images": list(self.core_images),
            "mood_field": sorted(self.mood_field),
            "allow_names": self.allow_names,
        }


@dataclass(frozen=True)
class LineIntent:
    index: int
    subject: str
    action_wanted: bool     # this line should contain a verb
    object: str
    modifier: str           # setting / mood colouring
    mood: str
    relation_to_previous: str
    action: str = ""        # corpus-observed action selected by the reasoning layer

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "subject": self.subject,
            "action": self.action,
            "action_wanted": self.action_wanted,
            "object": self.object,
            "modifier": self.modifier,
            "mood": self.mood,
            "relation_to_previous": self.relation_to_previous,
        }


def build_poem_intent(
    *,
    theme: str,
    seed_concepts: list[str],
    graph: ConceptGraph,
    speaker: str = "я",
    allow_names: bool = False,
    setting: str | None = None,
    mood: str | None = None,
) -> PoemIntent:
    mood = mood if mood is not None else MOOD_BY_THEME.get(theme.lower().strip(), "neutral")
    core = tuple(dict.fromkeys(seed_concepts)) or ("тишина",)
    setting = setting or core[0]
    return PoemIntent(
        theme=theme,
        speaker=speaker,
        setting=setting,
        mood=mood,
        core_images=core,
        mood_field=_mood_field(mood, graph),
        allow_names=allow_names,
    )


def build_line_intents(poem: PoemIntent, structural_lines, decisions=None) -> list[LineIntent]:
    """One LineIntent per structural line. Reuses the structural planner's
    move/focus decisions (``planner.LinePlan``) and layers a semantic role onto
    each: the subject is the lyric speaker on framing moves and the line's focus
    image on developing moves; the object rotates through the core images."""

    core = poem.core_images
    intents: list[LineIntent] = []
    decisions_by_index = {decision.index: decision for decision in decisions or ()}
    for ln in structural_lines:
        decision = decisions_by_index.get(ln.index)
        if decision is not None:
            intents.append(
                LineIntent(
                    index=ln.index,
                    subject=decision.subject,
                    action_wanted=bool(decision.action),
                    object=decision.object,
                    modifier=decision.modifier,
                    mood=poem.mood,
                    relation_to_previous=decision.relation,
                    action=decision.action,
                )
            )
            continue
        move = ln.move
        if move in ("establish", "turn", "closure"):
            subject = poem.speaker
        else:
            subject = ln.focus or (core[ln.index % len(core)])
        obj = core[(ln.index + 1) % len(core)]
        intents.append(
            LineIntent(
                index=ln.index,
                subject=subject,
                action_wanted=True,
                object=obj,
                modifier=poem.setting,
                mood=poem.mood,
                relation_to_previous=_RELATION_BY_MOVE.get(move, "continue"),
            )
        )
    return intents


# -- scoring ----------------------------------------------------------------- #

# Weights, in the small-integer style of dialogue/constants.py. The action and
# proper-name terms are the strongest, matching the requested emphasis: reward a
# subject+action relation, punish random names and disconnected entity jumps.
SUBJECT_ACTION = 3
NO_ACTION = -3
PLAN_CONCEPT = 2
PLAN_CONCEPT_CAP = 2
# Exact decision roles have a higher value than the generic "some subject and
# some verb" morphology check above.  They make the selected line accountable
# to the thought it was meant to express, rather than merely grammatical.
DECISION_SUBJECT = 3
DECISION_ACTION = 4
DECISION_OBJECT = 3
MISSING_DECISION_ACTION = -4
MOOD_MATCH = 1
# Proper-name penalty is set to dominate the positive terms: a line with a
# random name should never win on the strength of also having a subject+verb
# and a planned concept (SUBJECT_ACTION + PLAN_CONCEPT*2 = 7), so the penalty
# must exceed that. Keeps "penalize random named entities" a hard preference.
PROPER_NAME = -8
PROPER_NAME_CAP = 3
ENTITY_JUMP = -1
ENTITY_JUMP_CAP = 3
_JUMP_ALLOWANCE = 1   # a line may introduce one new image before it counts as a jump


def line_plan_score(
    line: str,
    intent: LineIntent,
    poem: PoemIntent,
    graph: ConceptGraph,
    proper_names: frozenset[str],
    active_field: dict[str, float],
) -> tuple[float, tuple[tuple[str, float], ...]]:
    """Score a candidate line against its intent. Returns ``(score, breakdown)``
    with every point named, like ``dialogue/salience.py``."""

    toks = words(line)
    cwords = content_words(line)
    parts: list[tuple[str, float]] = []

    verb = has_verb(toks)
    subj = has_subject(toks, cwords)
    if verb and subj:
        parts.append(("subject_action", SUBJECT_ACTION))
    elif not verb:
        parts.append(("no_action", NO_ACTION))

    role_hits = decision_role_hits(line, intent)
    if role_hits["subject"]:
        parts.append(("decision_subject", DECISION_SUBJECT))
    if intent.action:
        if role_hits["action"]:
            parts.append(("decision_action", DECISION_ACTION))
        else:
            parts.append(("missing_decision_action", MISSING_DECISION_ACTION))
    if role_hits["object"]:
        parts.append(("decision_object", DECISION_OBJECT))

    planned = {intent.subject, intent.action, intent.object, intent.modifier} - {"", poem.speaker}
    hits = sum(1 for c in planned if c in cwords)
    if hits:
        parts.append(("plan_concept", PLAN_CONCEPT * min(hits, PLAN_CONCEPT_CAP)))

    if intent.mood != "neutral" and poem.mood_field and any(w in poem.mood_field for w in cwords):
        parts.append(("mood_match", MOOD_MATCH))

    if not poem.allow_names:
        pn = sum(1 for w in cwords if w in proper_names)
        if pn:
            parts.append(("proper_name", PROPER_NAME * min(pn, PROPER_NAME_CAP)))

    jumps = 0
    for w in cwords:
        if w in proper_names or w in planned:
            continue
        if active_field.get(w, 0.0) > 0.0:
            continue
        if any(nb in active_field for nb, _wt in graph.neighbors(w)[:8]):
            continue
        jumps += 1
    if jumps > _JUMP_ALLOWANCE:
        parts.append(("entity_jump", ENTITY_JUMP * min(jumps - _JUMP_ALLOWANCE, ENTITY_JUMP_CAP)))

    score = sum(points for _name, points in parts)
    return score, tuple(parts)


def decision_role_hits(line: str, intent: LineIntent) -> dict[str, bool]:
    """Exact planned-role coverage for scoring, evaluation, and trace output."""

    toks = set(words(line))
    return {
        "subject": bool(intent.subject) and intent.subject in toks,
        "action": bool(intent.action) and intent.action in toks,
        "object": bool(intent.object) and intent.object in toks,
    }


def seed_tokens(
    intent: LineIntent,
    poem: PoemIntent,
    graph: ConceptGraph,
    proper_names: frozenset[str] = frozenset(),
    *,
    max_seeds: int = 2,
    max_neighbors: int = 6,
) -> tuple[str, ...]:
    """Ordered intent tokens to try to force into a candidate line.

    Implements the soft fallback from the experiment brief: exact intent seeds
    first (subject, object, then core images, then a mood word), followed by
    their graph neighbours as related-concept fallbacks. The generator's growth
    forces the first of these that is a valid corpus predecessor; candidate
    ranking (``line_plan_score``'s ``plan_concept``) then prefers the ones that
    placed an *exact* seed, so 'exact over related' is honoured by selection
    without needing a separate pass. Pronouns, the lyric speaker, and proper
    names are excluded — a proper name IS a graph node, so an unfiltered
    neighbour walk would otherwise *seed* the very random names the plan score
    is trying to keep out."""

    def ok(w: str) -> bool:
        return bool(w) and w != poem.speaker and w not in _PRONOUNS \
            and w in graph.weight and w not in proper_names

    exact: list[str] = []
    for cand in (intent.action, intent.subject, intent.object, *poem.core_images):
        if ok(cand) and cand not in exact:
            exact.append(cand)
        if len(exact) >= max_seeds:
            break
    if len(exact) < max_seeds:
        for mw in sorted(poem.mood_field):
            if ok(mw) and mw not in exact:
                exact.append(mw)
                break

    neighbours: list[str] = []
    for s in exact:
        for nb, _w in graph.neighbors(s)[:max_neighbors]:
            if ok(nb) and nb not in exact and nb not in neighbours:
                neighbours.append(nb)
    return tuple(exact + neighbours)
