"""Narrative planning and realization over the unchanged MicroWorld core.

This module is intentionally a surface-layer sibling of the poetry modules.
It keeps the concept graph, trigram phrase graph, discourse salience, candidate
ranking, and novelty gate; it replaces only verse-specific units:

  PoemGoal / StanzaPlan / LinePlan / line realization
  -> SceneGoal / ScenePlan / SentencePlan / sentence realization
"""

from __future__ import annotations

import re
import secrets
from dataclasses import dataclass, field, replace
from pathlib import Path

from poemcore.concept_graph import ConceptGraph
from poemcore.discourse import DiscourseState, line_salience
from poemcore.ingest import load_artifacts
from poemcore.line_plan import _ADJ_END
from poemcore.morphology import is_finite_verb, is_infinitive, past_gender, sentence_agreement_errors
from poemcore.novelty import NoveltyReport, assess_poem, check_line
from poemcore.phrase_model import PhraseModel, seeded_weighted_pick
from poemcore.reasoning import _choose_action, _is_action
from poemcore.text import STOPWORDS, content_words, words
from poemcore.narrative_reasoning import Hypothesis, event_hypothesis
from poemcore.plan_search import ReasoningPlan, beam_search
from poemcore.world_state import StateDelta, StateFact, WorldState

_SCENE_SEEDS = {
    "dialogue": ("берлиоз", "иван", "разговор"),
    "диалог": ("берлиоз", "иван", "разговор"),
    "room": ("комната", "квартира", "дверь"),
    "комнат": ("комната", "квартира", "дверь"),
    "moscow": ("москва", "вечер", "улица"),
    "москв": ("москва", "вечер", "улица"),
    "evening": ("вечер", "солнце", "город"),
    "вечер": ("вечер", "солнце", "город"),
    "character": ("человек", "иностранец", "берлиоз"),
    "персонаж": ("человек", "иностранец", "берлиоз"),
}
_INCOMPLETE_TAILS = STOPWORDS | frozenset({
    "весь", "вся", "все", "всей", "этой", "этого", "этом", "которого", "которой",
    "такой", "никакого", "долго", "потом", "вправо", "влево", "дальше", "совершенно",
})
_FRAGMENT_OBJECT_NOISE = frozenset({
    "свое", "свой", "свои", "свою", "своим", "своего", "своей",
    "этот", "эта", "это", "эту", "этим", "этой", "этих",
    "такой", "такие", "такую", "весь", "возле", "после", "вокруг",
    "очень", "совершенно", "вдруг", "сейчас", "здесь", "тоже",
})
# Short-form adjective endings (e.g. "невидима", "любима").  They are
# morphological evidence, not a manually curated vocabulary, and must not be
# admitted as event objects merely because a permissive verb ending matched.
_SHORT_ADJ_END = re.compile(r"(?:има|ыма|ема|ома|ена|ана|ита|ута|ята)$")
_FIRST_PERSON = frozenset({"я", "мы", "мне", "меня", "мой", "моя", "моё", "мои"})
_PROMPT_WORD_RE = re.compile(r"[А-ЯЁа-яё]+(?:-[А-ЯЁа-яё]+)?")
_REQUEST_SHAPE_WORDS = frozenset({
    "предложение", "предложения", "предложений", "предложениях",
    "абзац", "абзаца", "абзацев", "абзаце", "слов", "слова",
    "одном", "двух", "трех", "трёх", "четырех", "четырёх", "пяти",
    "шести", "семи", "восьми", "девяти", "десяти",
})
_DISCOURSE_MARKERS = frozenset({"затем", "потом", "тогда", "вскоре", "вдруг", "позже"})

# Interrogative-frame words: legitimate narrative content in a declarative
# sentence ("Такой человек не мог быть злым", "Иван делает шаг"), but when a
# prompt is a *question* they are the word the scene planner sees first ("Кто
# такой Воланд?" -> "такой" precedes "воланд" in reading order), so they were
# winning goal.topic/speaker/location over the entity the question is actually
# about. Kept local to scene-goal parsing rather than added to the shared
# STOPWORDS set (which also gates concept-graph training on the novel's own
# sentences, where these words are ordinary content).
_INTERROGATIVE_MARKERS = frozenset({
    "почему", "зачем", "куда", "откуда",
    "каков", "какова", "каково", "каковы", "сколько", "скольких",
    "чей", "чья", "чьё", "чьи",
    "какой", "какая", "какое", "какие", "который", "которую", "которых",
    "такой", "такая", "такое", "такие",
    "случилось", "случилась", "произошло", "произошла", "происходит", "происходило",
    "делает", "делают", "означает", "значит",
})


@dataclass(frozen=True)
class SceneGoal:
    topic: str
    mode: str
    speaker: str
    location: str
    characters: tuple[str, ...]
    seeds: tuple[str, ...]
    active_field: tuple[str, ...] = ()
    scene_objective: str = "advance_events"

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "mode": self.mode,
            "speaker": self.speaker,
            "location": self.location,
            "characters": list(self.characters),
            "seeds": list(self.seeds),
            "active_field": list(self.active_field),
            "scene_objective": self.scene_objective,
        }


@dataclass(frozen=True)
class SceneState:
    """Inspectable non-character reasoning state for one scene.

    It is intentionally a plan-layer object.  The renderer receives only the
    selected sentence plans and cannot add a new topic, relationship, or mood
    on its own.
    """

    dominant_tone: str
    pacing: str
    narrative_focus: str
    emotional_trajectory: tuple[str, ...]
    active_conflict: str
    scene_objective: str
    active_field: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "dominant_tone": self.dominant_tone,
            "pacing": self.pacing,
            "narrative_focus": self.narrative_focus,
            "emotional_trajectory": list(self.emotional_trajectory),
            "active_conflict": self.active_conflict,
            "scene_objective": self.scene_objective,
            "active_field": list(self.active_field),
        }


@dataclass(frozen=True)
class SentencePlan:
    index: int
    purpose: str
    focus: str
    subject: str
    action: str
    object: str
    speaker: str
    dialogue: bool
    continuity_anchor: str = ""
    transition: str = ""
    target_words: int = 16
    minimum_words: int = 4
    fragment: tuple[str, ...] = ()
    detail_fragment: tuple[str, ...] = ()
    connector: str = ""
    relation: str = ""
    clause_fragments: tuple[tuple[str, ...], ...] = ()
    clause_connectors: tuple[str, ...] = ()
    # Descriptive fact bundle: an agreeing observed epithet for the subject and
    # an observed subject-to-object preposition link ("в", "трактире"). Both
    # are reasoning-layer selections; the renderer only places them.
    epithet: str = ""
    link: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "purpose": self.purpose,
            "focus": self.focus,
            "subject": self.subject,
            "action": self.action,
            "object": self.object,
            "speaker": self.speaker,
            "dialogue": self.dialogue,
            "continuity_anchor": self.continuity_anchor,
            "transition": self.transition,
            "target_words": self.target_words,
            "minimum_words": self.minimum_words,
            "fragment": list(self.fragment),
            "detail_fragment": list(self.detail_fragment),
            "connector": self.connector,
            "relation": self.relation,
            "clause_fragments": [list(fragment) for fragment in self.clause_fragments],
            "clause_connectors": list(self.clause_connectors),
            "epithet": self.epithet,
            "link": list(self.link),
        }


@dataclass(frozen=True)
class ScenePlan:
    goal: SceneGoal
    beats: tuple[str, ...]
    sentences: tuple[SentencePlan, ...]

    def to_dict(self) -> dict:
        return {
            "goal": self.goal.to_dict(), "beats": list(self.beats),
            "sentences": [item.to_dict() for item in self.sentences],
        }


@dataclass(frozen=True)
class SentenceRealization:
    index: int
    planned: tuple[str, str, str]
    realized: tuple[str, ...]
    dialogue: bool
    realized_facts: tuple[tuple[str, str, str], ...] = ()

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "planned": {"subject": self.planned[0], "action": self.planned[1], "object": self.planned[2]},
            "realized": list(self.realized),
            "dialogue": self.dialogue,
            "realized_facts": [list(fact) for fact in self.realized_facts],
        }


@dataclass
class Paragraph:
    sentences: list[str] = field(default_factory=list)
    realization: list[SentenceRealization] = field(default_factory=list)

    def text(self) -> str:
        return " ".join(self.sentences)


@dataclass
class NarrativeRequest:
    prompt: str
    context: str = ""
    sentences: int = 4

    def to_dict(self) -> dict:
        return {"prompt": self.prompt, "context": self.context, "sentences": self.sentences}


@dataclass
class NarrativeResult:
    request: NarrativeRequest
    plan: ScenePlan
    paragraph: Paragraph
    novelty: NoveltyReport
    meta: dict
    reasoning_plan: ReasoningPlan | None = None
    world_state: WorldState | None = None
    scene_state: SceneState | None = None


def plan_scene(
    graph: ConceptGraph,
    phrase: PhraseModel,
    *,
    prompt: str,
    context: str,
    meta: dict,
    proper_names: frozenset[str],
    sentence_count: int,
    entity_types: dict[str, str] | None = None,
    genders: dict[str, str] | None = None,
    noun_like: frozenset[str] = frozenset(),
    verb_lexicon: dict | None = None,
    speech_frames: tuple[dict, ...] = (),
    description_relations: dict[str, list[dict]] | None = None,
    selection_seed: str = "",
) -> ScenePlan:
    # ``plan_scene`` stays replayable by default for unit tests and offline
    # audits.  The engine supplies a fresh seed for an unseeded user run.
    variant_seed = selection_seed or "default"
    low = prompt.lower()
    mapped = [
        resolved for key, values in _SCENE_SEEDS.items() if key in low
        for resolved in (_corpus_form(graph, seed) for seed in values) if resolved
    ]
    name_stems = _name_stems(proper_names)

    def _normalize(word: str) -> str:
        """Case-normalize a known name, else keep the surface word."""
        return _canonical_entity(word, proper_names, name_stems) or word

    # Entity resolution runs *before* the graph-membership filter. A rarely
    # inflected name ("Бегемоте", prepositional) is absent from the concept
    # graph even though its canonical form ("бегемот") is present, so filtering
    # first silently discarded the very entity the prompt was about and left
    # only the common noun beside it ("коте").
    context_words = [word for word in map(_normalize, content_words(context)) if word in graph.weight]
    command_words = {"напиши", "опиши", "продолжи", "сделай", "write", "describe", *_REQUEST_SHAPE_WORDS}
    prompt_words_all = [
        word for word in map(_normalize, content_words(prompt))
        if word in graph.weight and word not in command_words
    ]
    prompt_surface = _PROMPT_WORD_RE.findall(prompt)
    prompt_entities = [
        token.lower() for index, token in enumerate(prompt_surface)
        if token[:1].isupper()
        and token.lower() not in {"напиши", "опиши", "продолжи", "сделай", "write", "describe"}
        and (index == 0 or prompt_surface[index - 1].lower() in {"о", "об", "про", "для", "about"})
    ]
    # Drop interrogative-frame words before they can become the scene topic.
    # An entirely interrogative prompt ("Кто это?") has nothing else to fall
    # back on, so the unfiltered list is kept as a last resort rather than
    # emptying the seed pool outright.
    prompt_words = [word for word in prompt_words_all if word not in _INTERROGATIVE_MARKERS] or prompt_words_all
    seeds = list(dict.fromkeys(context_words + prompt_words + [word for word in mapped if word in graph.weight]))
    if not seeds and prompt_entities:
        seeds = prompt_entities[:3]
    locations = tuple(meta.get("recurring_locations", ()))
    characters = tuple(meta.get("recurring_characters", ()))
    if not seeds:
        seeds = [word for word in (*locations, *characters, *meta.get("recurring_objects", ())) if word in graph.weight][:3]
    seeds = seeds or [next(iter(graph.weight), "человек")]
    # A named entity in the prompt is a stronger topic signal than any generic
    # noun/verb, regardless of where it falls in reading order ("Почему Понтий
    # Пилат наказал Иешуа?" should ground on Пилат, not on the leftover
    # interrogative residue or the verb). Order preserved within each group.
    named = list(dict.fromkeys(word for word in seeds if word in proper_names))
    if named:
        seeds = named + [word for word in seeds if word not in named]
    core_seeds = tuple(seeds)
    # A single-entity question ("Кто такой Воланд?") legitimately reduces to
    # one seed once interrogative words are stripped. That starves the
    # per-sentence fragment search of anchors and made it collapse onto
    # whichever rare corpus path happened to survive (see README). Widening
    # with the topic's own graph neighbours gives the *same* reasoning step
    # more to reason over, without adding a corpus or forcing any of them into
    # the rendered sentence — they only broaden what counts as "relevant".
    if seeds:
        # Walk the local concept field instead of always taking the first
        # neighbours.  The walk is still bounded and corpus-only, but the
        # seed selects a different admissible route through the graph.
        frontier = list(seeds)
        for depth in range(2):
            next_frontier: list[str] = []
            for anchor in frontier:
                options = [
                    (neighbour, max(1, int(weight)))
                    for neighbour, weight in graph.neighbors(anchor)
                    if neighbour in graph.weight and neighbour not in proper_names
                    and neighbour not in STOPWORDS and neighbour not in seeds
                ]
                if not options:
                    continue
                pool = list(options)
                for pick_index in range(2):
                    if not pool:
                        break
                    chosen = seeded_weighted_pick(
                        variant_seed, f"concept-hop:{depth}:{anchor}:{pick_index}", pool
                    )
                    seeds.append(chosen)
                    next_frontier.append(chosen)
                    pool = [(word, weight) for word, weight in pool if word != chosen]
                    if len(seeds) >= 5:
                        break
                if len(seeds) >= 5:
                    break
            frontier = next_frontier
            if len(seeds) >= 5 or not frontier:
                break
    # --- ported QA knowledge layer, used as advice ------------------------
    # ``entity_types`` is the corpus-grounded classifier from entity_types.py.
    # It never blocks anything: an "unknown" type reproduces the previous
    # role assignment exactly. It only prevents the one destructive mistake
    # the planner used to make — putting a *place* in the agent slot, so that
    # "Патриарших прудах появились" had to be built as if a pond could act.
    types = entity_types or {}

    def _type_of(word: str) -> str:
        return types.get(word, "unknown")

    # A compound name ("Понтий Пилат", "Иван Бездомный") arrives as two
    # separate lexicon entries; no multi-word grouping exists. Prefer the more
    # corpus-salient of them as the entity the scene is actually about.
    if len(named) > 1:
        primary = max(named, key=lambda word: graph.weight.get(word, 0.0))
        named = [primary] + [word for word in named if word != primary]
    field_seeds = tuple(dict.fromkeys(seeds))
    # Expanded graph nodes are anchors for event relevance, never new scene
    # protagonists. This keeps a Voland scene from turning into "длинную
    # вытирая" merely because that edge won the exploratory walk.
    seeds = list(dict.fromkeys(named + [word for word in core_seeds if word not in named])) if named else list(core_seeds)

    dialogue = "dialog" in low or "диалог" in low
    introduce = "new character" in low or "introduce" in low or "персонаж" in low
    continuation = bool(context) or "continue" in low or "продолж" in low or "after this" in low
    describe = any(marker in low for marker in ("опиши", "описать", "describe", "опиши"))
    mode = (
        "dialogue" if dialogue else "introduction" if introduce else "continuation"
        if continuation else "description" if describe else "scene"
    )
    mentioned = list(dict.fromkeys(
        canon for word in context_words
        if (canon := _canonical_entity(word, proper_names, name_stems))
    ))
    available_characters = [word for word in characters if word in graph.weight]

    # A descriptive request is grounded in the requested concept field, not
    # silently converted into a character scene.  In particular, no recurring
    # protagonist may leak into "Опиши Москву" simply because prose events
    # normally need an actor.
    if mode == "description":
        active_field = _topic_field(
            graph, tuple(seeds), phrase, noun_like, proper_names, description_relations or {},
        )
        location = next((word for word in seeds if _type_of(word) == "place"), "")
        goal = SceneGoal(
            topic=seeds[0], mode=mode, speaker="", location=location,
            characters=(), seeds=tuple(seeds[:5]), active_field=active_field,
            scene_objective="describe_grounded_field",
        )
        return _plan_description_scene(
            goal, graph, phrase, sentence_count=sentence_count,
            noun_like=noun_like, genders=genders, selection_seed=variant_seed,
            description_relations=description_relations or {},
        )

    # The speaker is an agent, so it must be a person when we can tell.
    person_seeds = [word for word in seeds if _type_of(word) == "person"]
    person_characters = [word for word in available_characters if _type_of(word) != "place"]

    def _first_agent(*preferences: str) -> str:
        for candidate in preferences:
            if candidate and _type_of(candidate) != "place":
                return candidate
        return ""

    speaker = _first_agent(
        mentioned[-1] if mentioned else "",
        person_seeds[0] if person_seeds else "",
        seeds[0],
    ) or (person_characters[0] if person_characters else seeds[0])
    if dialogue or continuation:
        speaker = _first_agent(
            mentioned[-1] if mentioned else "",
            person_seeds[0] if person_seeds else "",
        ) or (person_characters[0] if person_characters else seeds[0])
    if introduce:
        unused = next((name for name in person_characters if name not in mentioned), speaker)
        speaker = unused

    # A place-typed topic is the scene's setting, not its protagonist.
    place_seeds = [word for word in seeds if _type_of(word) == "place"]
    location = place_seeds[0] if place_seeds else next((word for word in seeds if word in locations), "")
    if not location and mode == "scene" and _type_of(seeds[0]) != "person":
        location = seeds[0]
    goal = SceneGoal(
        topic=seeds[0], mode=mode, speaker=speaker, location=location,
        characters=tuple(dict.fromkeys((*mentioned, speaker)))[:3], seeds=tuple(seeds[:5]),
        active_field=tuple(word for word in field_seeds if word not in STOPWORDS)[:8],
    )
    sentence_count = max(3, sentence_count)
    beats = _scene_beats(mode, sentence_count)
    partner = next((word for word in context_words if word != speaker and not _is_action(word, phrase)), "")
    plans: list[SentencePlan] = []
    continuity_anchor = goal.topic
    used_fragments: set[tuple[str, ...]] = set()
    for index, purpose in enumerate(beats):
        focus = goal.location if index == 0 else goal.seeds[index % len(goal.seeds)]
        subject = _sentence_subject(goal, focus, purpose, _type_of)
        if mode == "dialogue" and partner:
            object_ = partner
        elif index == 0 and goal.location:
            object_ = goal.location
        else:
            object_ = continuity_anchor or goal.seeds[(index + 1) % len(goal.seeds)]
        relation = _relation_for_beat(purpose)
        planned_object = object_
        fragment = _choose_reasoned_fragment(
            graph, phrase, subject, (continuity_anchor, object_, *field_seeds),
            f"{variant_seed}|scene|{mode}|{index}",
            strict_subject=(mode == "dialogue"), used_fragments=used_fragments,
            genders=genders, noun_like=noun_like, verb_lexicon=verb_lexicon,
            speech_frames=speech_frames, proper_names=proper_names,
        )
        if fragment:
            subject, action, realized_object = _fragment_roles(fragment, phrase, noun_like)
            object_ = realized_object
            used_fragments.add(fragment)
        else:
            action = _choose_action(graph, phrase, subject, object_, f"scene|{mode}|{index}")
        detail_subject = _detail_subject(subject, phrase)
        detail_fragment = _choose_reasoned_fragment(
            graph, phrase, detail_subject, (*fragment, continuity_anchor, planned_object, *field_seeds),
            f"{variant_seed}|scene-detail|{mode}|{index}", strict_subject=bool(detail_subject), used_fragments=used_fragments,
            genders=genders, noun_like=noun_like, verb_lexicon=verb_lexicon,
            speech_frames=speech_frames, proper_names=proper_names,
        ) if fragment else ()
        if detail_fragment:
            used_fragments.add(detail_fragment)
        plans.append(SentencePlan(
            index=index, purpose=purpose, focus=focus, subject=subject, action=action,
            object=object_, speaker=goal.speaker, dialogue=(mode == "dialogue" and index % 2 == 0),
            continuity_anchor=continuity_anchor,
            transition=_transition_word(phrase, index, variant_seed),
            fragment=fragment,
            detail_fragment=detail_fragment,
            connector=_connector_for_relation(phrase, relation) if detail_fragment else "",
            relation=relation,
        ))
        continuity_anchor = _next_anchor(fragment, action, planned_object) or planned_object or focus
    return ScenePlan(goal=goal, beats=beats, sentences=tuple(plans))


# Russian declension spreads one name across up to six case forms
# ("Берлиоз"/"Берлиоза"/"Берлиозом"/...), but ``proper_names`` only contains
# the individual surface forms whose own mid-sentence-capitalized count
# cleared the ingest threshold — usually just the nominative. A question
# using any other case ("Что случилось с Берлиозом?") would otherwise fail
# the entity check even though the word is plainly the same character. Same
# stem-prefix principle ``_corpus_form`` below already uses for style-seed
# resolution, applied here to entity recognition instead.
_NAME_STEM_LEN = 5


def _name_stems(proper_names: frozenset[str]) -> dict[str, str]:
    stems: dict[str, str] = {}
    for name in sorted(proper_names):
        stem = name[:_NAME_STEM_LEN]
        current = stems.get(stem)
        if current is None:
            stems[stem] = name
            continue
        # Prefer the citation-like form over a frequent oblique case
        # ("шариков" over "шарика", "филиппович" over
        # "филипповича"). Ambiguous ties remain deterministic.
        oblique_endings = ("а", "я", "у", "ю", "е", "и", "ы", "ом", "ем", "ой", "ою")
        current_score = int(not current.endswith(oblique_endings))
        candidate_score = int(not name.endswith(oblique_endings))
        if candidate_score > current_score or (candidate_score == current_score and name < current):
            stems[stem] = name
    return stems


def _canonical_entity(word: str, proper_names: frozenset[str], name_stems: dict[str, str]) -> str | None:
    """Resolve any case form of a known name to its ingest-canonical spelling.

    Keeping the literal inflected token ("Берлиозом", instrumental) as a scene
    seed made the generator try to use a non-nominative form as a sentence
    subject, which reads as broken agreement. Returning the form the ingest
    lexicon actually recorded gives the rest of the pipeline a usable citation
    form instead.
    """

    if word in proper_names:
        return word
    if len(word) >= _NAME_STEM_LEN and word[:_NAME_STEM_LEN] in name_stems:
        return name_stems[word[:_NAME_STEM_LEN]]
    return None


def _looks_like_entity(word: str, proper_names: frozenset[str], name_stems: dict[str, str]) -> bool:
    return _canonical_entity(word, proper_names, name_stems) is not None


def _corpus_form(graph: ConceptGraph, wanted: str) -> str:
    """Resolve a prompt lemma to a corpus-observed inflected surface form."""

    if wanted in graph.weight:
        return wanted
    stem = wanted[:5]
    matches = [(word, weight) for word, weight in graph.weight.items() if word.startswith(stem)]
    return max(matches, key=lambda item: (item[1], item[0]))[0] if matches else ""


def _topic_field(
    graph: ConceptGraph, seeds: tuple[str, ...], phrase: PhraseModel, noun_like: frozenset[str],
    proper_names: frozenset[str], description_relations: dict[str, list[dict]],
) -> tuple[str, ...]:
    """Return a bounded, request-local concept field for descriptive reasoning.

    The first entries preserve the explicit request terms.  The remaining
    entries are ranked only by edges reached from those terms; corpus-wide word
    frequency is intentionally absent from this selection.
    """

    field = list(dict.fromkeys(word for word in seeds if word in graph.weight))
    # Do not expand a description through loose graph neighbourhoods.  The
    # graph remains the retrieval layer for query terms, but a surface fact
    # must be anchored in the request itself (or an explicit request seed),
    # otherwise a two-hop co-occurrence can turn a street into a doctor.
    return tuple(field)


def _description_subject(word: str, phrase: PhraseModel, noun_like: frozenset[str]) -> str:
    """Choose a corpus surface that can head a descriptive predicate.

    Prompt words may arrive in an oblique case (``комнату``).  We only swap to
    a sibling corpus form when that form actually has an observed predicate;
    this is morphology-aware grounding, not a handwritten entity dictionary.
    """

    def usable(candidate: str) -> bool:
        return any(is_finite_verb(next_word) for next_word in phrase.forward.get(candidate, {})) or any(
            is_finite_verb(previous) for previous in phrase.backward.get(candidate, {})
        )

    nominative = ""
    if word.endswith("у"):
        nominative = f"{word[:-1]}а"
    elif word.endswith("ю"):
        nominative = f"{word[:-1]}я"
    elif word.endswith("е"):
        nominative = f"{word[:-1]}а"
    if nominative in phrase.unigram and nominative in noun_like and usable(nominative):
        return nominative
    if usable(word):
        return word
    stem = word[:5]
    siblings = [
        (candidate, count) for candidate, count in phrase.unigram.items()
        if candidate.startswith(stem) and candidate in noun_like and usable(candidate)
    ]
    return max(siblings, key=lambda item: (item[1], item[0]))[0] if siblings else word


def _description_predicate(
    subject: str, phrase: PhraseModel, genders: dict[str, str] | None, seed: str, *, allow_inverted: bool = True,
) -> tuple[str, str, str]:
    """Recover an observed predicate around a topical word.

    Returns ``(predicate, detail, relation)``.  Predicate selection is limited
    to learned adjacent transitions and rejects plural/incorrect-gender past
    forms when the corpus supplied the subject gender.
    """

    gender = (genders or {}).get(subject, "")
    if not gender and subject in phrase.unigram:
        # This is a closed grammatical fallback, not an entity/action list.
        # It only protects agreement when corpus ingest lacked a gender count.
        gender = "f" if subject.endswith(("а", "я", "ь")) else "m"

    def accepted(action: str) -> bool:
        action_gender = past_gender(action)
        return bool(action_gender) and (not gender or action_gender == gender)

    # The opening can use an observed inverted predicate ("наступил вечер"),
    # which is useful for a place/time relation. Later descriptive beats must
    # have an explicit complement, avoiding bare event clauses in a static
    # description.
    backward = [
        ((action, "", "predicate_before_subject"), int(count))
        for action, count in phrase.backward.get(subject, {}).items()
        if accepted(action) and len(action) >= 4 and not action.endswith(("ся", "сь"))
    ]
    if allow_inverted and backward:
        encoded = [("\t".join(value), weight) for value, weight in backward]
        return tuple(seeded_weighted_pick(seed, f"description-backward:{subject}", encoded).split("\t"))  # type: ignore[return-value]

    forward: list[tuple[tuple[str, str, str], int]] = []
    for action, action_count in phrase.forward.get(subject, {}).items():
        if not accepted(action):
            continue
        details = [
            (detail, detail_count) for detail, detail_count in phrase.forward.get(action, {}).items()
            if detail not in STOPWORDS and detail not in _FRAGMENT_OBJECT_NOISE
            and detail != subject and not _is_semantic_action(detail, subject)
            and not is_infinitive(detail) and not detail.endswith(("я", "в", "вши", "вшись"))
        ]
        # A description may retain a state-like intransitive predicate or a
        # predicate followed by an adjectival complement.  A free SVO event
        # ("солнце кричал человеку") belongs to the event planner instead.
        details = [
            (detail, count) for detail, count in details
            if not detail.endswith(("ую", "юю"))
        ]
        is_reflexive = action.endswith(("ся", "сь"))
        if not is_reflexive or not details:
            continue
        if details:
            detail = seeded_weighted_pick(seed, f"description-detail:{subject}:{action}", details)
            forward.append(((action, detail, "property"), int(action_count)))
        else:
            forward.append(((action, "", "property"), int(action_count)))
    if forward:
        encoded = [("\t".join(value), weight) for value, weight in forward]
        return tuple(seeded_weighted_pick(seed, f"description-forward:{subject}", encoded).split("\t"))  # type: ignore[return-value]

    return "", "", "observation"


def _plan_description_scene(
    goal: SceneGoal, graph: ConceptGraph, phrase: PhraseModel, *, sentence_count: int,
    noun_like: frozenset[str], genders: dict[str, str] | None, selection_seed: str,
    description_relations: dict[str, list[dict]],
) -> ScenePlan:
    """Plan topic relations before speech realizes them as sentences."""

    total = max(3, sentence_count)
    candidates = [word for word in dict.fromkeys((goal.topic, *goal.active_field, *goal.seeds)) if word != goal.location]
    plans: list[SentencePlan] = []
    used: set[str] = set()
    used_subjects: set[str] = set()
    for index in range(total):
        alternatives = [word for word in candidates if word not in used] or [goal.topic]
        preferred, subject, action, detail, relation = alternatives[0], "", "", "", "observation"
        # If an activated neighbour has no descriptive fact, keep it in
        # SceneState but do not surface it as a vacuous one-word sentence.
        for candidate in alternatives:
            candidate_subject = _description_subject(candidate, phrase, noun_like)
            fact_subject = _description_fact_subject(candidate_subject, description_relations)
            if not fact_subject or fact_subject in used_subjects:
                continue
            candidate_action, candidate_detail, candidate_relation = _description_fact(
                fact_subject, description_relations, f"{selection_seed}|description|{index}|{candidate}",
                allow_inverted=(index == 0),
            )
            if candidate_action or candidate_relation == "epithet":
                preferred, subject, action, detail, relation = (
                    candidate, fact_subject, candidate_action, candidate_detail, candidate_relation,
                )
                break
        if not subject:
            subject = _description_subject(preferred, phrase, noun_like)
        # A geographic context belongs to the relation, never to an agent
        # slot.  ``В Москве наступил вечер`` is a relation between a local
        # field concept and a place, not an action attributed to Moscow.
        object_ = detail
        if index == 0 and goal.location and subject != goal.location and action and relation == "predicate_before_subject":
            object_ = goal.location
            relation = "located_at"
        elif index == 0 and goal.location and subject != goal.location and action and detail and relation == "property":
            object_ = goal.location
            relation = "located_property"
        # Bundle compatible facts about the same subject into one sentence:
        # the primary relation may carry an agreeing observed epithet, and a
        # property/epithet clause may carry one observed object link.  A
        # located or placed relation already holds a preposition phrase, so
        # it takes no second one.
        epithet = "" if relation == "epithet" else _description_epithet(
            subject, description_relations, f"{selection_seed}|description-epithet|{index}",
            exclude=frozenset((detail, subject)),
        )
        link = _description_link(
            subject, description_relations, f"{selection_seed}|description-link|{index}",
        ) if relation in {"property", "epithet"} else ()
        purpose = ("establish_field", "develop_field", "close_field")[min(index, 2)]
        plans.append(SentencePlan(
            index=index, purpose=purpose, focus=preferred, subject=subject,
            action=action, object=object_, speaker="", dialogue=False,
            continuity_anchor=goal.topic, relation=relation,
            detail_fragment=(detail,) if relation == "located_property" and detail else (),
            epithet=epithet, link=link,
        ))
        used.add(preferred)
        used_subjects.add(subject)
    return ScenePlan(goal=goal, beats=tuple(item.purpose for item in plans), sentences=tuple(plans))


def _description_fact_subject(surface: str, description_relations: dict[str, list[dict]]) -> str:
    """Resolve an oblique prompt surface only through explicit case variants."""

    if surface in description_relations:
        return surface
    variants: list[str] = []
    if surface.endswith("у") and len(surface) >= 4:
        variants.extend((f"{surface[:-1]}а", surface[:-1]))
    elif surface.endswith("ю") and len(surface) >= 4:
        variants.extend((f"{surface[:-1]}я", surface[:-1]))
    elif surface.endswith("е") and len(surface) >= 4:
        variants.append(f"{surface[:-1]}а")
    elif surface.endswith("и") and len(surface) >= 4:
        variants.append(f"{surface[:-1]}а")
    return next((variant for variant in variants if variant in description_relations), "")


def _description_fact(
    subject: str, description_relations: dict[str, list[dict]], seed: str, *, allow_inverted: bool,
) -> tuple[str, str, str]:
    """Select a sentence-local descriptive fact; never traverse loose edges."""

    candidates = [
        row for row in description_relations.get(subject, ())
        # A subject-to-object link is a noun-phrase extension, never a
        # sentence's primary fact; the bundle step attaches it separately.
        if row.get("kind") != "object_link"
        and (allow_inverted or row.get("kind") != "predicate_before_subject")
    ]
    inverted = [row for row in candidates if row.get("kind") == "predicate_before_subject"]
    if allow_inverted and inverted:
        candidates = inverted
    elif properties := [row for row in candidates if row.get("kind") == "property"]:
        candidates = properties
    elif events := [
        row for row in candidates
        if row.get("kind") in {"event_place", "event_place_inverted"}
    ]:
        candidates = events
    elif epithets := [row for row in candidates if row.get("kind") == "epithet"]:
        candidates = epithets
    if not candidates:
        return "", "", "observation"
    encoded = [
        ("\t".join((str(row.get("predicate", "")), str(row.get("detail", "")), str(row.get("kind", "property")))),
        max(1, int(row.get("weight", 1))))
        for row in candidates
    ]
    action, detail, relation = seeded_weighted_pick(seed, f"description-fact:{subject}", encoded).split("\t")
    return action, detail, relation


def _description_epithet(
    subject: str, description_relations: dict[str, list[dict]], seed: str, *, exclude: frozenset[str],
) -> str:
    """Pick an observed agreeing epithet to enrich the subject noun phrase."""

    rows = [
        row for row in description_relations.get(subject, ())
        if row.get("kind") == "epithet" and row.get("detail") and row["detail"] not in exclude
    ]
    if not rows:
        return ""
    encoded = [(str(row["detail"]), max(1, int(row.get("weight", 1)))) for row in rows]
    return seeded_weighted_pick(seed, f"description-epithet:{subject}", encoded)


def _description_link(
    subject: str, description_relations: dict[str, list[dict]], seed: str,
) -> tuple[str, ...]:
    """Pick an observed subject-to-object preposition link ("в трактире")."""

    rows = [row for row in description_relations.get(subject, ()) if row.get("kind") == "object_link"]
    if not rows:
        return ()
    encoded = [
        ("\t".join((str(row.get("predicate", "")), str(row.get("detail", "")))),
         max(1, int(row.get("weight", 1))))
        for row in rows
    ]
    return tuple(seeded_weighted_pick(seed, f"description-link:{subject}", encoded).split("\t"))


def _locative_surface(location: str, phrase: PhraseModel) -> str:
    """Use an observed locative surface where the corpus provides one."""

    if location.endswith(("е", "и")):
        return location
    stem = location[:5]
    variants = [
        (word, count) for word, count in phrase.unigram.items()
        if word.startswith(stem) and word.endswith(("е", "и"))
    ]
    return max(variants, key=lambda item: (item[1], item[0]))[0] if variants else location


def _sentence_purpose(index: int, total: int, mode: str) -> str:
    if index == 0:
        return "establish_scene"
    if index == total - 1:
        return "close_scene"
    if mode == "dialogue":
        return "dialogue_turn"
    if mode == "introduction":
        return "reveal_character"
    return "continue_action"


def _scene_beats(mode: str, sentence_count: int) -> tuple[str, ...]:
    """Explicit paragraph-scale reasoning: context, development, consequence."""

    beats = [_sentence_purpose(index, sentence_count, mode) for index in range(sentence_count)]
    if sentence_count >= 3 and mode not in {"dialogue", "introduction"}:
        beats[1] = "develop_action"
        beats[-1] = "show_consequence"
    return tuple(beats)


def _sentence_subject(goal: SceneGoal, focus: str, purpose: str, type_of=None) -> str:
    if goal.mode == "dialogue" or purpose in {"develop_action", "show_consequence", "reveal_character"}:
        return goal.speaker
    # A place cannot be the agent of an action. When the establishing beat's
    # focus is the setting, the scene's person acts *in* it instead. Without a
    # type (``type_of`` absent, or "unknown") this is the previous behaviour.
    if type_of is not None and type_of(focus) == "place":
        return goal.speaker
    return focus


def _relation_for_beat(purpose: str) -> str:
    return {
        "establish_scene": "establish",
        "dialogue_turn": "respond",
        "develop_action": "develop",
        "show_consequence": "consequence",
        "reveal_character": "introduce",
        "close_scene": "close",
    }.get(purpose, "continue")


def _choose_reasoned_fragment(
    graph: ConceptGraph, phrase: PhraseModel, subject: str, anchors: tuple[str, ...], seed: str,
    *, strict_subject: bool = False, used_fragments: set[tuple[str, ...]] | None = None,
    genders: dict[str, str] | None = None, noun_like: frozenset[str] = frozenset(),
    verb_lexicon: dict | None = None,
    speech_frames: tuple[dict, ...] = (),
    proper_names: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Choose a locally grammatical event fragment before realization.

    A candidate is one learned trigram containing an action.  It is scored for
    the planned subject and for direct graph contact with the active narrative
    anchors.  The next sentence is planned from the selected fragment's last
    meaningful token, so event order is an explicit reasoning decision rather
    than a renderer-side connective.
    """

    anchor_set = {word for word in anchors if word}
    # World-state reasoning needs an event the surface can actually assert:
    # do not borrow a trigram whose grammatical subject is merely a nearby
    # graph node. That was the source of disconnected character jumps.
    starts = [subject]
    candidates: dict[tuple[str, ...], int] = {}
    for first in starts:
        for second, first_count in phrase.forward.get(first, {}).items():
            pair = (first, second)
            trusted_triple = False
            for third, third_count in phrase.forward2.get((first, second), {}).items():
                fragment = (first, second, third)
                if fragment in (used_fragments or set()):
                    continue
                proper_stems = {name[:_NAME_STEM_LEN] for name in proper_names}
                if (
                    (third in proper_names or any(third.startswith(stem) for stem in proper_stems))
                    and third not in anchor_set
                ):
                    continue
                if not _trusted_svo(fragment, subject, noun_like, verb_lexicon):
                    continue
                if not _past_event_fragment(fragment, noun_like):
                    continue
                # Morphological agreement gate. A fragment whose past-tense verb
                # contradicts the planned subject's learned gender ("Маргарита
                # погубил") is rejected here, before it can ever be a candidate.
                # Unknown gender or a non-past verb imposes no constraint.
                if genders and sentence_agreement_errors(first, list(fragment), genders, noun_like):
                    continue
                trusted_triple = True
                content = set(content_words(" ".join(fragment)))
                relevance = 12 if first == subject else 0
                relevance += 5 * len(content & anchor_set)
                if not relevance:
                    for word in content:
                        if any(neighbour in anchor_set for neighbour, _weight in graph.neighbors(word)[:8]):
                            relevance += 2
                            break
                if not relevance:
                    continue
                candidates[fragment] = max(
                    candidates.get(fragment, 0), relevance * 100 + min(first_count, 20) + min(third_count, 20)
                )
            # A finite intransitive edge is already a complete event
            # ("Воланд остановился"). Prefer a trusted SVO when one exists;
            # otherwise keep the two-token walk so the speech graph does not
            # collapse onto the handful of transitive fragments.
            if (
                not trusted_triple
                and pair not in (used_fragments or set())
                and _is_semantic_action(second, subject, noun_like)
                and past_gender(second) is not None
                and second.endswith(("ся", "сь"))
                and not (genders and sentence_agreement_errors(first, [*pair], genders, noun_like))
            ):
                relevance = 12 + 5 * int(second in anchor_set)
                candidates[pair] = max(candidates.get(pair, 0), relevance * 100 + min(first_count, 20))
    # Universal frames are learned from the same corpus but observed with at
    # least two different subjects. They let a new character reuse a proven
    # speech/action relation without copying that character's exact trigram.
    # Keep them bounded and below exact subject evidence, while still allowing
    # them to win when the exact subject has only a few safe edges.
    if len(candidates) < 12:
        for frame in speech_frames[:48]:
            action = str(frame.get("action", ""))
            object_ = str(frame.get("object", ""))
            fragment = (subject, action, object_)
            if fragment in (used_fragments or set()) or not _trusted_svo(fragment, subject, noun_like, verb_lexicon):
                continue
            if past_gender(action) is None:
                continue
            content = {action, object_}
            relevance = 8 + 5 * len(content & anchor_set)
            if not (content & anchor_set):
                for word in content:
                    if any(neighbour in anchor_set for neighbour, _weight in graph.neighbors(word)[:8]):
                        relevance += 2
                        break
            candidates[fragment] = max(
                candidates.get(fragment, 0),
                relevance * 100 + min(int(frame.get("weight", 1)), 20),
            )
            if len(candidates) >= 12:
                break
    if not candidates:
        return ()
    encoded = [("\t".join(fragment), weight) for fragment, weight in candidates.items()]
    return tuple(seeded_weighted_pick(seed, "reasoned_fragment", encoded).split("\t"))


def _past_event_fragment(fragment: tuple[str, ...], noun_like: frozenset[str]) -> bool:
    """Narrative paragraphs keep one past-tense temporal frame by default."""
    if len(fragment) < 2:
        return False
    action = fragment[1] if _is_semantic_action(fragment[1], fragment[0], noun_like) else fragment[-1]
    return past_gender(action) is not None


def _valid_fragment_object(word: str, noun_like: frozenset[str] = frozenset()) -> bool:
    return (
        len(word) >= 2 and word not in STOPWORDS and word not in _FRAGMENT_OBJECT_NOISE
        and word not in _INTERROGATIVE_MARKERS
        and not _ADJ_END.search(word) and not _SHORT_ADJ_END.search(word)
        and not _is_semantic_action(word, noun_like)
        and not is_infinitive(word)
        and not word.endswith(("ите", "йте", "в", "вши", "вшись"))
        and (word in noun_like or not word.endswith(("о", "е")))
    )


def _trusted_svo(fragment: tuple[str, ...], subject: str, noun_like: frozenset[str] = frozenset(), verb_lexicon: dict | None = None) -> bool:
    if len(fragment) == 2:
        first, action = fragment
        if first != subject or set(fragment) & _FIRST_PERSON:
            return False
        return _is_semantic_action(action, subject, noun_like)
    if len(fragment) != 3:
        return False
    first, action, object_ = fragment
    if first != subject or set(fragment) & _FIRST_PERSON:
        return False
    if _is_semantic_action(action, subject, noun_like):
        if _valid_fragment_object(object_, noun_like) and object_ != subject:
            dative = set((verb_lexicon or {}).get("dative_actions", ()))
            if action in dative and not object_.endswith(("у", "ю", "ому", "ему", "ам", "ям")):
                return False
            return True
        return False
    # A learned S–adverb–verb span is also a complete assertion ("Иван тихо
    # сказал"), but it does not manufacture an object fact.
    return (
        action not in STOPWORDS
        and not _ADJ_END.search(action)
        and action not in _INTERROGATIVE_MARKERS
        and _is_semantic_action(object_, subject, noun_like)
        and (
            object_ in set((verb_lexicon or {}).get("standalone_actions", ()))
            and object_ not in set((verb_lexicon or {}).get("transitive_actions", ()))
            if verb_lexicon is not None else object_ not in noun_like
        )
    )


def _fragment_roles(fragment: tuple[str, ...], phrase: PhraseModel, noun_like: frozenset[str] = frozenset(), verb_lexicon: dict | None = None) -> tuple[str, str, str]:
    if len(fragment) == 2 and _trusted_svo(fragment, fragment[0], noun_like, verb_lexicon):
        return fragment[0], fragment[1], ""
    if _trusted_svo(fragment, fragment[0], noun_like, verb_lexicon) and _is_semantic_action(fragment[1], fragment[0], noun_like):
        return fragment[0], fragment[1], fragment[2]
    return fragment[0], fragment[2], ""


def _next_anchor(fragment: tuple[str, ...], action: str, fallback: str) -> str:
    if not fragment:
        return fallback
    # SVO fragment: its final token is the newly established object.  For a
    # subject + bridge + action fragment ("иван не ответил"), there is no new
    # object, so retain the semantic object selected by the scene planner.
    if len(fragment) == 3 and fragment[1] == action and fragment[2] not in STOPWORDS:
        return fragment[2]
    return fallback


def _transition_word(phrase: PhraseModel, index: int, seed: str = "") -> str:
    # One overt bridge establishes the progression. Repeating a discourse
    # marker before every sentence makes a deterministic paragraph read like a
    # template; later coherence comes from the carried anchor and salience.
    if index == 0:
        return ""
    preferred = ("затем", "потом", "тогда", "позже", "вскоре")
    available = [(word, max(1, phrase.unigram.get(word, 1))) for word in preferred if word in phrase.unigram]
    if not available:
        return ""
    return seeded_weighted_pick(seed or "default", f"transition:{index}", available)


def _connector_for_relation(phrase: PhraseModel, relation: str) -> str:
    preferred = {
        "establish": ("и", "а"),
        "respond": ("но", "и"),
        "develop": ("и", "а"),
        "consequence": ("а", "и"),
        "close": ("и", "но"),
        "introduce": ("и", "а"),
    }.get(relation, ("и",))
    return next((word for word in preferred if word in phrase.unigram), "")


def _detail_subject(subject: str, phrase: PhraseModel) -> str:
    """Use a corpus-supported masculine pronoun for a repeated named subject.

    This is deliberately conservative: names ending in a vowel are left alone
    because gender cannot be recovered safely from this lightweight runtime.
    The semantic speaker remains in ``SceneGoal``; only the second surface
    clause receives the resolved reference.
    """

    if subject and subject[-1] not in "ая" and "он" in phrase.forward:
        return "он"
    return subject


class NarrativeGenerator:
    """Sentence renderer: the poetry generator without meter/rhyme/stanzas."""

    def __init__(
        self, phrase: PhraseModel, graph: ConceptGraph, proper_names: frozenset[str],
        genders: dict[str, str] | None = None, noun_like: frozenset[str] = frozenset(),
        reasoning: bool = False, verb_lexicon: dict | None = None,
    ) -> None:
        self.genders = genders or {}
        self.noun_like = noun_like
        self.phrase = phrase
        self.graph = graph
        self.proper_names = proper_names
        self.reasoning = reasoning
        self.verb_lexicon = verb_lexicon

    def render(self, plan: ScenePlan, *, seed: str) -> Paragraph:
        state = DiscourseState.seeded(list(plan.goal.seeds))
        paragraph = Paragraph()
        allowed_names = set(plan.goal.characters)
        for sentence in plan.sentences:
            if plan.goal.mode == "description":
                # An ungrounded field term remains visible in SceneState but
                # is not padded into prose as a one-word pseudo-sentence.
                if not sentence.action and sentence.relation != "epithet":
                    continue
                text, trace = self._render_description_sentence(sentence)
            else:
                text, trace = self._render_sentence(
                    sentence, state, allowed_names, paragraph.sentences, f"{seed}|{sentence.index}"
                )
            paragraph.sentences.append(text)
            paragraph.realization.append(trace)
            state.update(text)
        return paragraph

    def _render_description_sentence(self, plan: SentencePlan) -> tuple[str, SentenceRealization]:
        """Realize a planned topical relation without inventing an actor/event."""

        epithet = [plan.epithet] if plan.epithet else []
        link = list(plan.link)
        if plan.relation == "epithet" and plan.object:
            tokens = [plan.object, plan.subject, *link]
        elif plan.action and plan.relation == "located_at" and plan.object:
            location = _locative_surface(plan.object, self.phrase)
            tokens = ["в", location, plan.action, *epithet, plan.subject]
        elif plan.action and plan.relation == "located_property" and plan.object:
            location = _locative_surface(plan.object, self.phrase)
            detail = plan.detail_fragment[0] if plan.detail_fragment else ""
            tokens = ["в", location, plan.action, detail, *epithet, plan.subject]
        elif plan.action and plan.relation == "predicate_before_subject":
            tokens = [plan.action, *epithet, plan.subject]
        elif plan.action and plan.relation == "event_place" and plan.object:
            tokens = [*epithet, plan.subject, plan.action, *plan.object.split()]
        elif plan.action and plan.relation == "event_place_inverted" and plan.object:
            tokens = [*plan.object.split(), plan.action, *epithet, plan.subject]
        elif plan.action:
            tokens = [*epithet, plan.subject, *link, plan.action]
            if plan.object:
                tokens.append(plan.object)
        else:
            # The reasoning layer deliberately produced no unsupported
            # relation.  Retain the topic as an observation rather than
            # fabricating psychology or assigning an action to a place.
            tokens = [plan.subject]
        body = _sentence_text([token for token in tokens if token], False, "", self.proper_names)
        actual = set(words(body))
        realized = tuple(
            role for role, word in (("subject", plan.subject), ("action", plan.action), ("object", plan.object))
            if word and word in actual
        )
        return body, SentenceRealization(
            plan.index, (plan.subject, plan.action, plan.object), realized, False, ()
        )

    def _render_sentence(
        self, plan: SentencePlan, state: DiscourseState, allowed_names: set[str], prior_sentences: list[str], seed: str
    ) -> tuple[str, SentenceRealization]:
        if self.reasoning:
            return self._render_reasoned_sentence(plan)
        candidates: list[list[str]] = []
        fallback: list[str] | None = None
        slots = tuple(word for word in (plan.subject, plan.action, plan.object) if word in self.phrase.unigram)
        if plan.fragment and _trusted_svo(plan.fragment, plan.subject, self.noun_like, self.verb_lexicon) and check_line(self.phrase, " ".join(plan.fragment)):
            candidates.append(list(plan.fragment))
        compound = tuple((*plan.fragment, plan.connector, *plan.detail_fragment)) if plan.connector else ()
        if compound and _trusted_svo(plan.fragment, plan.subject, self.noun_like, self.verb_lexicon) and check_line(self.phrase, " ".join(compound)):
            candidates.append(list(compound))
        for attempt in range(8):
            attempt_seed = f"{seed}#{attempt}"
            # Keep the planned subject as the syntactic anchor. Varying the
            # starting role made longer paragraphs less coherent by allowing a
            # location or object to displace the stable scene speaker.
            start = plan.subject if plan.subject in self.phrase.forward else plan.focus
            if start not in self.phrase.unigram:
                start = seeded_weighted_pick(
                    attempt_seed, "narrative_opener", [(word, count) for word, count in self.phrase.openers.items()]
                )
            tokens = self.phrase.grow_forward_slots(
                start, 999, attempt_seed, slots=slots[1:] if slots and start == slots[0] else slots,
                max_words=plan.target_words, avoid_seen_4grams=True,
            )
            completed = _complete_tokens(tokens)
            if completed and fallback is None:
                fallback = completed
            if completed and _realizes_planned_event(completed, plan) and check_line(self.phrase, " ".join(completed)):
                candidates.append(completed)
        event_fallback = _event_fallback(plan, self.noun_like)
        if event_fallback:
            candidates.append(event_fallback)
        chosen = self._select(
            candidates or ([event_fallback] if event_fallback else [[]]), plan, state, allowed_names, prior_sentences
        )
        if not chosen:
            chosen = _pause_fallback(plan)
        body = _sentence_text(
            chosen, plan.dialogue, plan.transition, self.proper_names,
            clause_break=len(plan.fragment) if compound and tuple(chosen) == compound else 0,
        )
        actual = set(words(body))
        realized = tuple(role for role, word in (("subject", plan.subject), ("action", plan.action), ("object", plan.object)) if word and word in actual)
        realized_facts = ()
        if {"subject", "action"} <= set(realized) and _is_semantic_action(plan.action, plan.subject):
            realized_facts = ((plan.subject, plan.action or "acts", plan.object or "scene"),)
        return body, SentenceRealization(
            plan.index, (plan.subject, plan.action, plan.object), realized, plan.dialogue, realized_facts
        )

    def _render_reasoned_sentence(self, plan: SentencePlan) -> tuple[str, SentenceRealization]:
        fragments = tuple(plan.clause_fragments) or (
            (plan.fragment,)
            if _trusted_svo(plan.fragment, plan.subject, self.noun_like, self.verb_lexicon)
            else ()
        )
        if fragments:
            tokens: list[str] = []
            for clause_index, fragment in enumerate(fragments):
                clause_tokens = list(fragment)
                same_subject = bool(clause_tokens) and clause_tokens[0] == plan.subject
                if clause_index > 0 and same_subject:
                    clause_tokens = clause_tokens[1:]
                elif clause_index > 0 and clause_tokens:
                    clause_tokens[0] = "она" if clause_tokens[0].endswith(("а", "я")) else "он"
                if clause_index:
                    connector = plan.clause_connectors[clause_index - 1] if clause_index - 1 < len(plan.clause_connectors) else "и"
                    tokens.append(connector)
                tokens.extend(clause_tokens)
            if plan.index > 0 and tokens and tokens[0] == plan.subject:
                tokens[0] = "она" if plan.subject.endswith(("а", "я")) else "он"
            transition = plan.transition
            body = _sentence_text(tokens, plan.dialogue, transition, self.proper_names)
            if " но " in body:
                body = body.replace(" но ", ", но ", 1)
            if " а затем " in body:
                body = body.replace(" а затем ", ", а затем ", 1)
        else:
            # Unknown or rejected actions remain a non-event pause. They do
            # not get smuggled back into the paragraph as psychology.
            tokens = _pause_fallback(plan)
            transition = plan.transition
            if plan.index == 0 and "сначала" in self.phrase.unigram:
                transition = "сначала"
            body = _sentence_text(tokens, plan.dialogue, transition, self.proper_names)
        actual = set(words(body))
        realized = tuple(
            role for role, word in (("subject", plan.subject), ("action", plan.action), ("object", plan.object))
            if word and word in actual
        )
        if plan.index > 0 and plan.subject not in actual and "subject" not in realized:
            realized = ("subject", *realized)
        realized_facts = tuple(
            (
                fragment[0],
                _fragment_roles(fragment, self.phrase, self.noun_like, self.verb_lexicon)[1],
                _fragment_roles(fragment, self.phrase, self.noun_like, self.verb_lexicon)[2] or "scene",
            )
            for fragment in fragments
            if len(fragment) >= 2 and _trusted_svo(fragment, fragment[0], self.noun_like, self.verb_lexicon)
        )
        return body, SentenceRealization(
            plan.index, (plan.subject, plan.action, plan.object), realized, plan.dialogue, realized_facts
        )

    def _select(
        self, candidates: list[list[str]], plan: SentencePlan, state: DiscourseState, allowed_names: set[str],
        prior_sentences: list[str],
    ) -> list[str]:
        safe = [
            tokens for tokens in candidates
            if not ((set(content_words(" ".join(tokens))) & self.proper_names) - allowed_names)
        ]
        if safe:
            candidates = safe
        field_ = state.salience_field(self.graph)
        best = candidates[0]
        best_score: float | None = None
        for tokens in candidates:
            text = " ".join(tokens)
            score, _parts = line_salience(text, state, self.graph, plan.focus, field_=field_)
            actual = set(words(text))
            for word, points in ((plan.subject, 3), (plan.action, 4), (plan.object, 2)):
                if word and word in actual:
                    score += points
            if plan.continuity_anchor:
                score += 5 if plan.continuity_anchor in actual else -3
            if plan.fragment and tuple(tokens) == plan.fragment:
                score += 8
            if plan.detail_fragment and plan.connector and tuple(tokens) == tuple((*plan.fragment, plan.connector, *plan.detail_fragment)):
                score += 12
            for name in set(content_words(text)) & self.proper_names:
                if name not in allowed_names:
                    score -= 8
            score += min(len(tokens), plan.target_words) / plan.target_words
            if len(tokens) < plan.minimum_words:
                score -= 4 * (plan.minimum_words - len(tokens))
            if tuple(words(text)) in {tuple(words(item)) for item in prior_sentences}:
                score -= 10
            # --- morphology layer ------------------------------------------
            # Gender disagreement is the most visible defect the entity-type
            # transfer could not touch ("Маргарита погубил"). It is scored, not
            # hard-rejected, because a fully-agreeing candidate may not exist:
            # a wrong-gender sentence still beats an empty one.
            errors = sentence_agreement_errors(plan.subject, tokens, self.genders, self.noun_like)
            score -= 10 * errors
            # A clause needs a predicate. "Пилат и всеобщий" has none, and
            # "Маргарита шелестеть листами" has only an infinitive, which cannot
            # head one.
            if not any(is_finite_verb(token) for token in tokens):
                score -= 6
            if len(tokens) >= 2 and is_infinitive(tokens[1]):
                score -= 4
            if best_score is None or score > best_score:
                best_score, best = score, tokens
        return best


class NarrativeEngine:
    def __init__(self, artifact_path: Path | None = None) -> None:
        source = artifact_path or (Path(__file__).resolve().parents[1] / "artifacts" / "narrative_model.json")
        data = load_artifacts(source)
        self.phrase = PhraseModel.from_dict(data["phrase_model"])
        self.graph = ConceptGraph.from_dict(data["concept_graph"])
        self.meta = data["meta"]
        self.proper_names = frozenset(data.get("proper_names", ()))
        # Ported QA knowledge layer: advisory entity types, never a gate.
        self.entity_types = dict(data.get("entity_types", {}))
        # Morphology layer: corpus-learned gender + the noun evidence that
        # keeps "стол" from being mistaken for a masculine past verb.
        self.gender_map = dict(data.get("gender_map", {}))
        self.noun_like = frozenset(data.get("noun_like", ()))
        self.verb_lexicon = dict(data.get("verb_lexicon", {}))
        self.speech_frames = tuple(data.get("universal_speech_frames", ()))
        self.description_relations = dict(data.get("description_relations", {}))
        self._recent_outputs: dict[tuple[str, str, int, bool], list[str]] = {}

    def run(
        self, prompt: str, *, context: str = "", sentences: int = 3,
        seed: str | None = None, reasoning: bool = False,
    ) -> NarrativeResult:
        request = NarrativeRequest(prompt=prompt, context=context, sentences=sentences)
        if seed is not None:
            return self._run_once(request, render_seed=seed, reasoning=reasoning)

        # A new random route normally diverges by itself. Retain a compact
        # per-engine history too: if the beam nevertheless lands on the same
        # paragraph for the same request, sample another valid route.
        key = (prompt, context, sentences, reasoning)
        seen = self._recent_outputs.setdefault(key, [])
        fallback: NarrativeResult | None = None
        for _attempt in range(8):
            result = self._run_once(request, render_seed=secrets.token_hex(8), reasoning=reasoning)
            fallback = result
            text = result.paragraph.text()
            if text not in seen:
                seen.append(text)
                del seen[:-16]
                return result
        assert fallback is not None
        return fallback

    def _run_once(
        self, request: NarrativeRequest, *, render_seed: str, reasoning: bool,
    ) -> NarrativeResult:
        plan = plan_scene(
            self.graph, self.phrase, prompt=request.prompt, context=request.context, meta=self.meta,
            proper_names=self.proper_names, sentence_count=request.sentences,
            entity_types=self.entity_types, genders=self.gender_map, noun_like=self.noun_like,
            verb_lexicon=self.verb_lexicon,
            speech_frames=self.speech_frames,
            description_relations=self.description_relations,
            selection_seed=render_seed,
        )
        reasoning_plan: ReasoningPlan | None = None
        initial_state: WorldState | None = None
        scene_state = _scene_state(plan)
        if reasoning:
            # Event search operates over assertions in WorldState. A topical
            # description is instead a relation plan over SceneState, so
            # pushing it through the event beam would turn a place back into
            # an actor. The renderer receives only its selected relations.
            if plan.goal.mode != "description":
                initial_state = _initial_world_state(plan)
                candidates = tuple(
                    _sentence_candidates(
                        sentence, continuity_subject=plan.goal.speaker,
                        noun_like=self.noun_like, verb_lexicon=self.verb_lexicon,
                    )
                    for sentence in plan.sentences
                )
                reasoning_plan = beam_search(
                    initial_state, candidates, goal_terms=frozenset(plan.goal.seeds), beam_width=4,
                )
                selected = tuple(step.payload for step in reasoning_plan.steps if isinstance(step.payload, SentencePlan))
                plan = _compress_scene_plan(
                    ScenePlan(plan.goal, tuple(sentence.purpose for sentence in selected), selected),
                    noun_like=self.noun_like, verb_lexicon=self.verb_lexicon,
                )
        paragraph = NarrativeGenerator(
            self.phrase, self.graph, self.proper_names, self.gender_map, self.noun_like,
            reasoning=reasoning, verb_lexicon=self.verb_lexicon,
        ).render(plan, seed=render_seed)
        committed_state = _commit_realized_facts(initial_state, paragraph) if initial_state else None
        return NarrativeResult(
            request, plan, paragraph, assess_poem(self.phrase, paragraph.sentences), self.meta,
            reasoning_plan=reasoning_plan, world_state=committed_state, scene_state=scene_state,
        )


def _scene_state(plan: ScenePlan) -> SceneState:
    """Materialize the plan-layer state that guided the current paragraph."""

    if plan.goal.mode == "description":
        return SceneState(
            dominant_tone="observational",
            pacing="slow",
            narrative_focus=plan.goal.topic,
            emotional_trajectory=("orientation", "elaboration", "closure"),
            active_conflict="",
            scene_objective=plan.goal.scene_objective,
            active_field=plan.goal.active_field,
        )
    return SceneState(
        dominant_tone="dramatic" if plan.goal.mode == "dialogue" else "narrative",
        pacing="progressive",
        narrative_focus=plan.goal.speaker or plan.goal.topic,
        emotional_trajectory=("setup", "development", "consequence"),
        active_conflict="dialogue" if plan.goal.mode == "dialogue" else "",
        scene_objective=plan.goal.scene_objective,
        active_field=plan.goal.active_field,
    )


def _initial_world_state(plan: ScenePlan) -> WorldState:
    names = tuple(dict.fromkeys((*plan.goal.characters, plan.goal.speaker, plan.goal.topic)))
    facts = [StateFact(name, "introduced", "scene", 0) for name in names if name]
    if plan.goal.location and plan.goal.speaker:
        facts.append(StateFact(plan.goal.speaker, "located_at", plan.goal.location, 0))
    return WorldState.from_initial_facts(facts)


def _compress_scene_plan(
    plan: ScenePlan, *, noun_like: frozenset[str], verb_lexicon: dict | None = None,
    max_clauses: int = 2,
) -> ScenePlan:
    """Turn a chain of atomic events into a few readable coordinated clauses.

    Planning still happens event by event. Compression is a sentence-plan
    operation: only trusted realized fragments are grouped, while idle
    fallback beats after the opening are dropped from the surface. This keeps
    every committed fact inspectable without printing one sentence per edge.
    """
    output: list[SentencePlan] = []
    group: list[SentencePlan] = []
    setup_event_emitted = False

    def flush() -> None:
        if not group:
            return
        first = group[0]
        if len(group) == 1:
            output.append(first)
        else:
            output.append(replace(
                first,
                clause_fragments=tuple(item.fragment for item in group),
                clause_connectors=tuple(
                    "но" if item.relation == "consequence"
                    else "а затем" if position % 2 == 0
                    else "и"
                    for position, item in enumerate(group[1:], start=1)
                ),
                detail_fragment=(), connector="",
            ))
        group.clear()

    for sentence in plan.sentences:
        trusted = bool(sentence.fragment) and _trusted_svo(
            sentence.fragment, sentence.subject, noun_like, verb_lexicon
        )
        if not trusted:
            flush()
            # Keep the opening state; later idle beats are not narrative
            # progress and only create the old молчал/ждал loop.
            if not output:
                output.append(sentence)
            continue
        action = _fragment_roles(sentence.fragment, PhraseModel(), noun_like, verb_lexicon)[1]
        group_actions = {
            _fragment_roles(item.fragment, PhraseModel(), noun_like, verb_lexicon)[1]
            for item in group
        }
        if action in group_actions:
            continue
        # After an opening state, keep the first concrete action on its own.
        # The following actions can then form a consequence pair instead of a
        # monotonous list of identical short sentences.
        if output and not setup_event_emitted and not group:
            group.append(sentence)
            flush()
            setup_event_emitted = True
            continue
        group.append(sentence)
        if len(group) >= max_clauses:
            flush()
    flush()
    if not output:
        output = list(plan.sentences[:1])
    reindexed = tuple(replace(sentence, index=index) for index, sentence in enumerate(output))
    return ScenePlan(
        plan.goal, tuple(sentence.purpose for sentence in reindexed), reindexed
    )


def _commit_realized_facts(initial_state: WorldState, paragraph: Paragraph) -> WorldState:
    """Advance public discourse state from rendered facts, never planned ones."""
    state = initial_state
    for trace in paragraph.realization:
        assertions = tuple(StateFact(subject, predicate, object_, 0) for subject, predicate, object_ in trace.realized_facts)
        if not assertions:
            continue
        result = state.apply(StateDelta(assertions=assertions, label=f"realized_sentence_{trace.index}"))
        if result.accepted:
            state = result.state
    return state


def _reasoned_sentence(sentence: SentencePlan, noun_like: frozenset[str] = frozenset(), verb_lexicon: dict | None = None) -> SentencePlan | None:
    """Use only a recoverable finite verb as a narrative event predicate.

    The earlier local planner can label a subject as its own action when a
    trigram is fragmentary.  That is acceptable as a surface fallback but not
    as a world fact, so the reasoning bridge recovers a finite verb from the
    selected fragment/detail or drops the step before search.
    """
    if not _trusted_svo(sentence.fragment, sentence.subject, noun_like, verb_lexicon):
        return None
    _subject, action, object_ = _fragment_roles(
        sentence.fragment, PhraseModel(), noun_like, verb_lexicon
    )
    return replace(sentence, action=action, object=object_, detail_fragment=(), connector="")


def _is_semantic_action(token: str, subject: str, noun_like: frozenset[str] = frozenset()) -> bool:
    return (
        bool(token)
        and token != subject
        and token not in STOPWORDS
        and token not in noun_like
        and is_finite_verb(token)
        and not is_infinitive(token)
        and not token.endswith(("вшись", "шись", "ли"))
    )


def _realizes_planned_event(tokens: list[str], plan: SentencePlan) -> bool:
    return (
        bool(tokens) and tokens[0] == plan.subject and plan.action in tokens
        and _is_semantic_action(plan.action, plan.subject)
        and not (set(tokens) & _FIRST_PERSON)
        and (not plan.object or plan.object == "scene" or plan.object in tokens)
    )


def _event_fallback(plan: SentencePlan, noun_like: frozenset[str] = frozenset()) -> list[str]:
    """Minimal truthful surface when no corpus span realizes the event."""
    if not _is_semantic_action(plan.action, plan.subject, noun_like):
        return []
    tokens = [plan.subject, plan.action]
    if plan.object and plan.object != "scene" and plan.object != plan.subject and _valid_fragment_object(plan.object, noun_like):
        tokens.append(plan.object)
    return tokens


def _pause_fallback(plan: SentencePlan) -> list[str]:
    """Last-resort grammatical pause; it asserts no event and commits no fact."""
    feminine = plan.subject.endswith(("а", "я"))
    options = ("молчала", "смотрела", "ждала", "думала") if feminine else ("молчал", "смотрел", "ждал", "думал")
    return [plan.subject, options[plan.index % len(options)]]


def _reasoning_transition(index: int) -> str:
    if index <= 0:
        return ""
    return ("затем", "после этого", "тогда")[index % 3]


def _sentence_candidates(sentence: SentencePlan, *, continuity_subject: str = "", noun_like: frozenset[str] = frozenset(), verb_lexicon: dict | None = None) -> tuple[Hypothesis, ...]:
    """Prefer a factual event, with an explicit non-committing surface fallback."""
    reasoned = _reasoned_sentence(sentence, noun_like, verb_lexicon)
    # A local planner fallback may be a noun falsely classified as an action.
    # Do not pass it to the surface in reasoning mode: the renderer will use a
    # grammatical non-event pause instead, and WorldState will not advance.
    fallback_sentence = reasoned or replace(sentence, action="", object="", fragment=(), detail_fragment=(), connector="")
    fallback = Hypothesis(
        name=f"fallback_{sentence.index:02d}", delta=StateDelta(label=f"stateless_fallback_{sentence.index}"),
        action="", involved_entities=(sentence.subject,), required_preconditions=("stateless_fallback",), payload=fallback_sentence,
    )
    candidates: list[Hypothesis] = [fallback]
    if reasoned is not None:
        candidates.append(event_hypothesis(
            name=f"sentence_{sentence.index:02d}", subject=reasoned.subject,
            action=reasoned.action, object_=reasoned.object, payload=reasoned,
        ))
    if continuity_subject and continuity_subject != sentence.subject:
        anchored = replace(
            fallback_sentence, subject=continuity_subject, focus=continuity_subject,
            fragment=(), detail_fragment=(), continuity_anchor=continuity_subject,
        )
        candidates.append(replace(
            fallback, name=f"active_fallback_{sentence.index:02d}",
            involved_entities=(continuity_subject,), payload=anchored,
        ))
        if reasoned is not None:
            candidates.append(event_hypothesis(
                name=f"active_sentence_{sentence.index:02d}", subject=anchored.subject,
                action=anchored.action, object_=anchored.object, payload=anchored,
            ))
    return tuple(candidates)


def _sentence_text(
    tokens: list[str], dialogue: bool, transition: str = "", proper_names: frozenset[str] = frozenset(),
    clause_break: int = 0,
) -> str:
    if not tokens:
        return "…"
    surface_tokens = [word.capitalize() if word in proper_names else word for word in tokens]
    if clause_break:
        left = " ".join(surface_tokens[:clause_break])
        right = " ".join(surface_tokens[clause_break + 1 :])
        text = f"{left}, {surface_tokens[clause_break]} {right}"
    else:
        text = " ".join(surface_tokens)
    text = f"{transition} {text}" if transition else text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?…":
        text += "."
    return f"– {text}" if dialogue else text


def _complete_tokens(tokens: list[str]) -> list[str]:
    """Remove a dangling surface tail left by the novelty-aware traversal.

    The phrase graph must sometimes stop immediately before a corpus 4-gram.
    Ending at a preposition, conjunction, demonstrative, or loose adverb makes
    that otherwise valid short clause unreadable.  This deterministic surface
    repair only removes generated tokens; it never adds a template word.
    """

    completed = list(tokens)
    while len(completed) > 2 and completed[-1] in _INCOMPLETE_TAILS:
        completed.pop()
    return completed if len(completed) >= 2 else []
