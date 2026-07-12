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
from poemcore.ingest import _looks_adjective, load_artifacts
from poemcore.line_plan import _ADJ_END
from poemcore.morphology import is_finite_verb, is_infinitive, is_latin_word, past_gender, sentence_agreement_errors
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
_PROMPT_WORD_RE = re.compile(r"[A-ZА-ЯЁa-zа-яё]+(?:-[A-ZА-ЯЁa-zа-яё]+)?")
_REQUEST_SHAPE_WORDS = frozenset({
    "предложение", "предложения", "предложений", "предложениях",
    "абзац", "абзаца", "абзацев", "абзаце", "слов", "слова",
    "одном", "двух", "трех", "трёх", "четырех", "четырёх", "пяти",
    "шести", "семи", "восьми", "девяти", "десяти",
    # English: nouns that name the requested output's *form* ("write a
    # SCENE about X"), not its content — same role as предложение/абзац above.
    "scene", "story", "paragraph", "passage", "sentence", "sentences",
    "paragraphs", "scenes", "stories", "words", "word", "poem", "poems",
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
    knowledge_actions: dict[str, tuple[tuple[str, str], ...]] | None = None,
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
        # A bare descriptive adjective ("a SHORT scene about a battle") is a
        # request-shape modifier, not the topic — the same role _REQUEST_SHAPE_WORDS
        # already filters for nouns. Skip only when a non-adjective word
        # survives too, so a genuinely adjective-only prompt still has a seed.
        and not (_looks_adjective(word) and word not in proper_names)
    ] or [
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
    #
    # "Write a scene about X" goes through the *same* three-layer machine, only
    # with ``scene_bias`` so the shared schema pool tilts toward event/action
    # shapes (something happens) rather than static description. The dedicated
    # multi-character narrative path below is reserved for dialogue /
    # introduction / continuation, which are genuinely different.
    if mode in ("description", "scene"):
        active_field = _topic_field(
            graph, tuple(seeds), phrase, noun_like, proper_names, description_relations or {},
        )
        location = next((word for word in seeds if _type_of(word) == "place"), "")
        goal = SceneGoal(
            topic=seeds[0], mode="description", speaker="", location=location,
            characters=(), seeds=tuple(seeds[:5]), active_field=active_field,
            scene_objective="describe_grounded_field" if mode == "description" else "narrate_grounded_scene",
        )
        return _plan_description_scene(
            goal, graph, phrase, sentence_count=sentence_count,
            noun_like=noun_like, genders=genders, selection_seed=variant_seed,
            description_relations=description_relations or {},
            adjective_collocations=meta.get("adjective_collocations", {}),
            mode_is_creative=True, verb_lexicon=verb_lexicon,
            scene_bias=(mode == "scene"),
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
        fact_options = (knowledge_actions or {}).get(subject, ())
        if fact_options:
            action, object_ = fact_options[index % len(fact_options)]
            # A knowledge triple is a planned event just like a corpus SVO
            # fragment.  Marking it as such lets the unchanged beam/search and
            # speech fallback preserve it instead of replacing it with the
            # non-committing "молчал" pause.
            fragment = (subject, action, object_)
        else:
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


# ===========================================================================
# Three-layer description generation, mirroring the QA open-synthesis path
# (knowledge -> discourse reasoning -> speech). Each layer is a separate
# function so the boundary is inspectable, exactly like QA's
# entity_answer_planner -> semantic_speech_planner -> phrase_graph split.
#
#   Layer 1  _gather_topic_knowledge   : WHAT is known about the topic,
#            (knowledge)                 bucketed by relation role. No wording.
#   Layer 2  _plan_topic_discourse     : HOW to say it — pick a rhetorical
#            (reasoning)                 schema and clause order per sentence,
#                                        seeded-random so shape varies; in
#                                        creative mode it may combine facts the
#                                        corpus never showed together and reach
#                                        for a figurative connector (the
#                                        "allowed to lie" freedom QA refuses).
#   Layer 3  _realize_topic_clauses    : surface each clause through the
#            (speech)                    phrase graph so connective tissue is
#                                        corpus-grown, not a fixed template.
# ===========================================================================


@dataclass(frozen=True)
class TopicKnowledge:
    """Layer 1 output: the bucketed facts about one topic. Pure content."""

    subject: str
    epithets: tuple[tuple[str, int], ...] = ()      # (adjective, weight)
    properties: tuple[tuple[str, str], ...] = ()     # (copula, detail) — "is dark"
    links: tuple[tuple[str, str], ...] = ()          # (preposition, object) — "of glory"
    predicates: tuple[str, ...] = ()                 # bare fronted predicate — "hath"
    events: tuple[tuple[str, str], ...] = ()         # (action, place phrase)
    actions: tuple[tuple[str, str], ...] = ()        # (verb, object) — real SVO ("banished duke")

    def has_any(self) -> bool:
        return bool(
            self.epithets or self.properties or self.links
            or self.predicates or self.events or self.actions
        )


# Light/support verbs need a real complement; standing alone ("battle let",
# "ship gave") they read as incomplete, and their objects are usually junk.
_EN_LIGHT_VERBS = frozenset({
    "let", "made", "make", "makes", "gave", "give", "gives", "got", "get",
    "gets", "put", "puts", "kept", "keep", "keeps", "had", "has", "have",
    "did", "does", "do", "took", "take", "takes", "done", "went", "put",
    "set", "sets", "laid", "lay", "held", "hold",
})
# Objects that are really adverbs/particles/quantifiers — not the patient of an
# action ("married BESIDES", "cried SUDDENLY", "warped THREE").
_EN_NUMBER_WORDS = frozenset({
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
    "ten", "hundred", "thousand", "million", "first", "second", "third",
})
_EN_WEAK_OBJECT = frozenset({
    "away", "back", "besides", "almost", "everywhere", "somewhere", "anywhere",
    "nowhere", "here", "there", "then", "now", "thus", "quite", "rather",
    "indeed", "forth", "aside", "apart", "along", "around", "about", "forward",
    "backward", "onward", "upward", "downward", "hither", "thither", "again",
    "once", "twice", "ever", "never", "still", "yet", "else", "too", "also",
    "enough", "instead", "perhaps", "however", "moreover", "therefore",
    # directional prepositions that leak into noun_like via the permissive
    # English noun detector — never a patient noun or a focus anchor
    "toward", "towards", "against", "between", "among", "beyond", "beneath",
    "beside", "across", "behind", "below", "above", "under", "over", "near",
})


def _valid_action_object_en(obj: str, noun_like: frozenset[str]) -> bool:
    """Is ``obj`` a plausible patient noun (English)? Rejects only the
    unambiguous non-nouns — adverbs (-ly), number words, and weak particles.
    Kept deliberately permissive: over-filtering shrinks the event pool and
    costs coherence more than the odd loose object costs readability."""
    if obj in _EN_NUMBER_WORDS or obj in _EN_WEAK_OBJECT or obj in STOPWORDS:
        return False
    if obj.endswith("ly") and len(obj) > 3:        # adverb
        return False
    return len(obj) >= 3


def _gather_topic_actions(
    subject: str, phrase: PhraseModel, noun_like: frozenset[str], verb_lexicon: dict | None,
    *, limit: int = 8,
) -> tuple[tuple[str, str], ...]:
    """Read real subject→verb(→object) events straight from the learned phrase
    graph, gated by the same SVO trust checks the scene renderer uses plus an
    English object-quality gate. This lets a *scene* about any topic show
    something happening ("murder committed", "ship lost treasure") without a
    hand-written template — verbs and objects are all corpus transitions of this
    exact subject — while dropping the noise ("ship married besides", "battle
    let george") that a permissive Cyrillic-only object check let through."""

    verbs = phrase.forward.get(subject, {})
    if not verbs:
        return ()
    english = is_latin_word(subject)
    scored: list[tuple[tuple[str, str], int]] = []
    seen: set[tuple[str, str]] = set()
    for verb, verb_count in verbs.items():
        if not _is_semantic_action(verb, subject, noun_like):
            continue
        # A light/support verb produces junk we can't trust — "battle let
        # george", "ship kept bucking", "ship gave". Skip it (English). This is
        # the one action filter that is a clear win; aggressive object/gerund
        # filtering was tried and reverted (it shrank the event pool and cost
        # more coherence than the loose objects cost readability — the residual
        # roughness like "looks"-as-subject is an ingest-level noun-detection
        # problem, to be fixed there, not by runtime patching).
        if english and verb in _EN_LIGHT_VERBS:
            continue
        objects = phrase.forward2.get((subject, verb), {})
        placed_object = False
        for obj, obj_count in objects.items():
            if not (_trusted_svo((subject, verb, obj), subject, noun_like, verb_lexicon) and obj != subject):
                continue
            if english and not _valid_action_object_en(obj, noun_like):
                continue
            key = (verb, obj)
            if key not in seen:
                seen.add(key)
                scored.append((key, verb_count + obj_count))
                placed_object = True
        # A bare intransitive event ("murder committed") is still a complete
        # assertion when no trusted object exists.
        if not placed_object and _trusted_svo((subject, verb), subject, noun_like, verb_lexicon):
            key = (verb, "")
            if key not in seen:
                seen.add(key)
                scored.append((key, verb_count))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return tuple(pair for pair, _weight in scored[:limit])


def _gather_topic_knowledge(
    subject: str, description_relations: dict[str, list[dict]],
    phrase: PhraseModel | None = None, noun_like: frozenset[str] = frozenset(),
    verb_lexicon: dict | None = None,
) -> TopicKnowledge:
    """Layer 1 (knowledge): read every observed relation about ``subject`` and
    sort it into role buckets. This is the analogue of QA's fact planner
    (``entity_answer_planner``/``synthesis_engine``) — it decides *what is
    known*, with no commitment yet to sentence shape.

    When ``phrase`` is supplied, real subject→verb→object events are also read
    from the phrase graph (``actions`` bucket), so a scene can show the topic
    *doing* something, not only be described."""

    rows = description_relations.get(subject, ())

    def _sorted(items: list[tuple], key_weight) -> tuple:
        return tuple(sorted(items, key=key_weight, reverse=True))

    epithets = _sorted(
        [(str(r["detail"]), int(r.get("weight", 1))) for r in rows
         if r.get("kind") == "epithet" and r.get("detail")],
        key_weight=lambda pair: pair[1],
    )
    properties = tuple(
        (str(r.get("predicate", "")), str(r["detail"])) for r in rows
        if r.get("kind") == "property" and r.get("detail")
    )
    links = tuple(
        (str(r.get("predicate", "")), str(r["detail"])) for r in rows
        if r.get("kind") == "object_link" and r.get("detail")
    )
    predicates = tuple(dict.fromkeys(
        str(r.get("predicate", "")) for r in rows
        if r.get("kind") == "predicate_before_subject" and r.get("predicate")
        # A fronted predicate only reads as a sentence opener when it is a
        # lexical verb ("Made war", "Contrived murder"). A copula, bare
        # auxiliary, or bare modal fronted ("Is true love", "Would murder")
        # is ungrammatical / incomplete without a following verb, so both are
        # excluded here — a copula can still appear in the copula_epithet clause.
        and str(r.get("predicate", "")) not in _COPULA_AUXILIARIES
        and str(r.get("predicate", "")) not in _BARE_MODALS
    ))
    events = tuple(
        (str(r.get("predicate", "")), str(r["detail"])) for r in rows
        if r.get("kind") in {"event_place", "event_place_inverted"} and r.get("predicate")
    )
    actions = (
        _gather_topic_actions(subject, phrase, noun_like, verb_lexicon)
        if phrase is not None else ()
    )
    return TopicKnowledge(subject, epithets, properties, links, predicates, events, actions)


# A "clause" is a role tag plus its already-chosen surface tokens; a sentence
# is an ordered list of clauses joined by connectors. The realizer (Layer 3)
# turns each into text. Roles are deliberately coarse — they name a rhetorical
# move, not a grammar rule.
Clause = tuple[str, tuple[str, ...]]  # (role, tokens)


# Rhetorical schemas: each names an ordered sequence of clause roles. Layer 2
# picks one per sentence (seeded), filtered to those whose required buckets are
# non-empty, so successive sentences take different shapes instead of the fixed
# "N adjectives + noun". Weights bias toward the more natural shapes without
# ever forcing one. `req` lists buckets that must be non-empty to use it.
# `dynamic` marks schemas that assert an *event* (something happens) rather than
# a static description — a scene re-weights toward these, a description away
# from them, from the *same* pool (no separate templated path per mode).
_DESCRIPTION_SCHEMAS: tuple[tuple[str, int, tuple[str, ...], tuple[str, ...], bool], ...] = (
    # name,               weight, required buckets,         clause roles,                         dynamic
    ("copula",            5, ("epithets",),                 ("subject", "copula_epithet"),          False),
    ("copula_link",       3, ("epithets", "links"),         ("subject", "copula_epithet", "link"),  False),
    ("epithet_np",        4, ("epithets",),                 ("epithet_subject",),                    False),
    ("epithet_np_link",   4, ("epithets", "links"),         ("epithet_subject", "link"),             False),
    ("link_np",           3, ("links",),                    ("subject", "link"),                     False),
    ("pred_front",        3, ("predicates", "epithets"),    ("predicate", "epithet_subject"),        True),
    ("pred_link",         2, ("predicates", "links"),       ("predicate", "subject", "link"),        True),
    # A subject known only through dialogue-tag verbs ("asked Sherlock",
    # "cried Silver") and nothing else — no epithet, property, or link at all
    # — still deserves a sentence ("Sherlock asked.") instead of silently
    # producing nothing.
    ("subject_predicate", 3, ("predicates",),               ("subject_predicate",),                  True),
    ("property",          3, ("properties",),               ("subject", "property"),                 False),
    ("property_epithet",  3, ("properties", "epithets"),    ("epithet_subject", "property"),         False),
    ("event",             3, ("events",),                   ("subject", "event"),                    True),
    ("event_epithet",     2, ("events", "epithets"),        ("epithet_subject", "event"),            True),
    ("link_copula",       2, ("links", "epithets"),         ("subject", "link", "copula_epithet"),   False),
    # Real corpus SVO events (the ``actions`` bucket) — the heart of "scene":
    # the topic *does* something to something. These clauses are self-contained
    # (they carry the subject), so they are never paired with a separate
    # ``subject`` clause that would duplicate it.
    ("action",            4, ("actions",),                  ("action",),                             True),
    ("action_object",     4, ("actions",),                  ("action_object",),                      True),
    ("epithet_action",    3, ("actions", "epithets"),       ("epithet_action",),                     True),
    ("action_then_link",  2, ("actions", "links"),          ("action_object", "link"),               True),
)
# How much a scene multiplies dynamic schemas' weight (and shrinks static ones);
# a description does the reverse. Same pool, just re-biased — universal.
_SCENE_DYNAMIC_BOOST = 6
_SCENE_STATIC_DAMP = 3

# Words that are grammatically noun-ish but make weak focus-chain anchors
# (deictics, quantifiers, light nouns). Following one produces "King cried ha.
# Higher, little ha." — kept out of the chain.
_WEAK_FOCUS = frozenset({
    "back", "one", "nothing", "something", "anything", "everything", "thing",
    "way", "part", "side", "kind", "sort", "deal", "lot", "bit", "end",
    "ha", "sir", "lord", "miss", "master", "madam", "man", "men", "some",
    "such", "same", "other", "another", "any", "none", "all", "half",
})

# Connectors joining two clauses of one sentence. In creative mode a figurative
# one may be chosen even where the corpus never joined those exact facts.
_PLAIN_CONNECTORS = (("", 6), (",", 2), ("and", 1))
_FIGURATIVE_CONNECTORS = (("", 6), (",", 2), ("and", 2), ("yet", 1))

# Words allowed as a graph bridge between two clause tokens (Layer 3). Kept to
# connectives/relativizers so bridging never inserts negation, an article, or a
# content word that would change or ungrammaticalize the clause.
_BRIDGE_WORDS = frozenset({
    "of", "in", "on", "to", "with", "from", "by", "for", "at", "that",
    "and", "as", "like", "through", "upon", "into", "within",
})

# Copulas / bare auxiliaries: fine inside a copula clause, wrong as a fronted
# sentence-opening predicate.
_COPULA_AUXILIARIES = frozenset({
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "have", "has", "had",
})
# Bare modals need a following main verb ("would murder someone"), so fronting
# one alone ("Would murder.") reads as an incomplete clause.
_BARE_MODALS = frozenset({
    "would", "will", "shall", "should", "could", "can", "might", "may", "must",
})

# A clause beginning with one of these attaches to the running subject on its
# own (copula, auxiliary, or preposition) — so no inter-clause conjunction and
# no graph bridge is placed before/after it.
_CLAUSE_ATTACH_HEADS = frozenset({
    "is", "are", "was", "were", "be", "being", "been", "hath", "has", "have",
    "had", "do", "does", "doth", "did", "made", "let", "shall", "will",
    "of", "in", "on", "to", "with", "from", "by", "for", "at", "into",
    "upon", "within", "through", "as", "like", "that", "and",
})


def _new_used_state() -> dict:
    """Per-subject bookkeeping so repeated sentences about one topic each use a
    *different* rhetorical schema and *fresh* facts (a several-sentence
    description is multiple angles on the topic, not the same clause restated)."""
    return {"schemas": set(), "epithets": set(), "links": set(), "properties": set(),
            "predicates": set(), "actions": set()}


def _plan_topic_discourse(
    knowledge: TopicKnowledge, *, index: int, mode_is_creative: bool, seed: str,
    field_knowledge: tuple[TopicKnowledge, ...],
    adjective_collocations: dict[str, dict[str, int]],
    used: dict | None = None, scene_bias: bool = False,
    connect_field: set[str] | None = None,
) -> tuple[list[Clause], str]:
    """Layer 2 (reasoning/discourse): choose a rhetorical schema and fill its
    clauses from the knowledge buckets. Returns ``(clauses, relation)``.

    This is the layer QA's ``semantic_speech_planner`` provides and the old
    description path lacked entirely: it decides discourse *shape*, seeded so
    each sentence differs, rather than always emitting the same template.

    ``used`` (per subject) is consumed and updated so that across a
    several-sentence paragraph the same topic gets a *new* schema and *unused*
    facts each time — describing more of it rather than repeating one clause.

    ``scene_bias`` re-weights the *same* schema pool toward event/action shapes
    (a scene: something happens) vs static description shapes — universal, no
    separate templated path per mode.

    Creative licence: unlike QA (which may only state observed facts), in
    creative mode this layer may borrow an epithet from a related field topic
    and attach it as if it belonged to the subject — a controlled, corpus-word
    but not corpus-fact combination ("allowed to lie").
    """

    used = used if used is not None else _new_used_state()
    schema_seed = f"{seed}|schema|{index}"
    available = {
        "epithets": bool([a for a, _w in knowledge.epithets if a not in used["epithets"]]),
        "properties": bool([p for i, p in enumerate(knowledge.properties) if i not in used["properties"]]),
        "links": bool([l for l in knowledge.links if "\t".join(l) not in used["links"]]),
        "predicates": bool([p for p in knowledge.predicates if p not in used["predicates"]]),
        "events": bool(knowledge.events),
        "actions": bool([a for a in knowledge.actions if a not in used["actions"]]),
    }

    def _weight(base: int, dynamic: bool) -> int:
        if not scene_bias:
            return base
        return base * _SCENE_DYNAMIC_BOOST if dynamic else max(1, base // _SCENE_STATIC_DAMP)

    usable = [
        (name, _weight(weight, dynamic), roles)
        for name, weight, req, roles, dynamic in _DESCRIPTION_SCHEMAS
        if all(available.get(bucket) for bucket in req)
    ]
    # Prefer a schema not yet used for this subject; fall back to any usable one
    # only once every shape has been spent.
    fresh = [entry for entry in usable if entry[0] not in used["schemas"]] or usable
    if not fresh:
        # Nothing observed (or all facts spent). In creative mode we are
        # permitted to *invent*: borrow a field neighbour's epithet.
        if mode_is_creative and field_knowledge:
            borrowed = next((fk for fk in field_knowledge if fk.epithets and fk.subject != knowledge.subject), None)
            if borrowed:
                adj = borrowed.epithets[0][0]
                return [("epithet_subject", (adj, knowledge.subject))], "epithet"
        return [("subject", (knowledge.subject,))], "observation"

    picked_name = seeded_weighted_pick(
        schema_seed, "pick", [(name, weight) for name, weight, _roles in fresh]
    )
    roles = next(roles for name, _w, roles in fresh if name == picked_name)
    used["schemas"].add(picked_name)

    epithet_budget = seeded_weighted_pick(f"{seed}|adjn|{index}", "n", [(1, 5), (2, 3), (3, 1)])
    chosen_epithets = _pick_collocated_epithets(
        knowledge, adjective_collocations, epithet_budget, f"{seed}|adj|{index}",
        exclude=used["epithets"],
    )
    used["epithets"].update(chosen_epithets)
    # Creative licence: occasionally spike in a borrowed field epithet.
    if mode_is_creative and field_knowledge and chosen_epithets:
        borrow = seeded_weighted_pick(f"{seed}|borrow|{index}", "b", [(0, 3), (1, 1)])
        if borrow:
            donor = next((fk for fk in field_knowledge if fk.epithets and fk.subject != knowledge.subject), None)
            if donor:
                chosen_epithets = (*chosen_epithets, donor.epithets[0][0])

    clauses: list[Clause] = []
    for role in roles:
        clause = _fill_clause(
            role, knowledge, chosen_epithets, seed=f"{seed}|fill|{index}|{role}", used=used,
            connect_field=connect_field or set(),
        )
        if clause[1]:
            clauses.append(clause)
    if not clauses:
        clauses = [("subject", (knowledge.subject,))]
    relation = clauses[0][0]
    return clauses, relation


def _pick_collocated_epithets(
    knowledge: TopicKnowledge, adjective_collocations: dict[str, dict[str, int]],
    count: int, seed: str, *, exclude: set | None = None,
) -> tuple[str, ...]:
    """Choose ``count`` epithets, weighting the 2nd+ by observed adjacency to
    an already-chosen one (collocation) rather than topic-fit alone."""

    exclude = exclude or set()
    pool = [(adj, max(1, weight)) for adj, weight in knowledge.epithets if adj not in exclude]
    if not pool:
        # Every epithet already used elsewhere in the paragraph — reuse is
        # better than an empty clause, so fall back to the full set.
        pool = [(adj, max(1, weight)) for adj, weight in knowledge.epithets]
    if not pool:
        return ()
    chosen: list[str] = []
    for pick_index in range(min(count, len(pool))):
        if not pool:
            break
        if chosen:
            reweighted = [
                (adj, weight * (1 + sum(adjective_collocations.get(a, {}).get(adj, 0) for a in chosen)))
                for adj, weight in pool
            ]
        else:
            reweighted = pool
        adj = seeded_weighted_pick(seed, f"e{pick_index}", reweighted)
        chosen.append(adj)
        pool = [(w, k) for w, k in pool if w != adj]
    return tuple(chosen)


def _connect_weight(token: str, connect_field: set[str]) -> int:
    """Weight multiplier for a fact whose object/detail links to the previous
    sentence (coherence #1). ×5 when connected, ×1 otherwise — a bias, never a
    force, so an unconnected but only-remaining fact is still usable."""
    return 5 if token in connect_field else 1


def _fill_clause(
    role: str, knowledge: TopicKnowledge, chosen_epithets: tuple[str, ...], *, seed: str,
    used: dict | None = None, connect_field: set[str] | None = None,
) -> Clause:
    """Turn a role tag into its surface tokens, drawing from the buckets. Uses
    ``used`` to pick a not-yet-spent link/property/predicate where possible, and
    ``connect_field`` to prefer a fact that develops the previous sentence."""

    used = used if used is not None else _new_used_state()
    connect_field = connect_field or set()
    subj = knowledge.subject
    if role == "subject":
        return ("subject", (subj,))
    if role == "epithet_subject":
        return ("epithet_subject", (*_comma_join(chosen_epithets), subj) if chosen_epithets else (subj,))
    if role == "copula_epithet":
        # A plain copula, not an arbitrary observed predicate: pulling
        # properties[0][0] here produced "love to true" / "war do great".
        adjs = chosen_epithets or ((knowledge.epithets[0][0],) if knowledge.epithets else ())
        return ("copula_epithet", ("is", *_comma_join(adjs)) if adjs else ())
    if role == "property":
        # A property's adjective must avoid both what this noun phrase already
        # carries ("dark night is dark") and any adjective already spent in the
        # paragraph as an epithet ("...is great" after "great ... love") — the
        # epithet and property buckets share the same adjectives.
        blocked = set(chosen_epithets) | used["epithets"]
        options = [
            (i, p) for i, p in enumerate(knowledge.properties)
            if i not in used["properties"] and p[1] not in blocked
        ]
        if not options:
            options = [(i, p) for i, p in enumerate(knowledge.properties) if p[1] not in set(chosen_epithets)]
        if not options:
            return ("property", ())
        pick_i = int(seeded_weighted_pick(
            seed, "prop", [(str(i), _connect_weight(p[1], connect_field)) for i, p in options]))
        used["properties"].add(pick_i)
        pred, detail = knowledge.properties[pick_i]
        if detail:
            used["epithets"].add(detail)  # so a later epithet clause won't reuse it
        return ("property", tuple(t for t in (pred, detail) if t))
    if role == "link":
        prep, obj = _pick_link(knowledge, seed, exclude=used["links"], connect_field=connect_field)
        if prep and obj:
            used["links"].add(f"{prep}\t{obj}")
        return ("link", (prep, obj) if prep and obj else ())
    if role == "predicate":
        options = [p for p in knowledge.predicates if p not in used["predicates"]] or list(knowledge.predicates)
        if not options:
            return ("predicate", ())
        pred = seeded_weighted_pick(seed, "pred", [(p, 1) for p in options])
        used["predicates"].add(pred)
        return ("predicate", (pred,))
    if role == "subject_predicate":
        options = [p for p in knowledge.predicates if p not in used["predicates"]] or list(knowledge.predicates)
        if not options:
            return ("subject_predicate", ())
        pred = seeded_weighted_pick(seed, "predtail", [(p, 1) for p in options])
        used["predicates"].add(pred)
        # One clause, subject then verb ("Sherlock asked") — not two clauses
        # joined by the generic connector logic, which would insert a comma
        # or "and" between them since "asked" isn't a clause-attach head.
        return ("subject_predicate", (subj, pred))
    if role == "event":
        action, place = knowledge.events[0] if knowledge.events else ("", "")
        return ("event", (action, *place.split())) if action else ("event", ())
    if role in {"action", "action_object", "epithet_action"}:
        options = [a for a in knowledge.actions if a not in used["actions"]] or list(knowledge.actions)
        if not options:
            return (role, ())
        encoded = [("\t".join(pair), _connect_weight(pair[1], connect_field)) for pair in options]
        verb, obj = seeded_weighted_pick(seed, "act", encoded).split("\t")
        used["actions"].add((verb, obj))
        keep_object = role != "action" and obj
        head: tuple[str, ...] = (
            (*_comma_join(chosen_epithets), subj) if role == "epithet_action" and chosen_epithets else (subj,)
        )
        tokens = (*head, verb, obj) if keep_object else (*head, verb)
        return (role, tokens)
    return (role, ())


def _pick_link(
    knowledge: TopicKnowledge, seed: str, *, exclude: set | None = None,
    connect_field: set[str] | None = None,
) -> tuple[str, str]:
    exclude = exclude or set()
    connect_field = connect_field or set()
    links = [pair for pair in knowledge.links if "\t".join(pair) not in exclude] or list(knowledge.links)
    if not links:
        return "", ""
    encoded = [("\t".join(pair), _connect_weight(pair[1], connect_field)) for pair in links]
    prep, obj = seeded_weighted_pick(seed, "link", encoded).split("\t")
    return prep, obj


def _comma_join(adjectives: tuple[str, ...]) -> tuple[str, ...]:
    """Insert a comma token between stacked adjectives ("wide, wild")."""
    if len(adjectives) <= 1:
        return adjectives
    out: list[str] = []
    for i, adj in enumerate(adjectives):
        if i:
            out.append(",")
        out.append(adj)
    return tuple(out)


def _plan_description_scene(
    goal: SceneGoal, graph: ConceptGraph, phrase: PhraseModel, *, sentence_count: int,
    noun_like: frozenset[str], genders: dict[str, str] | None, selection_seed: str,
    description_relations: dict[str, list[dict]],
    adjective_collocations: dict[str, dict[str, int]] | None = None,
    mode_is_creative: bool = True, verb_lexicon: dict | None = None,
    scene_bias: bool = False,
) -> ScenePlan:
    """Plan a topic description/scene through the three layers (see the block
    comment above ``TopicKnowledge``): gather knowledge (L1), choose a
    per-sentence rhetorical schema and fill its clauses (L2). The realizer (L3)
    turns each sentence's clause list into surface text.

    The *same* machine serves both "Describe X" and "Write a scene about X":
    ``scene_bias`` only re-weights the shared schema pool toward event/action
    shapes for a scene. There is no separate templated path per mode."""

    total = max(3, sentence_count)
    collocations = adjective_collocations or {}
    candidates = [word for word in dict.fromkeys((goal.topic, *goal.active_field, *goal.seeds)) if word != goal.location]

    # Layer 1: knowledge for the topic and every field neighbour, once. Passing
    # ``phrase`` lets Layer 1 also read real SVO events for the ``actions``
    # bucket (what the topic is observed *doing*), which the scene bias needs.
    knowledge_by_word: dict[str, TopicKnowledge] = {}
    for candidate in candidates:
        subj = _description_fact_subject(_description_subject(candidate, phrase, noun_like), description_relations)
        if subj:
            knowledge_by_word[candidate] = _gather_topic_knowledge(
                subj, description_relations, phrase, noun_like, verb_lexicon,
            )
    field_knowledge = tuple(knowledge_by_word.values())

    # Order the grounded subjects by how much material each has (richest first),
    # so a several-sentence paragraph opens on the topic itself and then works
    # through its neighbours. The primary topic always leads.
    grounded = [w for w in candidates if knowledge_by_word.get(w) and knowledge_by_word[w].has_any()]
    def _material(word: str) -> int:
        k = knowledge_by_word[word]
        return (len(k.epithets) + len(k.properties) + len(k.links)
                + len(k.predicates) + len(k.events) + len(k.actions))
    ordered_subjects = sorted(grounded, key=lambda w: (w != goal.topic, -_material(w), w))
    if not ordered_subjects:
        ordered_subjects = [goal.topic]

    # Per-subject "used" state so repeat visits to a subject each take a new
    # schema + fresh facts, and the paragraph reads as multiple facets rather
    # than one clause restated.
    used_state: dict[str, dict] = {}

    # Coherence (#1): concepts the previous sentence introduced. Layer 2 prefers
    # a fact whose object/detail connects to this set, so consecutive sentences
    # develop images already in play instead of listing disconnected facets.
    carried: list[str] = []
    first_mention: set[str] = set()  # subjects already named once (→ pronoun after)

    plans: list[SentencePlan] = []
    subjects_used: list[str] = []
    for index in range(total):
        # Gather knowledge on demand for carried concepts, so the focus chain
        # can follow a concept the pre-computed field didn't include.
        for word in carried:
            if word not in knowledge_by_word:
                subj = _description_fact_subject(word, description_relations)
                if subj:
                    knowledge_by_word[word] = _gather_topic_knowledge(
                        subj, description_relations, phrase, noun_like, verb_lexicon,
                    )
        # Focus chain (coherence #1): if the previous sentence introduced a
        # concept that is itself a grounded, not-yet-exhausted subject, follow
        # it — "Sea of glory. Glory faded." reads as developing images, and it
        # makes consecutive sentences share an anchor. Fall back to the
        # round-robin over the topic + field neighbours otherwise.
        # Follow probabilistically (~60%), not always: a mix of chained and
        # topic-anchored sentences keeps continuity high without every sentence
        # drifting to a new image (which costs local grammaticality).
        take_chain = seeded_weighted_pick(f"{selection_seed}|chain|{index}", "c", [(1, 3), (0, 2)])
        follow = next(
            (w for w in carried
             if w in knowledge_by_word and knowledge_by_word[w].has_any()
             and w != (subjects_used[-1] if subjects_used else "")
             and len(used_state.get(w, {}).get("schemas", ())) < 3
             # Only follow a real corpus noun with real substance — not a
             # preposition/adverbial ("toward", "back", "ha") that rode along in
             # a link/action clause. noun_like is the corpus's noun set, but the
             # English noun detector is permissive (it wrongly admits "toward"/
             # "looks"), so also exclude the known function/adverb sets.
             and w in noun_like and _material(w) >= 3 and len(w) >= 4
             and w not in _WEAK_FOCUS and w not in _BRIDGE_WORDS
             and w not in _CLAUSE_ATTACH_HEADS and w not in _EN_WEAK_OBJECT
             and not w.endswith("ly")),
            "",
        ) if take_chain else ""
        preferred = follow or ordered_subjects[index % len(ordered_subjects)]
        knowledge = knowledge_by_word.get(preferred)
        if knowledge is None or not knowledge.has_any():
            subj = _description_subject(preferred, phrase, noun_like)
            knowledge = knowledge_by_word.get(preferred) or TopicKnowledge(subj)
        subjects_used.append(knowledge.subject)
        state = used_state.setdefault(knowledge.subject, _new_used_state())

        # Field to prefer connecting to: the carried concepts plus their graph
        # neighbours (so a related-but-not-identical image also counts).
        connect_field: set[str] = set(carried)
        for word in carried:
            for neighbour, _weight in graph.neighbors(word)[:8]:
                connect_field.add(neighbour)

        # Layer 2: discourse plan for this sentence (consumes/updates state).
        clauses, relation = _plan_topic_discourse(
            knowledge, index=index, mode_is_creative=mode_is_creative,
            seed=selection_seed, field_knowledge=field_knowledge,
            adjective_collocations=collocations, used=state, scene_bias=scene_bias,
            connect_field=connect_field,
        )
        clause_fragments = tuple(tokens for _role, tokens in clauses if tokens)
        # Record the non-subject content words this sentence introduced, to
        # carry into the next one.
        introduced = [
            tok for frag in clause_fragments for tok in frag
            if tok not in {knowledge.subject, ",", "—", "is"} and tok not in STOPWORDS
            and len(tok) > 1 and not is_finite_verb(tok) and not _looks_adjective(tok)
            # Function words (prepositions/relativizers) ride along inside link
            # and action clauses; they are not content and must not be followed
            # as a focus ("...toward. Toward marched.").
            and tok not in _BRIDGE_WORDS and tok not in _CLAUSE_ATTACH_HEADS
        ]
        carried = list(dict.fromkeys(introduced))[:4] or carried
        # Connector between successive clauses — plain, or figurative in
        # creative mode (a join the corpus never literally made).
        connector_menu = _FIGURATIVE_CONNECTORS if mode_is_creative else _PLAIN_CONNECTORS
        clause_connectors = tuple(
            seeded_weighted_pick(f"{selection_seed}|conn|{index}|{position}", "c", list(connector_menu))
            for position in range(max(0, len(clause_fragments) - 1))
        )
        purpose = ("establish_field", "develop_field", "close_field")[min(index, 2)]
        plans.append(SentencePlan(
            index=index, purpose=purpose, focus=preferred, subject=knowledge.subject,
            action="", object="", speaker="", dialogue=False,
            continuity_anchor=goal.topic, relation=relation,
            clause_fragments=clause_fragments, clause_connectors=clause_connectors,
        ))
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
    count: int = 3, collocate_with: str = "",
    adjective_collocations: dict[str, dict[str, int]] | None = None,
) -> str:
    """Pick up to ``count`` observed agreeing epithets, joined into one phrase.

    Every word placed is still a real corpus-observed epithet for this exact
    subject (see ``ingest.py``'s epithet extraction) — chaining several is
    just a longer *true* noun phrase, not invented material.

    The second and third picks are additionally weighted by how often the
    candidate is actually observed *near another already-chosen descriptive
    word* (``adjective_collocations``, corpus adjective-adjective adjacency —
    see ``ingest.py``), not just by how strongly each word independently
    relates to the subject. Two epithets that are individually apt for
    "winter" but never occur near each other in the corpus are exactly what
    a collocation-blind topic-similarity pick would wrongly treat as
    interchangeable; weighting by attested co-usage prefers the pairing the
    corpus actually supports.
    """

    rows = [
        row for row in description_relations.get(subject, ())
        if row.get("kind") == "epithet" and row.get("detail") and row["detail"] not in exclude
    ]
    if not rows:
        return ""
    encoded = [(str(row["detail"]), max(1, int(row.get("weight", 1)))) for row in rows]
    collocations = adjective_collocations or {}
    # ``anchors`` seeds collocation weighting (includes the primary epithet
    # already rendered elsewhere, if any); ``chosen`` collects only the *new*
    # words this call returns.
    anchors: list[str] = [collocate_with] if collocate_with else []
    chosen: list[str] = []
    pool = list(encoded)

    def _reweighted(against: list[str]) -> list[tuple[str, int]]:
        if not against:
            return pool
        boosted = []
        for word, weight in pool:
            collocation_hits = sum(collocations.get(anchor, {}).get(word, 0) for anchor in against)
            boosted.append((word, weight * (1 + collocation_hits)))
        return boosted

    for pick_index in range(min(count, len(encoded))):
        if not pool:
            break
        word = seeded_weighted_pick(seed, f"description-epithet:{subject}:{pick_index}", _reweighted(anchors))
        chosen.append(word)
        anchors.append(word)
        pool = [(w, weight) for w, weight in pool if w != word]
    return ", ".join(chosen)


def _description_link(
    subject: str, description_relations: dict[str, list[dict]], seed: str, *, count: int = 2,
) -> tuple[str, ...]:
    """Pick up to ``count`` observed subject-to-object preposition links
    ("в трактире", "у окна"), concatenated into one longer token run.

    Each pair is independently corpus-observed for this subject; chaining a
    second one lengthens the sentence without fabricating a relation.
    """

    rows = [row for row in description_relations.get(subject, ()) if row.get("kind") == "object_link"]
    if not rows:
        return ()
    encoded = [
        ("\t".join((str(row.get("predicate", "")), str(row.get("detail", "")))),
         max(1, int(row.get("weight", 1))))
        for row in rows
    ]
    chosen: list[str] = []
    pool = list(encoded)
    seen_details: set[str] = set()
    for pick_index in range(min(count, len(encoded))):
        if not pool:
            break
        picked = seeded_weighted_pick(seed, f"description-link:{subject}:{pick_index}", pool)
        detail = picked.split("\t", 1)[-1]
        if detail in seen_details:
            pool = [(w, weight) for w, weight in pool if w != picked]
            continue
        seen_details.add(detail)
        chosen.extend(picked.split("\t"))
        pool = [(w, weight) for w, weight in pool if w != picked]
    return tuple(chosen)


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
        prev_subject = ""   # subject of the previous rendered description sentence
        rendered_any = False
        for sentence in plan.sentences:
            if plan.goal.mode == "description":
                # A sentence with no clause content — or one whose only content
                # is the bare subject (facts for this topic are exhausted) —
                # stays in SceneState but is not padded into prose as a
                # one-word pseudo-sentence ("Winter.").
                bare_subject = (
                    len(sentence.clause_fragments) == 1
                    and tuple(sentence.clause_fragments[0]) == (sentence.subject,)
                )
                if not sentence.clause_fragments or bare_subject:
                    continue
                text, trace = self._render_description_sentence(
                    sentence, seed=f"{seed}|{sentence.index}",
                    prev_subject=prev_subject, is_first=not rendered_any,
                )
                prev_subject = sentence.subject
                rendered_any = True
            else:
                text, trace = self._render_sentence(
                    sentence, state, allowed_names, paragraph.sentences, f"{seed}|{sentence.index}"
                )
            paragraph.sentences.append(text)
            paragraph.realization.append(trace)
            state.update(text)
        return paragraph

    def _render_description_sentence(
        self, plan: SentencePlan, *, seed: str, prev_subject: str = "", is_first: bool = True,
    ) -> tuple[str, SentenceRealization]:
        """Layer 3 (speech): realize the discourse plan's clauses into text.

        The clause list and order were already decided by Layer 2
        (``_plan_topic_discourse``); this only turns each clause's tokens into
        surface, letting the phrase graph supply connective tissue where it can
        (``_grow_clause_bridge``) so successive words are corpus transitions
        rather than a bare template concatenation. Clauses are joined by the
        connectors the plan chose.

        Two cross-sentence reading aids (coherence #1): when this sentence
        repeats the previous sentence's subject it is pronominalised ("It grew
        civil"), and a light discourse connective may open a later sentence
        ("Then it cried") — both seeded, so they are occasional, not mechanical.
        """

        rendered: list[str] = []
        for position, clause in enumerate(plan.clause_fragments):
            surface = self._grow_clause_bridge(clause, f"{seed}|clause|{position}")
            if not surface:
                continue
            if rendered:
                connector = plan.clause_connectors[position - 1] if position - 1 < len(plan.clause_connectors) else ""
                # A clause that opens with a function word (copula/preposition/
                # auxiliary — "is sweet", "of glory", "hath given") attaches to
                # the subject directly; a conjunction there reads as "love AND
                # is sweet". Only keep an explicit connector before a clause
                # that begins with content.
                if connector and surface[0] not in _CLAUSE_ATTACH_HEADS:
                    rendered.append(connector)
            rendered.extend(surface)
        if not rendered:
            rendered = [plan.subject]
        rendered = _strip_stray_punctuation([token for token in rendered if token])

        # Co-reference: a non-proper subject repeated from the previous sentence
        # becomes a pronoun (abstract/object topics take "it"). Proper names are
        # left in full — we have no reliable gender.
        pronominal = ""
        if (not is_first and prev_subject == plan.subject and rendered
                and rendered[0] == plan.subject and plan.subject not in self.proper_names):
            if seeded_weighted_pick(f"{seed}|pron", "p", [(1, 3), (0, 2)]):
                pronominal = "it"
                rendered = [pronominal, *rendered[1:]]

        # A rare inter-sentence connective (a light touch, never on every
        # sentence — "Then… Then… Then" is exactly the stamp to avoid). Only a
        # sentence that opens on the subject can take one, so it reads as a
        # discourse marker, not a random prefix.
        transition = ""
        if not is_first and rendered and rendered[0] in {plan.subject, pronominal}:
            transition = seeded_weighted_pick(
                f"{seed}|trans", "t", [("", 10), ("then", 1), ("and", 1)])

        body = _sentence_text(rendered, False, transition, self.proper_names)
        actual = set(words(body))
        realized = tuple(role for role in ("subject",) if plan.subject in actual)
        return body, SentenceRealization(
            plan.index, (plan.subject, plan.relation, ""), realized, False, ()
        )

    def _grow_clause_bridge(self, clause: tuple[str, ...], seed: str) -> list[str]:
        """Connect a clause's content tokens through learned transitions, so a
        bridging word the corpus actually used ("sea OF glory", "love THAT
        burns") can appear between them. Falls back to the literal tokens when
        the graph offers no observed bridge — never invents a transition, like
        the QA phrase-graph's slot walk.

        Only *connective* bridges are allowed (``_BRIDGE_WORDS``): a raw
        "follows head, precedes next" walk otherwise smuggles in negation and
        articles ("is NOT hideous", "is A hideous") that change or break the
        clause. Content words are never inserted as bridges."""

        tokens = [token for token in clause if token]
        if len(tokens) < 2:
            return tokens
        out = [tokens[0]]
        for nxt in tokens[1:]:
            head = out[-1]
            # Only bridge from a content-word head. Bridging from a copula or
            # preposition ("is that furious") is what a relativizer bridge
            # would otherwise smuggle in; those heads already attach directly.
            if head in {",", "—"} or head in _CLAUSE_ATTACH_HEADS:
                out.append(nxt)
                continue
            direct = self.phrase.forward.get(head, {})
            if nxt in direct:
                out.append(nxt)
                continue
            bridges = [
                (word, count) for word, count in direct.items()
                if word in _BRIDGE_WORDS and word not in {head, nxt}
                and nxt in self.phrase.forward.get(word, {})
            ]
            if bridges:
                out.append(seeded_weighted_pick(seed, f"bridge:{head}:{nxt}", bridges))
            out.append(nxt)
        return out

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
        knowledge_facts: tuple[tuple[str, str, str], ...] = (),
    ) -> NarrativeResult:
        request = NarrativeRequest(prompt=prompt, context=context, sentences=sentences)
        if seed is not None:
            return self._run_once(
                request, render_seed=seed, reasoning=reasoning,
                knowledge_facts=knowledge_facts,
            )

        # A new random route normally diverges by itself. Retain a compact
        # per-engine history too: if the beam nevertheless lands on the same
        # paragraph for the same request, sample another valid route.
        key = (prompt, context, sentences, reasoning)
        seen = self._recent_outputs.setdefault(key, [])
        fallback: NarrativeResult | None = None
        for _attempt in range(8):
            result = self._run_once(
                request, render_seed=secrets.token_hex(8), reasoning=reasoning,
                knowledge_facts=knowledge_facts,
            )
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
        knowledge_facts: tuple[tuple[str, str, str], ...] = (),
    ) -> NarrativeResult:
        restore_graph = self._fuse_knowledge_layer(knowledge_facts)
        plan = plan_scene(
            self.graph, self.phrase, prompt=request.prompt, context=request.context, meta=self.meta,
            proper_names=self.proper_names, sentence_count=request.sentences,
            entity_types=self.entity_types, genders=self.gender_map, noun_like=self.noun_like,
            verb_lexicon=self.verb_lexicon,
            speech_frames=self.speech_frames,
            description_relations=self.description_relations,
            knowledge_actions=_knowledge_actions(knowledge_facts),
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
        result = NarrativeResult(
            request, plan, paragraph, assess_poem(self.phrase, paragraph.sentences), self.meta,
            reasoning_plan=reasoning_plan, world_state=committed_state, scene_state=scene_state,
        )
        restore_graph()
        return result

    def _fuse_knowledge_layer(
        self, facts: tuple[tuple[str, str, str], ...]
    ) -> callable:
        """Temporarily add QA facts to the concept graph for one scene.

        This changes only the knowledge layer: the NarrativeEngine's scene
        planner, beam search, phrase model, and speech realization are reused
        unchanged.  The bridge is serial, but the original weights/edges are
        restored after the request so transient QA context never becomes
        learned literary memory.
        """

        weight_before: dict[str, float | None] = {}
        edge_before: dict[tuple[str, str], float | None] = {}
        for subject, _predicate, object_ in facts:
            subjects = [term for term in words(subject) if len(term) > 1]
            objects = [term for term in words(object_) if len(term) > 1]
            for term in (*subjects, *objects):
                if term not in weight_before:
                    weight_before[term] = self.graph.weight.get(term)
                    self.graph.weight[term] = self.graph.weight.get(term, 0.5) + 2.0
            for left in subjects:
                for right in objects:
                    if left == right:
                        continue
                    for edge in ((left, right), (right, left)):
                        if edge not in edge_before:
                            edge_before[edge] = self.graph.edges.get(edge)
                            self.graph.edges[edge] = self.graph.edges.get(edge, 0.0) + 2.0
        self.graph._adjacency = None

        def restore() -> None:
            for term, previous in weight_before.items():
                if previous is None:
                    self.graph.weight.pop(term, None)
                else:
                    self.graph.weight[term] = previous
            for edge, previous in edge_before.items():
                if previous is None:
                    self.graph.edges.pop(edge, None)
                else:
                    self.graph.edges[edge] = previous
            self.graph._adjacency = None

        return restore


def _knowledge_actions(
    facts: tuple[tuple[str, str, str], ...]
) -> dict[str, tuple[tuple[str, str], ...]]:
    """Expose factual predicates to the existing planner, not the renderer."""

    grouped: dict[str, list[tuple[str, str]]] = {}
    for subject, predicate, object_ in facts:
        subjects = words(subject)
        predicates = words(predicate)
        objects = words(object_)
        if not subjects or not predicates or not objects:
            continue
        grouped.setdefault(subjects[0], []).append((predicates[0], objects[0]))
    return {subject: tuple(options) for subject, options in grouped.items()}


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
    if is_latin_word(plan.subject):
        # English marks no gender, so there is one options tuple, not two.
        options = ("paused", "waited", "watched", "wondered")
        return [plan.subject, options[plan.index % len(options)]]
    feminine = plan.subject.endswith(("а", "я"))
    options = ("молчала", "смотрела", "ждала", "думала") if feminine else ("молчал", "смотрел", "ждал", "думал")
    return [plan.subject, options[plan.index % len(options)]]


def _reasoning_transition(index: int, *, latin: bool = False) -> str:
    if index <= 0:
        return ""
    if latin:
        return ("then", "after that", "later")[index % 3]
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
        text = _join_with_punctuation(surface_tokens)
    text = f"{transition} {text}" if transition else text
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?…":
        text += "."
    return f"– {text}" if dialogue else text


def _strip_stray_punctuation(tokens: list[str]) -> list[str]:
    """Drop a comma/dash that isn't flanked by two words on both sides, so a
    deduped adjective list can't leave a dangling "Grim-visaged, war"."""
    marks = {",", ";", ":", "—", "…"}
    out: list[str] = []
    for i, token in enumerate(tokens):
        if token in marks:
            prev_word = bool(out) and out[-1] not in marks
            next_word = i + 1 < len(tokens) and tokens[i + 1] not in marks
            if not (prev_word and next_word):
                continue
        out.append(token)
    return out


def _join_with_punctuation(tokens: list[str]) -> str:
    """Join tokens with spaces, but attach bare punctuation ("," "—") to the
    preceding word so a comma-separated adjective list reads "wide, wild" not
    "wide , wild"."""
    out = ""
    for token in tokens:
        if not out:
            out = token
        elif token in {",", ";", ":", "—", "…"}:
            out += token
        else:
            out += f" {token}"
    return out


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
