"""Engine — orchestrates prompt → plan → render, the analogue of the
production ``answer_orchestrator``.

It parses a free-form prompt into an explicit request (mode, theme, style
author, seed concepts), runs the reasoning layer (planner) over the concept
graph, then the language layer (generator), and finally the novelty gate. Each
stage hands the next a typed artifact and nothing else — the same layer
discipline as production, minus the QA question-parsing machinery, which is
replaced by this small intent reader.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from poemcore.concept_graph import ConceptGraph
from poemcore.generator import Generator, RenderedPoem
from poemcore.ingest import load_artifacts
from poemcore.line_plan import LineIntent, PoemIntent, build_line_intents, build_poem_intent
from poemcore.novelty import NoveltyReport, assess_poem
from poemcore.phrase_model import PhraseModel
from poemcore.planner import PoemPlan, plan_poem
from poemcore.reasoning import PoemReasoning, reason_poem
from poemcore.text import content_words, rhyme_key

# Theme keyword -> classical corpus seed concepts. The map bridges an
# English/Russian prompt word to imagery that actually exists in the corpus, so
# "space" reaches for звёзды/небо/лазурь ("classical Russian imagery") rather
# than a modern term the poets never used. Seeds are filtered against the graph
# at request time, so only concepts present in the corpus survive.
THEME_SEEDS: dict[str, list[str]] = {
    "autumn": ["осень", "осени", "лист", "увяданье", "багрец", "поле", "туча"],
    "осень": ["осень", "осени", "лист", "увяданье", "багрец", "поле"],
    "winter": ["зима", "мороз", "снег", "вьюга", "иней", "буря"],
    "зима": ["зима", "мороз", "снег", "вьюга", "иней"],
    "spring": ["весна", "гроза", "гром", "лазурь", "птичий"],
    "весна": ["весна", "гроза", "гром", "лазурь"],
    "space": ["звезда", "звёзды", "небо", "лазурь", "луна", "небеса"],
    "космос": ["звезда", "звёзды", "небо", "лазурь", "луна", "небеса"],
    "sea": ["море", "волны", "парус", "буря", "лазури"],
    "море": ["море", "волны", "парус", "буря"],
    "love": ["любовь", "душа", "нежный", "сердце", "печаль"],
    "любовь": ["любовь", "душа", "нежный", "сердце"],
    "night": ["ночь", "луна", "звёзды", "тень", "сон"],
    "ночь": ["ночь", "луна", "звёзды", "тень"],
}

# Author names in both scripts, so an English prompt ("style of Pushkin") and a
# Russian one ("в стиле Пушкина") both resolve to the same profile key.
_AUTHOR_ALIASES: dict[str, str] = {
    "пушкин": "Пушкин", "pushkin": "Пушкин",
    "лермонтов": "Лермонтов", "lermontov": "Лермонтов",
    "тютчев": "Тютчев", "tyutchev": "Тютчев", "tютчев": "Тютчев",
    "фет": "Фет", "fet": "Фет",
    "блок": "Блок", "blok": "Блок",
    "ахматова": "Ахматова", "akhmatova": "Ахматова",
}


@dataclass
class Request:
    mode: str                       # "compose" | "continue" | "style"
    theme: str = ""
    style_author: str = ""
    seed_concepts: list[str] = field(default_factory=list)
    given_verse: str = ""
    stanzas: int = 2

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "theme": self.theme,
            "style_author": self.style_author,
            "seed_concepts": self.seed_concepts,
            "given_verse_lines": self.given_verse.count("\n") + 1 if self.given_verse else 0,
        }


@dataclass
class Result:
    request: Request
    plan: PoemPlan
    poem: RenderedPoem
    novelty: NoveltyReport
    poem_intent: PoemIntent | None = None
    line_intents: list[LineIntent] = field(default_factory=list)
    reasoning: PoemReasoning | None = None


class PoetryEngine:
    def __init__(self, artifact_path: Path | None = None) -> None:
        data = load_artifacts(artifact_path)
        self.phrase = PhraseModel.from_dict(data["phrase_model"])
        self.graph = ConceptGraph.from_dict(data["concept_graph"])
        self.rhyme_groups = data["rhyme_groups"]
        self.style_profiles = data["style_profiles"]
        self.proper_names = frozenset(data.get("proper_names", ()))
        self.meta = data["meta"]

    # -- intent reading ------------------------------------------------------

    def parse(self, prompt: str, given_verse: str = "") -> Request:
        low = prompt.lower()
        style_author = self._detect_author(low)

        if given_verse or re.search(r"\b(continue|продолж)", low):
            seeds = self._seed_from_verse(given_verse) or self._seed_from_theme(low)
            return Request(
                mode="continue",
                theme="продолжение",
                style_author=style_author,
                seed_concepts=seeds,
                given_verse=given_verse.strip(),
            )

        theme, seeds = self._detect_theme(low)
        if not seeds and style_author:
            return Request(mode="style", theme=theme or style_author, style_author=style_author,
                           seed_concepts=self._style_seed(style_author))
        return Request(mode="compose", theme=theme, style_author=style_author, seed_concepts=seeds)

    def _detect_author(self, low: str) -> str:
        for alias, canonical in _AUTHOR_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}", low):
                return canonical if canonical in self.style_profiles else canonical
        return ""

    def _detect_theme(self, low: str) -> tuple[str, list[str]]:
        # explicit "about X" / "о/об/про X"
        m = re.search(r"(?:about|про|об|о)\s+([a-zа-яё]+)", low)
        candidate = m.group(1) if m else ""
        for key in THEME_SEEDS:
            if key in low:
                return key, self._seed_from_theme_key(key)
        if candidate:
            seeds = self._seed_from_theme_key(candidate)
            if seeds:
                return candidate, seeds
        return candidate, self._seed_from_theme(low)

    def _seed_from_theme_key(self, key: str) -> list[str]:
        wanted = THEME_SEEDS.get(key, [key])
        return [w for w in wanted if w in self.graph.weight]

    def _seed_from_theme(self, low: str) -> list[str]:
        seeds: list[str] = []
        for w in content_words(low):
            if w in self.graph.weight:
                seeds.append(w)
        return seeds

    def _seed_from_verse(self, verse: str) -> list[str]:
        if not verse:
            return []
        graph_words = [w for w in content_words(verse) if w in self.graph.weight]
        # rank by activation potential (graph node weight) and keep the top few
        graph_words.sort(key=lambda w: -self.graph.weight.get(w, 0))
        return graph_words[:4]

    def _style_seed(self, author: str) -> list[str]:
        vocab = self.style_profiles.get(author, {}).get("signature_vocab", [])
        return [w for w in vocab if w in self.graph.weight][:5]

    # -- run -----------------------------------------------------------------

    def run(
        self,
        prompt: str,
        given_verse: str = "",
        *,
        stanzas: int = 2,
        seed: str | None = None,
        rank: bool = True,
        use_line_plan: bool = True,
        use_seeded_generation: bool = True,
        use_reasoning: bool = True,
    ) -> Result:
        request = self.parse(prompt, given_verse)
        request.stanzas = stanzas
        style = self.style_profiles.get(request.style_author, {})
        target = _target_syllables(style)
        plan = plan_poem(
            self.graph,
            request.seed_concepts or ["тишина"],
            theme=request.theme,
            style_author=request.style_author,
            stanzas=stanzas,
            target_syllables=target,
            continuation=(request.mode == "continue"),
        )
        # The reasoning layer commits to a goal, stanza purposes, and one
        # propositional line decision before language traversal begins.
        speaker = "я"
        reasoning = reason_poem(
            self.graph, self.phrase, plan, theme=request.theme, speaker=speaker
        ) if use_reasoning else None
        poem_intent = build_poem_intent(
            theme=request.theme,
            seed_concepts=request.seed_concepts,
            graph=self.graph,
            speaker=speaker,
            setting=reasoning.goal.setting if reasoning else None,
            mood=reasoning.goal.mood if reasoning else None,
        )
        line_intents = build_line_intents(
            poem_intent, plan.lines, reasoning.lines if reasoning else None
        )
        gen = Generator(
            self.phrase, self.graph, self.rhyme_groups, style,
            rank=rank, proper_names=self.proper_names, use_line_plan=use_line_plan,
            use_seeded_generation=use_seeded_generation,
        )
        render_seed = seed or self._render_seed(request)
        poem = gen.render(
            plan, seed=render_seed, poem_intent=poem_intent, line_intents=line_intents
        )
        if request.mode == "continue" and request.given_verse:
            poem.lines = request.given_verse.splitlines() + ["", "— — —", ""] + poem.lines
        novelty = assess_poem(self.phrase, poem.lines)
        return Result(
            request=request, plan=plan, poem=poem, novelty=novelty,
            poem_intent=poem_intent, line_intents=line_intents, reasoning=reasoning,
        )

    def _render_seed(self, request: Request) -> str:
        return f"{request.mode}|{request.theme}|{request.style_author}|{','.join(request.seed_concepts)}"


def _target_syllables(style: dict) -> int:
    common = style.get("common_syllables") or []
    if common:
        return int(common[0])
    avg = style.get("avg_syllables") or 8
    return max(6, min(11, round(avg)))
