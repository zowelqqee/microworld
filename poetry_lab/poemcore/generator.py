"""Language realization layer — render a PoemPlan into lines.

Kept as intact as the production language layer. In
``worldpgt/cognition/phrase_graph.py`` the renderer walks a learned frequency
graph of fragments/transitions and stitches a plan into connected prose via
deterministic, seeded traversal — never inventing content outside the learned
graph. This module does exactly that for verse:

  * a rhyme word is chosen first (the fixed point of the line),
  * the line is grown backward through learned word transitions to hit a
    syllable target (meter),
  * activation from the concept graph biases the walk toward the planned image,
  * the seeded pick makes every line replayable.

The only creative freedom added over production is intentional and required by
the research question: the walk is allowed to assemble word sequences the
corpus never contained (novel combination), while ``novelty.py`` forbids it from
reproducing a corpus 4-gram (memorisation). Facts-vs-nonfacts is not a concern
here; imagery-vs-imagery combination is the point.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from poemcore.concept_graph import ConceptGraph
from poemcore.discourse import DiscourseState, line_salience
from poemcore.line_plan import (
    LineIntent, PoemIntent, decision_role_hits, line_plan_score, seed_tokens,
)
from poemcore.novelty import check_line
from poemcore.phrase_model import PhraseModel, seeded_weighted_pick
from poemcore.planner import LinePlan, PoemPlan
from poemcore.text import rhyme_key, syllables

# Re-rolls per line before accepting a possibly-echoed fallback. Raised from 4
# when the phrase model went order-2: trigram growth reconstructs real corpus
# 4-grams more often, so the novelty gate needs more tries to find a clean line.
_MAX_LINE_ATTEMPTS = 8

# Weight on the (normalized ~0–2) discourse-salience term when combined with the
# line-plan score (~±8), so continuity stays a live tie-breaker. See
# _select_candidate.
_SALIENCE_WEIGHT = 3.0

# Intent-seeding aggressiveness. The A/B sweep (see README) found the
# graph-neighbour fallback tier from the brief's step 3 to be *counterproductive*
# — it perturbs the walk into name-dense, verb-poor regions without improving
# plan satisfaction (neighbours aren't the planned concept the metric counts).
# Forcing only the single exact seed hits the plan-satisfaction target ~5x while
# leaving grammaticality and rhyme at baseline. The neighbour tier stays
# implemented and callable; the default just doesn't use it.
_SEED_MAX = 1
_SEED_NEIGHBORS = 0


def _syllable_len(toks: list[str]) -> int:
    return syllables(" ".join(toks))


@dataclass
class RenderedPoem:
    title: str
    lines: list[str] = field(default_factory=list)
    realization: list["SurfaceRealization"] = field(default_factory=list)

    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass(frozen=True)
class SurfaceRealization:
    """Inspectable handoff from a line decision to its rendered surface."""

    index: int
    strategy: str
    planned: tuple[str, str, str]
    realized: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "strategy": self.strategy,
            "planned": {
                "subject": self.planned[0], "action": self.planned[1], "object": self.planned[2],
            },
            "realized": list(self.realized),
        }


@dataclass
class _RhymeIndex:
    key_to_words: dict[str, list[tuple[str, int]]]
    word_to_key: dict[str, str]

    @classmethod
    def build(cls, rhyme_groups: dict) -> "_RhymeIndex":
        key_to_words: dict[str, list[tuple[str, int]]] = {}
        word_to_key: dict[str, str] = {}
        for key, words_counts in rhyme_groups.items():
            pairs = sorted(words_counts.items(), key=lambda kv: (-kv[1], kv[0]))
            key_to_words[key] = pairs
            for w, _c in pairs:
                word_to_key.setdefault(w, key)
        return cls(key_to_words, word_to_key)


class Generator:
    def __init__(
        self,
        phrase: PhraseModel,
        graph: ConceptGraph,
        rhyme_groups: dict,
        style: dict | None = None,
        *,
        rank: bool = True,
        proper_names: frozenset[str] | None = None,
        use_line_plan: bool = True,
        use_seeded_generation: bool = True,
    ) -> None:
        self.phrase = phrase
        self.graph = graph
        self.rhyme = _RhymeIndex.build(rhyme_groups)
        self.style = style or {}
        # rank=False restores the pre-transfer 'accept first valid line'
        # behaviour, kept so the discourse-salience selection can be A/B'd.
        self.rank = rank
        # line-plan scoring: adds an intent-satisfaction term to selection.
        # use_line_plan=False A/B's it against discourse ranking alone.
        self.proper_names = proper_names or frozenset()
        self.use_line_plan = use_line_plan
        # intent-seeded generation: force an intent token into candidate lines
        # during growth (not just ranking). use_seeded_generation=False A/B's it.
        self.use_seeded_generation = use_seeded_generation

    # -- public --------------------------------------------------------------

    def render(
        self,
        plan: PoemPlan,
        *,
        seed: str,
        poem_intent: PoemIntent | None = None,
        line_intents: list[LineIntent] | None = None,
    ) -> RenderedPoem:
        per_stanza = len(plan.rhyme_scheme)
        boost_all = self._activation_boost(plan)
        # rhyme label -> chosen rhyme_key, reset each stanza so ABAB repeats
        label_keys: dict[str, str] = {}
        used_rhyme_words: set[str] = set()
        # Discourse state threads the poem's active images across lines. Seeded
        # from the prompt concepts so the first line already has a topic to
        # continue; each chosen line feeds its images back in, so the *next*
        # line is scored for continuing them. This is the ported dialogue
        # salience mechanism operating across the whole poem.
        state = DiscourseState.seeded(plan.seed_concepts) if self.rank else DiscourseState()
        lines: list[str] = []
        realization: list[SurfaceRealization] = []
        for ln in plan.lines:
            if ln.index % per_stanza == 0:
                label_keys = {}
            line_seed = f"{seed}|{ln.index}"
            boost = self._line_boost(ln, boost_all)
            intent = line_intents[ln.index] if line_intents and ln.index < len(line_intents) else None
            text, trace = self._render_line(
                ln, boost, line_seed, label_keys, used_rhyme_words, state, poem_intent, intent
            )
            lines.append(text)
            if trace is not None:
                realization.append(trace)
            # stanza spacing
            if (ln.index + 1) % per_stanza == 0 and ln.index + 1 < len(plan.lines):
                lines.append("")
        title = self._title(plan)
        return RenderedPoem(title=title, lines=lines, realization=realization)

    # -- line rendering ------------------------------------------------------

    def _render_line(
        self,
        ln: LinePlan,
        boost: dict[str, float],
        line_seed: str,
        label_keys: dict[str, str],
        used_rhyme_words: set[str],
        state: DiscourseState,
        poem_intent: PoemIntent | None = None,
        intent: LineIntent | None = None,
    ) -> tuple[str, SurfaceRealization | None]:
        # Produce candidates across attempts, each with a fresh seed (different
        # rhyme word, different trigram path). Meter and novelty are the gates
        # here; the discourse salience choice below only ranks *between* lines
        # that already passed them. ``fallback`` holds a novel-but-off-meter
        # line, ``last_resort`` any non-empty line at all — so a slot where every
        # attempt echoed a 4-gram still yields a real line, never a bare "…".
        # Intent-seeding: ordered tokens to try to force into the line during
        # growth (exact intent seeds, then graph-neighbour fallbacks). Empty
        # unless seeded generation is on and an intent is supplied — so this is
        # purely additive to the traversal, never a rewrite of it.
        must_include: tuple[str, ...] = ()
        if self.use_seeded_generation and intent is not None and poem_intent is not None:
            must_include = seed_tokens(
                intent, poem_intent, self.graph, self.proper_names,
                max_seeds=_SEED_MAX, max_neighbors=_SEED_NEIGHBORS,
            )

        candidates: list[tuple[list[str], bool, str]] = []  # toks, forward, strategy
        fallback: tuple[list[str], bool, str] | None = None
        last_resort: tuple[list[str], bool, str] | None = None
        for attempt in range(_MAX_LINE_ATTEMPTS):
            attempt_seed = f"{line_seed}#{attempt}"
            forward_used = False
            strategy = "ordinary"
            slots = self._decision_slots(intent)
            # The first occurrence of each rhyme label may establish its own
            # ending.  Starting that line from the decided subject gives the
            # surface layer a full forward route through its roles; the later
            # partner of the label still grows backward toward that ending.
            use_forward_roles = bool(
                slots and slots[0] in self.phrase.unigram and ln.rhyme_label not in label_keys
            )
            if ln.prefer_forward or use_forward_roles:
                if slots and slots[0] in self.phrase.unigram:
                    toks = self.phrase.grow_forward_slots(
                        slots[0], ln.target_syllables, attempt_seed, boost=boost, slots=slots[1:]
                    )
                    strategy = "role_anchored_forward"
                else:
                    toks = self._grow_opener(ln, boost, attempt_seed, must_include)
                forward_used = True
                # forward growth dead-ends on words that only ever ended a line
                # in the corpus (no successor transition); when that leaves the
                # line too short for the meter target, fall back to the
                # rhyme-seeded backward walk, which never starves.
                if _syllable_len(toks) < ln.target_syllables - 1:
                    forward_used = False
            if not forward_used:
                end_word = self._choose_rhyme_word(
                    ln, boost, attempt_seed, label_keys, used_rhyme_words
                )
                if slots:
                    toks = self.phrase.grow_backward_slots(
                        end_word, ln.target_syllables, attempt_seed, boost=boost, slots=slots
                    )
                    strategy = "role_anchored_backward"
                else:
                    toks = self.phrase.grow_backward(
                        end_word, ln.target_syllables, attempt_seed,
                        boost=boost, must_include=must_include,
                    )
            if toks and (last_resort is None or _syllable_len(toks) >= ln.target_syllables - 1):
                last_resort = (toks, forward_used, strategy)  # best on-meter line seen, novel or not
            if not toks or not check_line(self.phrase, " ".join(toks)):
                continue
            # Two-sided meter window, not just a lower bound: the salience
            # ranker below rewards content-word density, which correlates with
            # line length, so an open upper bound let it drift toward long,
            # off-meter lines. Gating both sides keeps every ranked candidate
            # on-meter, so selection cannot trade meter for continuity.
            if abs(_syllable_len(toks) - ln.target_syllables) <= 1:
                candidates.append((toks, forward_used, strategy))
            elif fallback is None:
                fallback = (toks, forward_used, strategy)  # novel but off-meter — last resort

        chosen = self._select_candidate(candidates, ln, state, poem_intent, intent) or fallback or last_resort
        if chosen is None:
            return "…", self._surface_trace(ln.index, "empty", intent, "…")
        toks, forward_used, strategy = chosen
        if toks:
            used_rhyme_words.add(toks[-1])
            label_keys.setdefault(ln.rhyme_label, rhyme_key(toks[-1]))
        state.update(" ".join(toks))
        text = _format_line(toks)
        return text, self._surface_trace(ln.index, strategy, intent, text)

    def _select_candidate(
        self,
        candidates: list[tuple[list[str], bool, str]],
        ln: LinePlan,
        state: DiscourseState,
        poem_intent: PoemIntent | None = None,
        intent: LineIntent | None = None,
    ) -> tuple[list[str], bool, str] | None:
        """Pick the candidate that best continues the poem's active images and,
        when a line intent is supplied, best satisfies it.

        The score is discourse salience (continuity) plus an optional line-plan
        term (subject/action present, planned concepts realized, proper names
        and disconnected entity jumps penalized). Both are additive scoring
        terms over already-gated candidates — the traversal is untouched. With
        ranking off it falls through to the first candidate (old behaviour)."""

        if not candidates:
            return None
        if not self.rank:
            return candidates[0]
        field_ = state.salience_field(self.graph)
        use_plan = self.use_line_plan and intent is not None and poem_intent is not None
        best = candidates[0]
        best_score = None
        for cand in candidates:
            text = " ".join(cand[0])
            score, _b = line_salience(text, state, self.graph, ln.focus, field_=field_)
            # Salience is normalized (~0–2); the plan score spans ~±8. Left
            # unscaled the plan term swamps continuity, so among candidates that
            # tie on subject/action the more-continuous one no longer wins.
            # Weighting salience up keeps continuity a real tie-breaker without
            # overriding the plan's hard preferences (name/no-verb penalties).
            score *= _SALIENCE_WEIGHT
            if use_plan:
                plan_score, _pb = line_plan_score(
                    text, intent, poem_intent, self.graph, self.proper_names, field_
                )
                score += plan_score
            if best_score is None or score > best_score:
                best_score = score
                best = cand
        return best

    def _decision_slots(self, intent: LineIntent | None) -> tuple[str, ...]:
        if intent is None:
            return ()
        slots: list[str] = []
        for token in (intent.subject, intent.action, intent.object):
            if token and token in self.phrase.unigram and token not in slots:
                slots.append(token)
        return tuple(slots)

    def _surface_trace(
        self, index: int, strategy: str, intent: LineIntent | None, text: str
    ) -> SurfaceRealization | None:
        if intent is None:
            return None
        hits = decision_role_hits(text, intent)
        return SurfaceRealization(
            index=index,
            strategy=strategy,
            planned=(intent.subject, intent.action, intent.object),
            realized=tuple(role for role in ("subject", "action", "object") if hits[role]),
        )

    def _grow_opener(
        self, ln: LinePlan, boost: dict[str, float], seed: str, must_include: tuple[str, ...] = ()
    ) -> list[str]:
        openers = self.style.get("top_openers") or list(self.phrase.openers)
        if not openers:
            openers = list(self.phrase.forward)
        choices = [(w, self.phrase.openers.get(w, 1) + 1) for w in openers[:20]]
        start = seeded_weighted_pick(seed, "opener", choices) if choices else "и"
        return self.phrase.grow_forward(
            start, ln.target_syllables, seed, boost=boost, must_include=must_include
        )

    def _choose_rhyme_word(
        self,
        ln: LinePlan,
        boost: dict[str, float],
        seed: str,
        label_keys: dict[str, str],
        used_rhyme_words: set[str],
    ) -> str:
        existing_key = label_keys.get(ln.rhyme_label)
        if existing_key and existing_key in self.rhyme.key_to_words:
            pool = [
                (w, c)
                for w, c in self.rhyme.key_to_words[existing_key]
                if w not in used_rhyme_words
            ]
            if pool:
                return self._boosted_pick(pool, boost, seed, "rhyme_partner")
        # first line of this label: pick a thematically-boosted rhyme word that
        # lives in a group with at least one partner
        candidates: list[tuple[str, int]] = []
        for key, pairs in self.rhyme.key_to_words.items():
            if len(pairs) < 2:
                continue
            for w, c in pairs:
                if w in used_rhyme_words:
                    continue
                b = boost.get(w, 0.0)
                candidates.append((w, int(c * (1 + b * 6)) + 1))
        if not candidates:
            return ln.focus
        return self._boosted_pick(candidates, boost, seed, "rhyme_lead")

    def _boosted_pick(
        self, pool: list[tuple[str, int]], boost: dict[str, float], seed: str, node: str
    ) -> str:
        weighted = [(w, int(c * (1 + boost.get(w, 0.0) * 6)) + 1) for w, c in pool]
        return seeded_weighted_pick(seed, node, weighted)

    # -- activation boost ----------------------------------------------------

    def _activation_boost(self, plan: PoemPlan) -> dict[str, float]:
        if not plan.activated:
            return {}
        top = plan.activated[0][1] or 1.0
        return {c: (a / top) for c, a in plan.activated}

    def _line_boost(self, ln: LinePlan, boost_all: dict[str, float]) -> dict[str, float]:
        boost = dict(boost_all)
        boost[ln.focus] = boost.get(ln.focus, 0.5) + 1.5
        for nb, w in self.graph.neighbors(ln.focus)[:8]:
            boost[nb] = boost.get(nb, 0.0) + 0.4
        return boost

    def _title(self, plan: PoemPlan) -> str:
        theme = plan.theme.strip() or (plan.seed_concepts[0] if plan.seed_concepts else "стихи")
        if plan.style_author:
            author = _genitive(plan.style_author)
            if theme.lower() == plan.style_author.lower():
                return f"В духе {author}"
            return f"{theme.capitalize()} (в духе {author})"
        return theme.capitalize()


_GENITIVE = {
    "Пушкин": "Пушкина", "Лермонтов": "Лермонтова", "Тютчев": "Тютчева",
    "Фет": "Фета", "Блок": "Блока", "Ахматова": "Ахматовой",
}


def _genitive(author: str) -> str:
    return _GENITIVE.get(author, author)


def _format_line(toks: list[str]) -> str:
    if not toks:
        return "…"
    text = " ".join(toks)
    return text[0].upper() + text[1:]
