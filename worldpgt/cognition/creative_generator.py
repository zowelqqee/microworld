"""Creative generation mode — the free-recombination counterpart to factual QA.

This is the layer split the poetry_lab experiment isolated: the factual path
answers *only* from grounded overlay facts and audits on a gap, while the
creative path runs the **inverted accept/reject gate** — generate freely by
recombining learned word transitions, and allow output only when it does NOT
reproduce a corpus 4-gram (recombine, never recite).

It is a token-level local language model ported from
``poetry_lab/poemcore/phrase_model.py`` and ``novelty.py``:

* order-1 and order-2 (bigram/trigram) word-transition frequency tables, so any
  three consecutive generated words are a span the corpus really used;
* a seeded deterministic weighted pick (identical primitive to
  ``phrase_graph._seeded_weighted_pick``), so a request is replayable yet two
  requests diverge proportionally to observed frequency;
* a 4-gram novelty gate — the whole point of the creative gate: output must
  recombine real 3-word fragments without reciting a 4-word one.

It trains from the same local English prose the factual system already ingests
(snapshot lead paragraphs, community context), never from live text at request
time. It asserts no grounded fact: the orchestrator labels every output as
creative/generated and only reaches this module *after* the hard-safety screens,
so it can never fabricate private or current-sensitive claims about real people.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Iterable

_ROOT = Path(__file__).resolve().parents[2]
_EXPERIMENTS = _ROOT / "worldpgt" / "experiments"
_DEFAULT_SNAPSHOT_DIR = _EXPERIMENTS / "wiki_snapshots_v1" / "normalized_docs"
_DEFAULT_COMMUNITY_CONTEXT_PATHS = (
    _EXPERIMENTS / "community_context_v1" / "reddit_community_context.json",
)

_WORD_RE = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
# Snapshot lead files carry a `Key: value` metadata header before the prose
# body; those lines are not natural language and must not train the model.
_SNAPSHOT_META_KEYS = frozenset({
    "Source", "Retrieved at", "Revision ID", "Raw text SHA256", "Status",
    "Safe for accepted memory",
    "Requires ingestion/quarantine/promotion/regression",
})

# Function words that make poor *starts* and poor concept seeds, but are kept in
# the transition tables (they carry the grammar between content words).
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "that", "which", "who", "whom", "whose", "this", "these", "those", "it", "its",
    "he", "she", "they", "them", "his", "her", "their", "we", "you", "i", "not",
    "no", "yes", "do", "does", "did", "has", "have", "had", "will", "would", "can",
    "could", "may", "might", "shall", "should", "must", "than", "then", "so", "if",
    "into", "over", "under", "about", "also", "such", "more", "most", "some", "any",
})

# Dangling trailing tokens that make a trimmed sentence read as unfinished.
_INCOMPLETE_TAIL = _STOPWORDS | frozenset({
    "e.g", "i.e", "etc", "per", "via", "upon", "among", "between", "within",
    "from", "during", "after", "before", "since", "until", "while", "because",
    "although", "though", "whether", "nor", "yet", "onto", "off", "out", "up",
    "down", "each", "every", "both", "either", "neither", "many", "much", "few",
})

_MIN_SENTENCE_WORDS = 5
_MAX_SENTENCE_WORDS = 18
_REROLL_ATTEMPTS = 8


def seeded_weighted_pick(seed: str, node: str, choices: list[tuple[str, int]]) -> str:
    """Deterministic frequency-weighted choice — the poetry_lab / phrase_graph
    primitive verbatim: sort for order-independence, hash ``seed:node``, take the
    weighted bucket."""

    ordered = sorted(choices, key=lambda item: (-item[1], item[0]))
    total = sum(weight for _value, weight in ordered)
    if total <= 0:
        return ordered[0][0]
    value = int(sha256(f"{seed}:{node}".encode("utf-8")).hexdigest()[:12], 16)
    pick = value % total
    acc = 0
    for option, weight in ordered:
        acc += weight
        if pick < acc:
            return option
    return ordered[-1][0]


@dataclass
class CreativeModel:
    """Order-1/order-2 word-transition tables plus opener stats and a 4-gram set."""

    forward: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    forward2: dict[tuple, Counter] = field(default_factory=lambda: defaultdict(Counter))
    openers: Counter = field(default_factory=Counter)
    unigram: Counter = field(default_factory=Counter)
    seen_4grams: set = field(default_factory=set)

    def learn_sentence(self, sentence: str) -> None:
        toks = _WORD_RE.findall(sentence.lower())
        if len(toks) < 2:
            return
        self.openers[toks[0]] += 1
        for token in toks:
            self.unigram[token] += 1
        for first, second in zip(toks, toks[1:]):
            self.forward[first][second] += 1
        for first, second, third in zip(toks, toks[1:], toks[2:]):
            self.forward2[(first, second)][third] += 1
        for i in range(len(toks) - 3):
            self.seen_4grams.add(tuple(toks[i : i + 4]))

    @property
    def trained(self) -> bool:
        return len(self.unigram) >= 200

    def contains_seen_4gram(self, toks: list[str]) -> bool:
        return any(
            tuple(toks[i : i + 4]) in self.seen_4grams for i in range(len(toks) - 3)
        )

    # -- traversal ---------------------------------------------------------- #

    def _allows_next(self, line: list[str], word: str) -> bool:
        """The step-level novelty gate: a next word is allowed only if it does
        not complete a corpus 4-gram with the last three placed words. Steering
        the walk away from recitation *during* growth (not just rejecting at the
        end) is what makes novel completions reachable in dense factual prose —
        the same ``avoid_seen_4grams`` mechanism poetry_lab used to hit ~0.99
        novelty."""

        return len(line) < 3 or tuple(line[-3:] + [word]) not in self.seen_4grams

    def grow_forward(
        self, start: str, seed: str, *, boost: dict[str, float], max_words: int
    ) -> list[str]:
        """Grow a clause from ``start`` via order-2 (falling back to order-1)
        transitions, biased toward the boosted concept words and steered away
        from seen 4-grams at every step. No transition is ever manufactured — a
        word is only placed if the corpus placed it after this context."""

        line = [start]
        for step in range(max_words - 1):
            nexts = None
            if len(line) >= 2:
                nexts = self.forward2.get((line[-2], line[-1]))
            if not nexts:
                nexts = self.forward.get(line[-1])
            if not nexts:
                break
            novel = [
                (word, max(1, int(count * (1.0 + boost.get(word, 0.0)) * 4)))
                for word, count in nexts.items()
                if self._allows_next(line, word)
            ]
            if not novel:
                # Every continuation would recite a 4-gram; stop here rather
                # than break novelty. The trimmed clause is still emitted if it
                # is long enough.
                break
            pick = seeded_weighted_pick(seed, f"fw:{step}:{line[-1]}", novel)
            line.append(pick)
        return line


def _complete(tokens: list[str]) -> list[str]:
    """Trim a dangling function-word tail so the clause does not read cut off."""

    out = list(tokens)
    while len(out) > _MIN_SENTENCE_WORDS and out[-1] in _INCOMPLETE_TAIL:
        out.pop()
    return out


def _boost_field(model: CreativeModel, seeds: list[str]) -> dict[str, float]:
    """Lightweight spreading activation over the token graph: the seed words get
    the strongest bias, their frequent forward neighbours a weaker one. This is
    the creative analogue of poetry_lab's concept-graph activation, read off the
    transition tables the model already learned instead of a separate graph."""

    boost: dict[str, float] = {}
    for word in seeds:
        if word in model.unigram:
            boost[word] = boost.get(word, 0.0) + 6.0
            for neighbour, count in model.forward.get(word, {}).items():
                if neighbour not in _STOPWORDS:
                    boost[neighbour] = boost.get(neighbour, 0.0) + min(count, 4) * 0.5
    return boost


def _sentence_start(
    model: CreativeModel, seeds: list[str], boost: dict[str, float], seed: str
) -> str:
    """Prefer a seed word that can actually open a clause, else a corpus opener
    biased toward the concept field."""

    seed_starts = [w for w in seeds if w in model.forward and w not in _STOPWORDS]
    if seed_starts:
        choices = [(w, max(1, model.unigram.get(w, 1))) for w in seed_starts]
        return seeded_weighted_pick(seed, "start:seed", choices)
    openers = [
        (word, max(1, int(count * (1.0 + boost.get(word, 0.0)))))
        for word, count in model.openers.items()
        if word not in _STOPWORDS
    ]
    if not openers:
        openers = [(word, count) for word, count in model.openers.items()]
    if not openers:
        return next(iter(model.unigram), "")
    return seeded_weighted_pick(seed, "start:opener", openers)


def _render_sentence(tokens: list[str]) -> str:
    text = " ".join(tokens).strip()
    if not text:
        return ""
    text = text[0].upper() + text[1:]
    return text if text[-1] in ".!?" else text + "."


def generate(
    model: CreativeModel, seeds: list[str], seed: str, *, num_sentences: int = 3
) -> str | None:
    """Recombine learned transitions into a short creative passage.

    Deterministic in ``seed``; biased toward ``seeds`` (the concept field);
    every emitted sentence passes the 4-gram novelty gate (recombine, not
    recite). Returns ``None`` when the model is untrained or nothing novel could
    be grown, so the caller can degrade honestly instead of emitting filler.
    """

    if not model.trained:
        return None
    boost = _boost_field(model, seeds)
    sentences: list[str] = []
    seen: set[tuple[str, ...]] = set()
    for index in range(max(1, num_sentences)):
        chosen: list[str] | None = None
        for attempt in range(_REROLL_ATTEMPTS):
            attempt_seed = f"{seed}|s{index}|a{attempt}"
            start = _sentence_start(model, seeds, boost, attempt_seed)
            if not start:
                break
            grown = _complete(
                model.grow_forward(
                    start, attempt_seed, boost=boost, max_words=_MAX_SENTENCE_WORDS
                )
            )
            key = tuple(grown)
            if (
                len(grown) >= _MIN_SENTENCE_WORDS
                and key not in seen
                and not model.contains_seen_4gram(grown)
            ):
                chosen = grown
                seen.add(key)
                break
        if chosen:
            sentences.append(_render_sentence(chosen))
    if not sentences:
        return None
    return " ".join(sentences)


# -- training from local prose --------------------------------------------- #

def _snapshot_body_sentences(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    body: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if ":" in stripped and stripped.split(":", 1)[0] in _SNAPSHOT_META_KEYS:
            continue
        body.append(stripped)
    return _SENTENCE_SPLIT_RE.split(" ".join(body))


def _community_sentences(path: Path) -> list[str]:
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    rows = data if isinstance(data, list) else data.get("items", [])
    if not isinstance(rows, list):
        return []
    out: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get("trust") or "") != "community_context_only":
            continue
        text = " ".join(
            part for part in (str(row.get("title") or ""), str(row.get("text") or "")) if part
        )
        out.extend(_SENTENCE_SPLIT_RE.split(text))
    return out


def build_creative_model(
    snapshot_dir: Path | None = None,
    community_context_paths: Iterable[Path] | None = None,
) -> CreativeModel:
    model = CreativeModel()
    resolved_dir = snapshot_dir if snapshot_dir is not None else _DEFAULT_SNAPSHOT_DIR
    if resolved_dir.is_dir():
        for path in sorted(resolved_dir.glob("*.md")):
            for sentence in _snapshot_body_sentences(path):
                model.learn_sentence(sentence)
    resolved_ctx = (
        _DEFAULT_COMMUNITY_CONTEXT_PATHS
        if community_context_paths is None
        else community_context_paths
    )
    for path in resolved_ctx:
        for sentence in _community_sentences(Path(path)):
            model.learn_sentence(sentence)
    return model


@lru_cache(maxsize=1)
def default_creative_model() -> CreativeModel:
    return build_creative_model()


def seed_concepts(question: str, model: CreativeModel, *, limit: int = 4) -> list[str]:
    """Content words from the request that the model actually knows, in order."""

    seeds: list[str] = []
    for token in _WORD_RE.findall((question or "").lower()):
        if (
            token not in _STOPWORDS
            and len(token) >= 3
            and token in model.unigram
            and token not in seeds
        ):
            seeds.append(token)
        if len(seeds) >= limit:
            break
    return seeds


def generate_creative(
    question: str, *, seed: str, num_sentences: int = 3, model: CreativeModel | None = None
) -> str | None:
    """Top-level creative entry: seed from the request, recombine, novelty-gate."""

    model = model or default_creative_model()
    seeds = seed_concepts(question, model)
    return generate(model, seeds, seed, num_sentences=num_sentences)
