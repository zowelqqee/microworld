"""Frequency phrase model — a tiny local language model over the corpus.

This is the direct port of the production ``worldpgt/cognition/phrase_graph.py``
idea: learn phrase fragments and word-to-word transitions from local text, then
render by deterministic graph traversal. The mechanism is identical — Counter
frequency tables plus a seeded weighted pick so the same request always renders
identically while different requests diverge. Only the *source* changed: wiki
overlay rows became lines of poetry.

Nothing here invents structure. It stores what the corpus showed (which word
follows which, which words open a line) and walks those transitions. The
seeded-choice primitive ``seeded_weighted_pick`` is a line-for-line port of
``phrase_graph._seeded_weighted_pick``.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256

from poemcore.text import syllables, words


def seeded_weighted_pick(seed: str, node: str, choices: list[tuple[str, int]]) -> str:
    """Deterministic frequency-weighted choice among ``(value, weight)`` pairs.

    Verbatim port of ``worldpgt/cognition/phrase_graph.py``'s
    ``_seeded_weighted_pick``: sort for order-independence, hash the
    ``seed:node`` pair, take the weighted bucket. Determinism is the whole
    point — a prompt is replayable, yet two different prompts (or two lines
    within one poem, keyed by node) spread proportionally to observed
    frequency instead of collapsing onto the single most common word.
    """

    ordered = sorted(choices, key=lambda item: (-item[1], item[0]))
    total = sum(weight for _v, weight in ordered)
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
class PhraseModel:
    """Word-transition frequency graph at orders 1 and 2, plus opener stats.

    The order-2 (trigram) tables are the port of what made QA prose coherent.
    The production ``cognition/phrase_graph.py`` never generated word-by-word:
    it stitched multi-word grammatical *fragments* ("was founded by
    {object_list}"), so local grammar was carried by whole learned spans, not a
    single bigram hop. A pure bigram walk has no such memory — each word is
    chosen from only its immediate neighbour, which is why free-verse lines came
    out word-salad. Order-2 restores that fragment-length context: every word is
    chosen conditioned on the *two* words already placed, so any three
    consecutive words are a span the corpus really used. Novelty still lives at
    the 4-gram level (``seen_4grams`` gate), so the line recombines real 3-word
    fragments without reciting a 4-word one.
    """

    forward: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    backward: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    # order-2 context tables (the "fragment" memory):
    #   forward2[(w1, w2)]  -> Counter over w3  (next word given previous two)
    #   backward2[(w2, w3)] -> Counter over w1  (preceding word given next two)
    forward2: dict[tuple, Counter] = field(default_factory=lambda: defaultdict(Counter))
    backward2: dict[tuple, Counter] = field(default_factory=lambda: defaultdict(Counter))
    openers: Counter = field(default_factory=Counter)
    unigram: Counter = field(default_factory=Counter)
    seen_4grams: set = field(default_factory=set)

    def learn_line(self, line: str) -> None:
        toks = words(line)
        if not toks:
            return
        self.openers[toks[0]] += 1
        for w in toks:
            self.unigram[w] += 1
        for a, b in zip(toks, toks[1:]):
            self.forward[a][b] += 1
            self.backward[b][a] += 1
        for a, b, c in zip(toks, toks[1:], toks[2:]):
            self.forward2[(a, b)][c] += 1
            self.backward2[(b, c)][a] += 1
        for i in range(len(toks) - 3):
            self.seen_4grams.add(tuple(toks[i : i + 4]))

    def to_dict(self) -> dict:
        return {
            "forward": {w: dict(c) for w, c in self.forward.items()},
            "backward": {w: dict(c) for w, c in self.backward.items()},
            "forward2": {"\t".join(k): dict(c) for k, c in self.forward2.items()},
            "backward2": {"\t".join(k): dict(c) for k, c in self.backward2.items()},
            "openers": dict(self.openers),
            "unigram": dict(self.unigram),
            "seen_4grams": ["".join(g) for g in sorted(self.seen_4grams)],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PhraseModel":
        model = cls()
        for w, c in data.get("forward", {}).items():
            model.forward[w] = Counter(c)
        for w, c in data.get("backward", {}).items():
            model.backward[w] = Counter(c)
        for k, c in data.get("forward2", {}).items():
            model.forward2[tuple(k.split("\t"))] = Counter(c)
        for k, c in data.get("backward2", {}).items():
            model.backward2[tuple(k.split("\t"))] = Counter(c)
        model.openers = Counter(data.get("openers", {}))
        model.unigram = Counter(data.get("unigram", {}))
        model.seen_4grams = {tuple(s.split("")) for s in data.get("seen_4grams", [])}
        return model

    # -- traversal -----------------------------------------------------------

    def grow_backward(
        self,
        end_word: str,
        target_syllables: int,
        seed: str,
        *,
        boost: dict[str, float] | None = None,
        max_words: int = 9,
        must_include: tuple[str, ...] = (),
    ) -> list[str]:
        """Build a line *ending* on ``end_word`` by walking backward transitions.

        Backward growth is what lets a rhyme be chosen first (the hard
        constraint) and the line assembled to reach it — the same "plan the
        fixed point, traverse toward it" shape the production renderer uses to
        weave a fact into a learned subordinator. ``boost`` biases the walk
        toward theme words activated by the concept graph, without ever adding
        a word the corpus never placed there.

        ``must_include`` is the intent-seeding hook (empty = identical to
        before). It is an ordered preference list of tokens; at each step, if
        none has been placed yet and one is a *valid corpus predecessor of the
        current head* (i.e. already in the same distribution the walk samples),
        that token is forced in. Because it must be a real predecessor, every
        forced hop stays a genuine corpus transition — grammar and novelty
        gates are unaffected, the line still ends on ``end_word``. If no seed is
        ever a valid predecessor along the walk, nothing is forced (the soft
        fallback handled by the caller).
        """

        line = [end_word]
        boost = boost or {}
        placed = not must_include
        for step in range(max_words - 1):
            if syllables(" ".join(line)) >= target_syllables:
                break
            # order-2 first: what word precedes the two words already at the
            # head of the line? Fall back to the order-1 table only when this
            # exact two-word context was never seen, so most hops keep a real
            # 3-word fragment intact and grammar holds across the join.
            prevs = None
            if len(line) >= 2:
                prevs = self.backward2.get((line[0], line[1]))
            if not prevs:
                prevs = self.backward.get(line[0])
            if not prevs:
                break
            if not placed:
                forced = next((tok for tok in must_include if tok in prevs), None)
                if forced is not None:
                    line.insert(0, forced)
                    placed = True
                    continue
            choices = []
            for w, count in prevs.items():
                weight = count * (1.0 + boost.get(w, 0.0))
                choices.append((w, max(1, int(weight * 4))))
            pick = seeded_weighted_pick(seed, f"bw:{step}:{line[0]}", choices)
            line.insert(0, pick)
        return line

    def grow_forward(
        self,
        start_word: str,
        target_syllables: int,
        seed: str,
        *,
        boost: dict[str, float] | None = None,
        max_words: int = 9,
        must_include: tuple[str, ...] = (),
    ) -> list[str]:
        """Build a line *starting* from ``start_word`` via forward transitions.

        ``must_include`` seeds the same way as ``grow_backward``: force an
        intent token when it is a valid corpus successor of the current tail."""

        line = [start_word]
        boost = boost or {}
        placed = not must_include or start_word in must_include
        for step in range(max_words - 1):
            if syllables(" ".join(line)) >= target_syllables:
                break
            # order-2 first (see grow_backward): what follows the two words
            # already at the tail? Fall back to order-1 only when unseen.
            nexts = None
            if len(line) >= 2:
                nexts = self.forward2.get((line[-2], line[-1]))
            if not nexts:
                nexts = self.forward.get(line[-1])
            if not nexts:
                break
            if not placed:
                forced = next((tok for tok in must_include if tok in nexts), None)
                if forced is not None:
                    line.append(forced)
                    placed = True
                    continue
            choices = []
            for w, count in nexts.items():
                weight = count * (1.0 + boost.get(w, 0.0))
                choices.append((w, max(1, int(weight * 4))))
            pick = seeded_weighted_pick(seed, f"fw:{step}:{line[-1]}", choices)
            line.append(pick)
        return line

    def grow_forward_slots(
        self,
        start_word: str,
        target_syllables: int,
        seed: str,
        *,
        boost: dict[str, float] | None = None,
        slots: tuple[str, ...] = (),
        max_words: int = 9,
        avoid_seen_4grams: bool = False,
    ) -> list[str]:
        """Grow from a decided subject while trying to realize later roles.

        ``slots`` is an ordered semantic commitment, normally
        ``(action, object)``.  A slot is only forced when it is a learned
        successor of the current trigram/bigram context; otherwise the normal
        deterministic walk continues until the context makes that role
        possible.  This makes the decision a genuine constraint on surface
        realization without manufacturing an unobserved transition.
        """

        line = [start_word]
        boost = boost or {}
        remaining = [slot for slot in slots if slot and slot != start_word]
        for step in range(max_words - 1):
            if syllables(" ".join(line)) >= target_syllables:
                break
            nexts = self.forward2.get((line[-2], line[-1])) if len(line) >= 2 else None
            if not nexts:
                nexts = self.forward.get(line[-1])
            if not nexts:
                break
            wanted = remaining[0] if remaining else ""
            if wanted in nexts and self._allows_next(line, wanted, avoid_seen_4grams):
                line.append(wanted)
                remaining.pop(0)
                continue
            # Reach a required role through one observed bridge when it is not
            # adjacent to the current token (``ветер тихо дует``).  The bridge
            # is still selected from the current transition distribution.
            bridge = [
                (word, count)
                for word, count in nexts.items()
                if self._allows_next(line, word, avoid_seen_4grams)
                if wanted in (self.forward2.get((line[-1], word)) or self.forward.get(word, {}))
            ] if wanted else []
            if bridge:
                choices = [
                    (word, max(1, int(count * (1.0 + boost.get(word, 0.0)) * 4)))
                    for word, count in bridge
                ]
                line.append(seeded_weighted_pick(seed, f"slot-fw-bridge:{step}:{line[-1]}", choices))
                continue
            choices = [
                (word, max(1, int(count * (1.0 + boost.get(word, 0.0)) * 4)))
                for word, count in nexts.items()
                if self._allows_next(line, word, avoid_seen_4grams)
            ]
            if not choices:
                break
            pick = seeded_weighted_pick(seed, f"slot-fw:{step}:{line[-1]}", choices)
            line.append(pick)
            if remaining and pick == remaining[0]:
                remaining.pop(0)
        return line

    def _allows_next(self, line: list[str], word: str, avoid_seen_4grams: bool) -> bool:
        return not avoid_seen_4grams or len(line) < 3 or tuple(line[-3:] + [word]) not in self.seen_4grams

    def grow_backward_slots(
        self,
        end_word: str,
        target_syllables: int,
        seed: str,
        *,
        boost: dict[str, float] | None = None,
        slots: tuple[str, ...] = (),
        max_words: int = 9,
    ) -> list[str]:
        """Rhyme-first counterpart to :meth:`grow_forward_slots`.

        Backward traversal encounters an SVO decision in reverse order.  It
        therefore first seeks the object, then action, then subject while
        preserving corpus-learned predecessor transitions.
        """

        line = [end_word]
        boost = boost or {}
        remaining = [slot for slot in reversed(slots) if slot and slot != end_word]
        for step in range(max_words - 1):
            if syllables(" ".join(line)) >= target_syllables:
                break
            prevs = self.backward2.get((line[0], line[1])) if len(line) >= 2 else None
            if not prevs:
                prevs = self.backward.get(line[0])
            if not prevs:
                break
            wanted = remaining[0] if remaining else ""
            if wanted in prevs:
                line.insert(0, wanted)
                remaining.pop(0)
                continue
            bridge = [
                (word, count)
                for word, count in prevs.items()
                if wanted in (self.backward2.get((word, line[0])) or self.backward.get(word, {}))
            ] if wanted else []
            if bridge:
                choices = [
                    (word, max(1, int(count * (1.0 + boost.get(word, 0.0)) * 4)))
                    for word, count in bridge
                ]
                line.insert(0, seeded_weighted_pick(seed, f"slot-bw-bridge:{step}:{line[0]}", choices))
                continue
            choices = [
                (word, max(1, int(count * (1.0 + boost.get(word, 0.0)) * 4)))
                for word, count in prevs.items()
            ]
            pick = seeded_weighted_pick(seed, f"slot-bw:{step}:{line[0]}", choices)
            line.insert(0, pick)
            if remaining and pick == remaining[0]:
                remaining.pop(0)
        return line

    def link_score(self, start: str, target: str) -> int:
        """Return a small corpus-derived score for a 1- or 2-hop word link.

        This is not a longer n-gram model: it only queries the existing
        learned transition tables.  The reasoning layer uses it to prefer an
        action that the renderer can actually reach from its chosen roles.
        """

        if not start or not target:
            return 0
        direct = self.forward.get(start, {}).get(target, 0)
        via_one = sum(
            min(count, 3) * self.forward.get(middle, {}).get(target, 0)
            for middle, count in self.forward.get(start, {}).items()
        )
        return direct * 4 + via_one

    def slot_link_score(self, start: str, target: str) -> int:
        """Reachability under the exact contexts used by slot growth.

        Unlike :meth:`link_score`, the two-hop case requires a real trigram
        ``(start, bridge, target)``.  The reasoning layer uses this stricter
        query before committing to an action, so planning and realization obey
        the same transition contract.
        """

        if not start or not target:
            return 0
        direct = self.forward.get(start, {}).get(target, 0)
        via_bridge = sum(
            min(count, 3) * self.forward2.get((start, bridge), {}).get(target, 0)
            for bridge, count in self.forward.get(start, {}).items()
        )
        return direct * 4 + via_bridge

    def contains_seen_4gram(self, toks: list[str]) -> bool:
        for i in range(len(toks) - 3):
            if tuple(toks[i : i + 4]) in self.seen_4grams:
                return True
        return False
