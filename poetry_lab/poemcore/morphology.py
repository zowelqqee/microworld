"""Morphological agreement — learned from the corpus, not from a dictionary.

The entity-type transfer fixed *role* selection (who acts, what is a setting).
It could not fix agreement: "Маргарита погубил одного" picks the right subject
and a real corpus verb, but the verb is masculine and the subject is not.

Russian makes this tractable without a morphological dictionary or a neural
tagger, because **the past tense marks gender on its own surface**:

    -л / -лся      masculine    ("сказал", "поднялся")
    -ла / -лась    feminine     ("сказала", "поднялась")
    -ло / -лось    neuter       ("сказало")
    -ли / -лись    plural       ("сказали")

So gender agreement is checkable from spelling alone. The remaining unknown is
the *subject's* gender — and that is learned the same way ``entity_types.py``
learns whether a name is a person or a place: by counting what the corpus put
next to it. A name that repeatedly governs "-ла" verbs is feminine.

Discipline, unchanged from every other layer here:

* no neural model, no external dictionary, no new corpus;
* evidence below a threshold yields ``None``, and ``None`` constrains nothing
  (advisory, never a refusal — the same rule the ported QA knowledge layer
  follows);
* an ending heuristic is only a *fallback* when the corpus stayed silent.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

# Gender/number codes used throughout: masculine, feminine, neuter, plural.
Gender = str  # "m" | "f" | "n" | "p"

# Past-tense surface endings, longest first so "-лась" wins over "-ла".
_PAST_ENDINGS: tuple[tuple[str, Gender], ...] = (
    ("лись", "p"), ("лась", "f"), ("лось", "n"), ("лся", "m"),
    ("ли", "p"), ("ла", "f"), ("ло", "n"), ("л", "m"),
)

_INFINITIVE = re.compile(r"(?:ться|тись|ть|ти)$")

# Present/future 3rd person, where number (not gender) is what must agree.
_PRESENT_SG = re.compile(r"(?:ает|яет|ует|еет|ит|ёт|ет)$")
_PRESENT_PL = re.compile(r"(?:ают|яют|уют|еют|ат|ят|ут|ют)$")

# Nouns/verbs that end in -ла/-ло/-ли without being past-tense verbs. Small and
# corpus-specific: these are the frequent false positives in this novel.
_NOT_PAST = frozenset({
    "села", "тела", "дела", "мела", "смыслы", "было", "жерла",
    "числа", "крыла", "масла", "весла", "зеркала", "покрывала", "одеяла",
})

# Minimum agreement observations before a learned gender is trusted, and the
# share of them that must agree. Same conservative bar as entity_types.py.
_MIN_EVIDENCE = 3
_DOMINANCE = 0.6


def past_gender(word: str) -> Gender | None:
    """Gender/number a past-tense verb marks on its own surface, else None."""

    if len(word) < 3 or word in _NOT_PAST:
        return None
    for ending, gender in _PAST_ENDINGS:
        if word.endswith(ending):
            # "-л" alone is only a verb ending on a long enough stem; "мел",
            # "стол", "угол" are nouns. Require a vowel before the final -л.
            if ending == "л" and not re.search(r"[аяеёиоуыэю]л$", word):
                return None
            return gender
    return None


def is_infinitive(word: str) -> bool:
    return bool(_INFINITIVE.search(word)) and len(word) >= 4


def is_finite_verb(word: str) -> bool:
    """A verb that can head a clause — i.e. a real predicate, not an infinitive."""

    if is_infinitive(word):
        return False
    if past_gender(word) is not None:
        return True
    return bool(_PRESENT_SG.search(word) or _PRESENT_PL.search(word)) and len(word) >= 4


def gender_by_ending(word: str) -> Gender | None:
    """Fallback gender for a noun/name from its nominative ending.

    Only used when the corpus gave no agreement evidence. Deliberately refuses
    to guess on the ambiguous cases rather than assert a wrong gender.
    """

    if not word or len(word) < 3:
        return None
    if word.endswith(("ость", "знь", "чь", "щь")):
        return "f"
    if word.endswith(("а", "я")):
        return "f"          # Маргарита, Гелла (misses "папа"/"дядя": rare here)
    if word.endswith(("о", "е")):
        return "n"
    if word.endswith(("ы", "и")):
        return None          # plural or genitive — not decidable
    if re.search(r"[бвгджзклмнпрстфхцчшщ]$", word):
        return "m"          # Пилат, Воланд, Бегемот, Коровьев
    return None


@dataclass
class GenderEvidence:
    """Per-word counts of the past-tense gender observed agreeing with it."""

    counts: dict[str, Counter] = field(default_factory=dict)

    def observe(self, word: str, gender: Gender) -> None:
        self.counts.setdefault(word, Counter())[gender] += 1

    def to_dict(self) -> dict:
        return {word: dict(counter) for word, counter in self.counts.items()}

    @classmethod
    def from_dict(cls, data: dict) -> "GenderEvidence":
        ev = cls()
        ev.counts = {word: Counter(counter) for word, counter in data.items()}
        return ev


def learned_gender(word: str, evidence: GenderEvidence) -> Gender | None:
    counter = evidence.counts.get(word)
    if not counter:
        return None
    total = sum(counter.values())
    if total < _MIN_EVIDENCE:
        return None
    gender, count = counter.most_common(1)[0]
    return gender if count / total >= _DOMINANCE else None


def build_gender_map(words: list[str], evidence: GenderEvidence) -> dict[str, Gender]:
    """Corpus-learned gender first; ending heuristic only where it stayed silent.

    Words whose gender remains undecidable are simply absent from the map, and
    an absent word imposes no agreement constraint downstream.
    """

    genders: dict[str, Gender] = {}
    for word in words:
        gender = learned_gender(word, evidence) or gender_by_ending(word)
        if gender:
            genders[word] = gender
    return genders


def verified_past_gender(word: str, noun_like: frozenset[str] = frozenset()) -> Gender | None:
    """``past_gender`` plus a corpus check that the word really is a past verb.

    Surface endings alone cannot separate a masculine past verb from a noun:
    "сказал" and "стол" both end in vowel+л, "видел" and "мел" both in -ел.
    Rather than hand-list the nouns, ask the corpus for the property that
    separates the two classes — **a noun follows a preposition, a finite verb
    does not**. In this corpus "стол"/"угол"/"пол" appear right after в/на/под
    7/7/29 times; "сказал"/"погубил"/"вступил" appear there zero times.

    ``noun_like`` is that evidence, collected in ``ingest``. It is preferred
    over a morphological test such as "does a plural past form exist?", which
    silently fails on hapaxes: "погубил" occurs once and has no "погубили" —
    exactly the verb whose disagreement this whole layer exists to catch.

    Same rule as everywhere else here: learn the distinction from the corpus
    rather than import a dictionary.
    """

    if word in noun_like:
        return None
    return past_gender(word)


def agreement_ok(
    subject: str, verb: str, genders: dict[str, Gender],
    noun_like: frozenset[str] = frozenset(),
) -> bool:
    """False only when both genders are known *and* they disagree.

    Unknown subject gender, or a token that is not a verified past verb
    (present tense, infinitive, or a noun that merely looks like one), returns
    True: this layer never rejects on ignorance.
    """

    verb_gender = verified_past_gender(verb, noun_like)
    if verb_gender is None:
        return True
    subject_gender = genders.get(subject)
    if subject_gender is None:
        return True
    return subject_gender == verb_gender


def sentence_agreement_errors(
    subject: str, tokens: list[str], genders: dict[str, Gender],
    noun_like: frozenset[str] = frozenset(),
) -> int:
    """Count verified past-tense verbs in *tokens* that disagree with *subject*."""

    if subject not in genders:
        return 0
    return sum(
        1 for token in tokens
        if not agreement_ok(subject, token, genders, noun_like)
    )
