"""Closed reference grammar for the dialogue context layer.

Every reference *form* the resolver understands is enumerated here as a
syntactic pattern — pronouns, typed demonstratives, role descriptors,
contrastives, selectives, elliptical subjects and topic-shift phrasings. A
form not in this file is *structurally* unresolvable and audits; it is never
fuzzy-matched.

The *vocabulary* inside those forms (which nouns denote a type, which nouns
denote a role) is not a private word list: it is delegated to
:mod:`worldpgt.dialogue.type_lexicon`, which reuses the same classifiers the
rest of Microworld already uses to type entities and relations — so "the
founder"/"that company" stay in sync with however ``EntitySurfaceIndex`` and
``relation_policy`` already understand those words, instead of drifting in a
second, uncoordinated copy. See ``type_lexicon`` for why a fuzzy/embedding
fallback is deliberately *not* used for this — it produces real false
positives on ordinary abstract nouns.

Detection is a single left-to-right scan with longest-match span dedup (the
same overlap policy the v1 resolver used). No parsing model, no similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from worldpgt.dialogue.type_lexicon import classify_reference_noun
from worldpgt.knowledge.entity_types import CANONICAL_ENTITY_TYPES
from worldpgt.relation_extraction_v2.relation_policy import relation_intent_from_text

# ── Reference classes ────────────────────────────────────────────────────────
PRONOUN_PERSON = "pronoun_person"
PRONOUN_THING = "pronoun_thing"
PRONOUN_PLURAL = "pronoun_plural"
DEMONSTRATIVE_TYPED = "demonstrative_typed"
DEMONSTRATIVE_BARE = "demonstrative_bare"
ROLE_DESCRIPTOR = "role_descriptor"
CONTRASTIVE = "contrastive"
SELECTIVE = "selective"
ELLIPTICAL = "elliptical"

# ── Type gates (hard filters; None means "any known type") ──────────────────
PERSON_TYPES = frozenset({"person"})
# "it" may denote anything that is not a person.
THING_TYPES = frozenset(CANONICAL_ENTITY_TYPES - {"person"})
# A role's holder isn't always a person (Starlink's "owner" is SpaceX, an
# organization) — the role_relation itself does the narrowing, not the gate.
_ANY_TYPE = None

_POSSESSIVES = frozenset({"his", "her", "its", "their", "его", "её", "их"})

_PERSON_PRONOUNS = frozenset({"he", "she", "his", "her", "him", "он", "она", "его", "её"})
_THING_PRONOUNS = frozenset({"it", "its", "оно"})
_PLURAL_PRONOUNS = frozenset({"they", "their", "them", "они", "их"})

# ── Token patterns ───────────────────────────────────────────────────────────

_PRONOUN_RE = re.compile(
    r"(?<!\w)(he|she|his|her|him|it|its|they|their|them|он|она|оно|они|его|её|их)(?!\w)",
    re.IGNORECASE,
)
# One noun-referring determiner class per group: group 1 (this/that/the same)
# is eligible for both role and type classification; group 2 (bare "the") is
# role-only — an unrecognized bare "the NOUN" is extremely common ordinary
# English ("the fact", "the answer") and must not become a typed-demonstrative
# slot just because some type word happens to follow "the".
# The trailing (?!\s+of\b) guard skips self-contained questions like "the
# founder of PayPal" — the anchor is named explicitly right there, so this is
# not an elliptical dialogue reference at all; ordinary parsing handles it.
_NOUN_REF_RE = re.compile(
    r"(?<!\w)(?:(this|that|the same|эта|та|это|тот)|(the|этот|тот))\s+"
    r"([a-zа-яё][a-zа-яё-]*)(?!\w)(?!\s+of\b)(?!\s+от\b)",
    re.IGNORECASE,
)
_CONTRASTIVE_RE = re.compile(
    r"(?<!\w)the\s+other\s+(?:(one)|([a-zа-яё][a-zа-яё-]*))(?!\w)"
    r"|(?<!\w)друг(?:ая|ой|ую)\s+([a-zа-яё-]+)(?!\w)",
    re.IGNORECASE,
)
_SELECTIVE_RE = re.compile(
    r"^\s*(?:which|какая|какой|кто)\s+"
    r"(?:(one|of\s+them|of\s+those|of\s+the\s+two|из\s+них)|([a-zа-яё][a-zа-яё-]*))(?!\w)",
    re.IGNORECASE,
)
# Bare demonstratives only where they cannot be relative pronouns: at the
# start of the question or as a prepositional object ("tell me about that").
_DEMONSTRATIVE_BARE_RE = re.compile(
    r"(?:^\s*|\b(?:about|of|for|with|про|о|об)\s+)(this|that|это|этом|этой)(?!\w)(?!\s+\w)",
    re.IGNORECASE,
)
_TOPIC_SHIFT_RE = re.compile(
    r"^\s*(?:and|what\s+about|how\s+about|а|а\s+что\s+про|а\s+про|что\s+про)\s+(.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)
# "what else / which other / other than" — exclusion modifier for the planner.
_EXCLUSION_RE = re.compile(
    r"(?<!\w)(?:what|who|which)\s+(?:else|other)(?!\w)|(?<!\w)besides(?!\w)|"
    r"(?<!\w)(?:что|кто)\s+ещё(?!\w)|(?<!\w)кроме(?!\w)",
    re.IGNORECASE,
)

# Russian relation keywords for ellipsis detection only. Deliberately a
# private supplement: extending relation_policy's KEYWORD_MAP would change
# extraction and query understanding globally, which is not this layer's call.
_RU_RELATION_KEYWORDS: dict[str, str] = {
    "основал": "founded_by",
    "создал": "founded_by",
    "владеет": "owned_by",
    "принадлежит": "owned_by",
    "разрабатывает": "develops",
    "производит": "produces",
    "возглавляет": "leader_of",
}


def _relation_intent(text: str) -> str | None:
    intent = relation_intent_from_text(text)
    if intent is not None:
        return intent
    lowered = (text or "").lower()
    for keyword, relation in _RU_RELATION_KEYWORDS.items():
        if re.search(r"(?<!\w)" + re.escape(keyword) + r"(?!\w)", lowered):
            return relation
    return None


# Tokens that carry no content for ellipsis-residue analysis.
_ELLIPSIS_STOPWORDS = frozenset({
    "who", "what", "which", "whom", "whose", "did", "does", "do", "is", "are",
    "was", "were", "has", "have", "had", "the", "a", "an", "and", "then",
    "so", "ok", "okay", "кто", "что", "чем", "а", "и", "же", "ну",
})

_WORD_RE = re.compile(r"[a-zа-яё]+(?:-[a-zа-яё]+)*", re.IGNORECASE)


@dataclass(frozen=True)
class ReferenceSlot:
    surface: str
    start: int
    end: int
    ref_class: str
    type_gate: frozenset[str] | None  # None = any known type
    role_relation: str | None = None  # role_descriptor only
    possessive: bool = False

    def to_dict(self) -> dict:
        return {
            "surface": self.surface,
            "start": self.start,
            "end": self.end,
            "ref_class": self.ref_class,
            "type_gate": sorted(self.type_gate) if self.type_gate is not None else None,
            "role_relation": self.role_relation,
            "possessive": self.possessive,
        }


@dataclass(frozen=True)
class GrammarParse:
    slots: tuple[ReferenceSlot, ...]
    topic_shift_surface: str | None = None  # raw tail of "what about X"
    has_exclusion: bool = False
    relation_intent: str | None = None  # exact-map hit on the raw question


def detect_slots(question: str, index) -> GrammarParse:
    """Scan *question* for reference slots. ``index`` is any object with the
    ``EntitySurfaceIndex`` read interface (used only to keep entity surfaces
    from being misread as references, and for ellipsis residue analysis)."""

    text = question or ""
    has_exclusion = bool(_EXCLUSION_RE.search(text))
    relation_intent = _relation_intent(text)

    topic_match = _TOPIC_SHIFT_RE.match(text)
    if topic_match:
        tail = topic_match.group(1).strip()
        # Only a bare known entity is a topic shift; anything else falls
        # through to the normal slot scan ("what about his brother?").
        if index.resolve(tail) is not None:
            return GrammarParse(
                slots=(),
                topic_shift_surface=tail,
                has_exclusion=has_exclusion,
                relation_intent=relation_intent,
            )

    entity_spans = [
        (start, end) for _s, _c, start, end in index.find_in_text(text)
    ]

    candidates: list[ReferenceSlot] = []

    for m in _NOUN_REF_RE.finditer(text):
        role_and_type_eligible = m.group(1) is not None
        noun = m.group(3)
        classified = classify_reference_noun(noun)
        if classified is None:
            continue
        kind, value = classified
        if kind == "role":
            candidates.append(ReferenceSlot(
                surface=m.group(0), start=m.start(), end=m.end(),
                ref_class=ROLE_DESCRIPTOR, type_gate=_ANY_TYPE, role_relation=value,
            ))
        elif kind == "type" and role_and_type_eligible:
            candidates.append(ReferenceSlot(
                surface=m.group(0), start=m.start(), end=m.end(),
                ref_class=DEMONSTRATIVE_TYPED, type_gate=frozenset({value}),
            ))
        # A bare "the NOUN" that only classifies as a type (not a role) is
        # ordinary definite English ("the price"), not a dialogue reference.

    for m in _CONTRASTIVE_RE.finditer(text):
        noun = m.group(1) or m.group(2) or m.group(3) or "one"
        gate = _type_gate_only(noun) if noun != "one" else None
        candidates.append(ReferenceSlot(
            surface=m.group(0), start=m.start(), end=m.end(),
            ref_class=CONTRASTIVE, type_gate=gate,
        ))

    # A selective reference ranges over *dialogue* candidates ("which one…?").
    # A question that names an entity ("Which company owns Starlink?") ranges
    # over the graph instead and belongs to the ordinary pipeline.
    if not entity_spans:
        m = _SELECTIVE_RE.match(text)
        if m:
            noun = m.group(2) or ""
            gate = _type_gate_only(noun) if noun else None
            candidates.append(ReferenceSlot(
                surface=m.group(0).strip(), start=m.start(), end=m.end(),
                ref_class=SELECTIVE, type_gate=gate,
            ))

    for m in _PRONOUN_RE.finditer(text):
        token = m.group(1).lower()
        if token in _PERSON_PRONOUNS:
            ref_class, gate = PRONOUN_PERSON, PERSON_TYPES
        elif token in _THING_PRONOUNS:
            ref_class, gate = PRONOUN_THING, THING_TYPES
        else:
            ref_class, gate = PRONOUN_PLURAL, None
        candidates.append(ReferenceSlot(
            surface=m.group(1), start=m.start(1), end=m.end(1),
            ref_class=ref_class, type_gate=gate,
            possessive=token in _POSSESSIVES,
        ))

    for m in _DEMONSTRATIVE_BARE_RE.finditer(text):
        candidates.append(ReferenceSlot(
            surface=m.group(1), start=m.start(1), end=m.end(1),
            ref_class=DEMONSTRATIVE_BARE, type_gate=None,
        ))

    slots = _dedupe_spans(candidates, entity_spans)

    if not slots and not entity_spans and relation_intent is not None:
        if _is_elliptical_residue(text, relation_intent):
            slots = (ReferenceSlot(
                surface="", start=0, end=0,
                ref_class=ELLIPTICAL, type_gate=None,
                role_relation=relation_intent,
            ),)

    return GrammarParse(
        slots=slots,
        topic_shift_surface=None,
        has_exclusion=has_exclusion,
        relation_intent=relation_intent,
    )


def _type_gate_only(noun: str) -> frozenset[str] | None:
    """Type gate for contrastive/selective nouns — these only ever narrow by
    *type* ("the other company"), never by role; a noun that classifies as a
    role (or as nothing) leaves the gate unset (inherits from context)."""

    classified = classify_reference_noun(noun)
    if classified is not None and classified[0] == "type":
        return frozenset({classified[1]})
    return None


def _dedupe_spans(
    candidates: list[ReferenceSlot],
    entity_spans: list[tuple[int, int]],
) -> tuple[ReferenceSlot, ...]:
    """Longest-match-first overlap resolution; drop anything inside a known
    entity surface span (e.g. the "it" inside a title)."""

    def _inside_entity(slot: ReferenceSlot) -> bool:
        return any(s <= slot.start and slot.end <= e for s, e in entity_spans)

    ordered = sorted(candidates, key=lambda c: (c.start, -(c.end - c.start)))
    out: list[ReferenceSlot] = []
    covered_until = -1
    for slot in ordered:
        if slot.start < covered_until:
            continue
        if _inside_entity(slot):
            continue
        out.append(slot)
        covered_until = slot.end
    return tuple(out)


def _is_elliptical_residue(text: str, relation_intent: str) -> bool:
    """True when nothing of substance remains after removing interrogative
    scaffolding and the relation keyword — i.e. "Who founded?" but not
    "Who founded the first private space company?"."""

    # Every content token must belong to the relation phrase itself; any other
    # content word ("the first private space company") means this is a full
    # question with an unknown subject, not an ellipsis.
    content = [
        t for t in (tok.lower() for tok in _WORD_RE.findall(text))
        if t not in _ELLIPSIS_STOPWORDS and _relation_intent(t) != relation_intent
    ]
    return not content
