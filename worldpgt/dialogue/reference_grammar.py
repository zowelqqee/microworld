"""Closed reference grammar for the dialogue context layer.

Every reference form the resolver understands is enumerated here — pronouns,
typed demonstratives, role descriptors, contrastives, selectives, elliptical
subjects and topic-shift phrasings. A form not in this file is *structurally*
unresolvable and audits; it is never fuzzy-matched. Type gates are hard
filters applied before scoring, so e.g. "he" can never reach an organization
at any salience.

Detection is a single left-to-right scan with longest-match span dedup (the
same overlap policy the v1 resolver used). No parsing model, no similarity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

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

# Noun → allowed canonical entity types, for typed demonstratives,
# contrastives ("the other company") and selectives ("which company").
NOUN_TYPE_GATES: dict[str, frozenset[str]] = {
    "company": frozenset({"organization"}),
    "companies": frozenset({"organization"}),
    "organization": frozenset({"organization"}),
    "organisation": frozenset({"organization"}),
    "firm": frozenset({"organization"}),
    "startup": frozenset({"organization"}),
    "person": PERSON_TYPES,
    "man": PERSON_TYPES,
    "woman": PERSON_TYPES,
    "rocket": frozenset({"vehicle", "product", "technology"}),
    "vehicle": frozenset({"vehicle"}),
    "car": frozenset({"vehicle", "product"}),
    "satellite": frozenset({"product", "service", "program", "technology"}),
    "constellation": frozenset({"product", "service", "program"}),
    "network": frozenset({"service", "program", "technology"}),
    "product": frozenset({"product"}),
    "service": frozenset({"service"}),
    "program": frozenset({"program"}),
    "project": frozenset({"program"}),
    "publication": frozenset({"publication"}),
    "book": frozenset({"publication"}),
    "place": frozenset({"place"}),
    "city": frozenset({"place"}),
    "country": frozenset({"place"}),
    "technology": frozenset({"technology"}),
    # Russian controlled forms
    "компания": frozenset({"organization"}),
    "организация": frozenset({"organization"}),
    "ракета": frozenset({"vehicle", "product", "technology"}),
    "человек": PERSON_TYPES,
}

# Role descriptor → the relation whose *answer* the referent must have been
# (dialogue-role pass), or whose graph lookup verifies it (graph pass).
ROLE_DESCRIPTOR_RELATIONS: dict[str, str] = {
    "founder": "founded_by",
    "co-founder": "founded_by",
    "cofounder": "founded_by",
    "ceo": "leader_of",
    "leader": "leader_of",
    "head": "leader_of",
    "owner": "owned_by",
    "creator": "created_by",
    "основатель": "founded_by",
    "владелец": "owned_by",
    "руководитель": "leader_of",
}

_POSSESSIVES = frozenset({"his", "her", "its", "their", "его", "её", "их"})

_PERSON_PRONOUNS = frozenset({"he", "she", "his", "her", "him", "он", "она", "его", "её"})
_THING_PRONOUNS = frozenset({"it", "its", "оно"})
_PLURAL_PRONOUNS = frozenset({"they", "their", "them", "они", "их"})

# ── Token patterns ───────────────────────────────────────────────────────────

_NOUN_ALTERNATION = "|".join(sorted(NOUN_TYPE_GATES, key=len, reverse=True))
_ROLE_ALTERNATION = "|".join(sorted(ROLE_DESCRIPTOR_RELATIONS, key=len, reverse=True))

_PRONOUN_RE = re.compile(
    r"(?<!\w)(he|she|his|her|him|it|its|they|their|them|он|она|оно|они|его|её|их)(?!\w)",
    re.IGNORECASE,
)
_DEMONSTRATIVE_TYPED_RE = re.compile(
    rf"(?<!\w)(?:(?:this|that|the same|эта|та|это|тот)\s+)({_NOUN_ALTERNATION})(?!\w)",
    re.IGNORECASE,
)
_ROLE_DESCRIPTOR_RE = re.compile(
    rf"(?<!\w)(?:the|this|that|этот|тот)\s+({_ROLE_ALTERNATION})(?!\w)",
    re.IGNORECASE,
)
_CONTRASTIVE_RE = re.compile(
    rf"(?<!\w)the\s+other\s+(one|{_NOUN_ALTERNATION})(?!\w)"
    rf"|(?<!\w)друг(?:ая|ой|ую)\s+({_NOUN_ALTERNATION})(?!\w)",
    re.IGNORECASE,
)
_SELECTIVE_RE = re.compile(
    rf"^\s*(?:which|какая|какой|кто)\s+(?:(one|of\s+them|of\s+those|of\s+the\s+two|из\s+них)|({_NOUN_ALTERNATION}))(?!\w)",
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

    for m in _DEMONSTRATIVE_TYPED_RE.finditer(text):
        noun = m.group(1).lower()
        candidates.append(ReferenceSlot(
            surface=m.group(0), start=m.start(), end=m.end(),
            ref_class=DEMONSTRATIVE_TYPED, type_gate=NOUN_TYPE_GATES[noun],
        ))

    for m in _ROLE_DESCRIPTOR_RE.finditer(text):
        role = m.group(1).lower()
        candidates.append(ReferenceSlot(
            surface=m.group(0), start=m.start(), end=m.end(),
            ref_class=ROLE_DESCRIPTOR, type_gate=PERSON_TYPES,
            role_relation=ROLE_DESCRIPTOR_RELATIONS[role],
        ))

    for m in _CONTRASTIVE_RE.finditer(text):
        noun = (m.group(1) or m.group(2) or "one").lower()
        gate = NOUN_TYPE_GATES.get(noun)  # "one" → None → inherit any
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
            noun = (m.group(2) or "").lower()
            gate = NOUN_TYPE_GATES.get(noun)
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
