"""Deterministic pre-gate checks for LLM-extracted relation nodes.

This layer is intentionally narrower than ``relation_candidate_validator``:
it rejects node surfaces which should not be presented to that validator at all.
It makes no overlay writes and does not alter relation policy or admission rules.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex


_AUTHORIAL_SUBJECT = re.compile(
    r"^(?:we|our\s+(?:approach|method|system|framework|model)|"
    r"this\s+(?:paper|study|work)|(?:the\s+)?proposed\s+(?:method|approach|system|framework|model))$",
    re.IGNORECASE,
)
_EVENT_LIKE = re.compile(
    r"^(?:identifying|achieving|providing|allowing|using|training|testing|"
    r"benchmarking|composing)\b|\b(?:to\s+(?:achieve|compose|identify|provide)|"
    r"can\s+be|being)\b",
    re.IGNORECASE,
)
_GENERIC_HEADS = frozenset({
    "activity", "approach", "challenge", "challenges", "classification",
    "data", "framework", "information", "method", "methods", "model",
    "models", "object", "objects", "outcome", "outcomes", "paper",
    "performance", "property", "security", "service", "services", "study",
    "system", "systems", "technique", "techniques", "uncertainties",
    "users", "vulnerabilities", "work",
})


@dataclass(frozen=True)
class NodeQualityDecision:
    accepted: bool
    reasons: tuple[str, ...]


# --------------------------------------------------------------------------- #
# Class-subject recognition
# --------------------------------------------------------------------------- #
# A statutory subject is frequently a *description of a class* — "whoever
# knowingly threatens ...", "any invention made in outer space", "a person who
# receives ...".  The entity-only validator rejects these (they are not named
# entities), which in the legal pilots deleted whole offences.  This recognizer
# lets such a node be routed to a review-only ``class_subject`` proposal instead
# of discarded.
#
# It is structural, not a keyword list: the decision rests on determiner /
# quantifier position, the presence of a relative clause, and phrase length —
# the same cues a reader uses to tell "SpaceX" (an entity) from "a company that
# builds rockets" (a class).  It never auto-admits anything.

# Leading quantifier / indefinite determiners that open a class description.
# The definite article "the" is deliberately excluded: it usually points at a
# specific or anaphoric referent ("the individual"), not a fresh class.
_CLASS_LEAD = frozenset({
    "whoever", "whosoever", "whomever", "anyone", "someone", "everyone",
    "a", "an", "any", "each", "every", "all", "no", "one", "another",
})
_RELATIVE_CLAUSE = re.compile(r"\b(?:who|whom|whose|which|that)\b", re.IGNORECASE)
_MIN_CLASS_WORDS = 5


def classify_subject_node(surface: str, evidence_sentence: str = "") -> str:
    """Return ``"class_subject"`` for a class description, else ``"entity"``.

    A node is a class subject when it opens with a quantifier/indefinite
    determiner (or a common-noun head) *and* is either qualified by a relative
    clause or long enough to be a real description rather than a short entity
    phrase.  Authorial and event-like fragments are never class subjects.
    """

    text = _normalise(surface)
    if not text:
        return "entity"
    if _AUTHORIAL_SUBJECT.fullmatch(text) or _EVENT_LIKE.search(text):
        return "entity"
    words = text.split()
    lead = words[0].lower()
    has_relative = bool(_RELATIVE_CLAUSE.search(text))
    # A common-noun head is a lowercase leading word that is not itself a
    # quantifier; a quantifier lead is the other opening shape.
    common_noun_head = lead.isalpha() and lead[:1].islower() and lead not in _CLASS_LEAD
    opens_class = lead in _CLASS_LEAD or common_noun_head
    if opens_class and (has_relative or len(words) >= _MIN_CLASS_WORDS):
        return "class_subject"
    return "entity"


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _has_named_qualifier(text: str) -> bool:
    """Conservative surface cue: a non-initial capitalized token indicates a name.

    This does not accept a node by itself; a candidate must still resolve through
    ``EntitySurfaceIndex``. It only prevents e.g. ``SkyServer system`` from
    receiving a second, misleading generic-head reason.
    """
    tokens = _normalise(text).split()
    return any(token[:1].isupper() for token in tokens[1:])


def _is_generic_noun_phrase(text: str) -> bool:
    tokens = re.findall(r"[A-Za-z][A-Za-z-]*", _normalise(text).lower())
    if not tokens or _has_named_qualifier(text):
        return False
    return tokens[-1] in _GENERIC_HEADS or (len(tokens) == 1 and tokens[0] in _GENERIC_HEADS)


def _is_list_derived(surface: str, sentence: str) -> bool:
    """Flag a model-created relation whose extracted node occurs in a colon list."""
    if ":" not in sentence:
        return False
    after_colon = sentence.split(":", 1)[1]
    # Require a list-shaped suffix; a normal colon in prose should not trigger.
    return after_colon.count(",") >= 1 and bool(re.search(re.escape(surface), after_colon, re.IGNORECASE))


def assess_node_quality(
    subject: str,
    object_: str,
    evidence_sentence: str,
    index: EntitySurfaceIndex,
) -> NodeQualityDecision:
    """Return deterministic reject reasons for a raw extracted triple.

    Resolution is deliberately required for both endpoints. Unknown proper names
    remain proposal candidates for a *separate* deterministic resolution lane;
    they are not safe serving-graph nodes merely because they are capitalized.
    """
    subject = _normalise(subject)
    object_ = _normalise(object_)
    reasons: list[str] = []
    if _AUTHORIAL_SUBJECT.fullmatch(subject):
        reasons.append("authorial_self_reference")
    for surface in (subject, object_):
        if _EVENT_LIKE.search(surface):
            reasons.append("event_like_node")
        if _is_generic_noun_phrase(surface):
            reasons.append("generic_abstract_node")
        if _is_list_derived(surface, evidence_sentence):
            reasons.append("list_derived_context")
        if index.resolve(surface) is None:
            reasons.append("unresolvable_entity")
    return NodeQualityDecision(not reasons, tuple(sorted(set(reasons))))


def filter_triples(
    triples: Iterable[dict],
    evidence_sentence: str,
    index: EntitySurfaceIndex,
) -> tuple[list[dict], list[dict]]:
    """Partition raw triple dictionaries into accepted and rejected, preserving data."""
    accepted: list[dict] = []
    rejected: list[dict] = []
    for triple in triples:
        decision = assess_node_quality(
            str(triple.get("subject") or ""), str(triple.get("object") or ""),
            evidence_sentence, index,
        )
        if decision.accepted:
            accepted.append(triple)
        else:
            rejected.append({**triple, "node_quality_reasons": list(decision.reasons)})
    return accepted, rejected
