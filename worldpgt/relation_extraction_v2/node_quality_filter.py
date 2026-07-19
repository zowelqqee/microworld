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
