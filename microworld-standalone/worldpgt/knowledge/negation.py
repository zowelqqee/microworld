"""Explicit negation helpers for entity ``is_a`` QA.

The module is deliberately small and deterministic. It only turns explicit,
safe overlay evidence into a negative answer; missing facts can at most produce
an audit hint about weak closed-world coverage.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Optional

from worldpgt.knowledge.entity_types import canonicalize_entity_type
from worldpgt.multihop_qa.path_validator import validate_hop_safety
from worldpgt.multihop_qa.types import HopEdge

NegationSupportKind = Literal["explicit_type_contradiction", "entity_type_mismatch"]

WELL_COVERED_FACT_THRESHOLD = 3

_CLASS_ALIASES: dict[str, set[str]] = {
    "person": {
        "person",
        "human",
        "human being",
        "individual",
        "businessman",
        "businesswoman",
        "businessperson",
        "entrepreneur",
        "engineer",
        "founder",
        "politician",
    },
    "organization": {
        "organization",
        "company",
        "private company",
        "public company",
        "corporation",
        "agency",
        "government agency",
        "space agency",
        "manufacturer",
        "aerospace manufacturer",
        "space transportation company",
        "news organization",
        "business news channel",
    },
    "publication": {
        "publication",
        "magazine",
        "newspaper",
        "news service",
        "news agency",
        "online publication",
        "business magazine",
    },
    "product": {
        "product",
        "software product",
        "service",
        "satellite internet constellation",
    },
    "vehicle": {
        "vehicle",
        "spacecraft",
        "rocket",
        "launch vehicle",
        "reusable launch vehicle",
    },
    "program": {"program"},
    "place": {"place", "country", "city", "location"},
    "concept": {"concept", "field", "industry"},
    "technology": {"technology"},
}

_EXPLICIT_CONTRADICTIONS: set[tuple[str, str]] = {
    ("private company", "government agency"),
    ("government agency", "private company"),
}


@dataclass(frozen=True)
class NegationCandidate:
    subject: str
    queried_class: str
    actual_class: str
    support_kind: NegationSupportKind
    support: str
    evidence_item: str


def normalize_label(value: str | None) -> str:
    text = (value or "").lower().strip()
    text = re.sub(r"[''`]", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    return text


def class_group(label: str | None) -> Optional[str]:
    normalized = normalize_label(label)
    canonical = canonicalize_entity_type(normalized)
    if canonical is not None and canonical != "other":
        return canonical
    for group, aliases in _CLASS_ALIASES.items():
        if normalized in aliases:
            return group
        if any(alias in normalized for alias in aliases if " " in alias):
            return group
    return None


def classes_contradict(actual_class: str, queried_class: str) -> bool:
    actual_norm = normalize_label(actual_class)
    queried_norm = normalize_label(queried_class)
    if (actual_norm, queried_norm) in _EXPLICIT_CONTRADICTIONS:
        return True
    actual_group = class_group(actual_class)
    queried_group = class_group(queried_class)
    if not actual_group or not queried_group:
        return False
    return actual_group != queried_group


def item_is_safe_negation_basis(item: dict) -> bool:
    overlay_type = item.get("overlay_type")
    predicate = "is_a"
    obj = item.get("object") if overlay_type == "overlay_relation" else item.get("definition")
    if overlay_type == "overlay_relation":
        predicate = str(item.get("predicate") or "")
    edge = HopEdge(
        subject=str(item.get("subject") or item.get("label") or ""),
        predicate=predicate,
        object=str(obj or item.get("entity_type") or ""),
        overlay_type=str(overlay_type or ""),
        trust=str(item.get("trust") or ""),
        stability=str(item.get("stability") or "stable"),
        risk=str(item.get("risk") or ""),
        source_page=str(item.get("source_page") or ""),
        temporal_class=str(item.get("temporal_class") or "historical"),
        as_of=item.get("as_of"),
    )
    valid, _reason = validate_hop_safety(edge)
    return valid and edge.stability != "volatile"


def coverage_score(provider, subject: str) -> int:
    """Return a simple count of known facts for *subject* in the overlay."""

    definition_count = 1 if provider.get_definition(subject) else 0
    relations = provider.get_relations(subject)
    return definition_count + len(relations)


def well_covered(provider, subject: str, threshold: int = WELL_COVERED_FACT_THRESHOLD) -> bool:
    return coverage_score(provider, subject) >= threshold
