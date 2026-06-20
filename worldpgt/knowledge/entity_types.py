"""Canonical entity type vocabulary shared by knowledge pipelines."""

from __future__ import annotations

from typing import Literal, Optional, TypeAlias, get_args

CanonicalEntityType: TypeAlias = Literal[
    "person",
    "organization",
    "publication",
    "product",
    "vehicle",
    "program",
    "place",
    "concept",
    "technology",
    "other",
]

CANONICAL_ENTITY_TYPES: frozenset[str] = frozenset(get_args(CanonicalEntityType))

_LEGACY_ENTITY_TYPE_MAP: dict[str, CanonicalEntityType] = {
    "unknown": "other",
    "country": "place",
    "location": "place",
}


def canonicalize_entity_type(value: str | None) -> Optional[CanonicalEntityType]:
    """Return a canonical entity type, or None for missing/unsupported values."""

    normalized = (value or "").strip().lower()
    if not normalized:
        return None
    mapped = _LEGACY_ENTITY_TYPE_MAP.get(normalized, normalized)
    if mapped in CANONICAL_ENTITY_TYPES:
        return mapped  # type: ignore[return-value]
    return None


def is_specific_entity_type(value: str | None) -> bool:
    """True when *value* is a meaningful type rather than a catch-all."""

    etype = canonicalize_entity_type(value)
    return etype is not None and etype != "other"
