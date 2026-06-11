"""Relation allow/deny defaults for graph reasoning features."""
from __future__ import annotations

DEFAULT_REASONING_RELATIONS: frozenset[str] = frozenset({
    "made_of",
    "part_of",
    "is_a",
})

DEFAULT_DISABLED_RELATIONS: frozenset[str] = frozenset({
    "at_location",
})


def is_relation_enabled(
    relation_type: str,
    include_disabled_relations: bool = False,
) -> bool:
    """Return True when *relation_type* should participate in reasoning."""
    return include_disabled_relations or relation_type not in DEFAULT_DISABLED_RELATIONS
