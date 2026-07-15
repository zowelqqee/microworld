"""Read-only graph-backed input resolution for evidence relation overlays.

The entity surface index is intentionally conservative: it primarily indexes
declared entities and aliases.  Proposal graphs, however, can contain valid
relation subjects that were never emitted as ``overlay_entity`` items.  A
question about one of those nodes previously failed before the planner could
inspect its local evidence frontier.

``GraphInputLayer`` exposes referential subject/object labels from actual
``overlay_relation`` edges as additional *input surfaces*.  It holds no new
facts, never writes accepted memory, and does not infer aliases: a match is an
exact graph-node label.  The normal entity index remains authoritative when
both layers know a surface.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable


_DEICTIC_LEADING = frozenset({
    "our", "my", "we", "us", "i", "this", "these", "those", "that",
})
_BARE_NONREFERENTIAL = frozenset({
    "it", "its", "we", "us", "the", "they", "them", "this", "these", "those",
})


def _compact(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())


def _is_referential_node(value: object) -> bool:
    """Keep named graph nodes; reject document-internal deixis and pronouns."""

    label = _compact(value)
    if not label:
        return False
    normalized = label.casefold()
    if normalized in _BARE_NONREFERENTIAL:
        return False
    return normalized.split(" ", 1)[0] not in _DEICTIC_LEADING


@dataclass(frozen=True)
class GraphInputLayer:
    """Canonical node labels made available to the input resolver.

    ``node_surfaces`` is sorted deterministically so overlay item order cannot
    change resolution when two sources expose the same case-insensitive label.
    """

    node_surfaces: tuple[tuple[str, str], ...]
    relation_count: int

    @classmethod
    def from_overlay_items(cls, items: Iterable[dict]) -> "GraphInputLayer":
        canonical_by_normalized_surface: dict[str, str] = {}
        relation_count = 0
        for item in items:
            if not isinstance(item, dict) or item.get("overlay_type") != "overlay_relation":
                continue
            relation_count += 1
            for value in (item.get("subject"), item.get("object")):
                label = _compact(value)
                if not _is_referential_node(label):
                    continue
                canonical_by_normalized_surface.setdefault(label.casefold(), label)
        pairs = tuple(sorted(
            ((label, label) for label in canonical_by_normalized_surface.values()),
            key=lambda pair: (-len(pair[0]), pair[0].casefold()),
        ))
        return cls(node_surfaces=pairs, relation_count=relation_count)

    @property
    def node_count(self) -> int:
        return len(self.node_surfaces)

