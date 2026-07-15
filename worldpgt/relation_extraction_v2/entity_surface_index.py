"""Entity surface form index for Relation Extraction v2.

Builds a lookup table from known surface forms to canonical entity names, using:
- accepted overlay entities + aliases
- promoted overlay entities + aliases
- snapshot dry-run overlay entities + aliases
- snapshot manifest titles and normalized titles

Longest-match wins for entity resolution. Word-boundary matching. Tiny generic
terms (< 4 chars, or from a small blocked list) are skipped unless allowlisted
as known safe entity names.

No ML, no embeddings, no network. Read-only over all sources.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional

from worldpgt.knowledge.entity_type_classifier import classify_entity_type
from worldpgt.knowledge.entity_types import canonicalize_entity_type
from worldpgt.reasoning.graph_input import GraphInputLayer

# Words that are too generic to be treated as entity surface forms.
_BLOCKED_SURFACES = frozenset({
    "a", "an", "the", "it", "is", "in", "on", "at", "to", "by", "of", "or",
    "and", "for", "as", "be", "was", "are", "were", "has", "have", "had",
    "not", "but", "if", "do", "did", "its", "his", "her", "with", "from",
    "that", "this", "they", "them", "also", "than", "then", "when", "who",
    "which", "more", "most", "some", "all", "one", "two", "use", "used",
    "new", "said", "each", "only", "over", "inc", "corp", "llc", "ltd",
    "company", "companies", "organization", "organizations", "business",
    "product", "products", "service", "services", "technology", "technologies",
    "project", "projects", "platform", "device", "system", "systems",
    "group", "global", "international", "national", "american", "public",
    "private", "based", "known", "related", "founded", "led", "owned",
    "created", "formed", "network", "industry",
})

# Minimum number of characters for a surface form to be indexed.
_MIN_SURFACE_LEN = 4

# Explicit allowlist: short proper names that are safe despite length < 4.
_ALLOWLISTED_SHORT = frozenset({
    "SpaceX", "Tesla", "NASA", "DARPA", "xAI", "ISS",
})

_BROAD_TOPIC_ALIASES = frozenset({
    "africa",
    "agriculture",
    "asia",
    "astronomy",
    "biology",
    "buddhism",
    "chemistry",
    "christianity",
    "computer science",
    "economics",
    "education",
    "energy",
    "europe",
    "film",
    "islam",
    "law",
    "mathematics",
    "medicine",
    "music",
    "north america",
    "oceania",
    "physics",
    "religion",
    "south america",
    "sport",
    "transportation",
})


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _is_blocked(surface: str) -> bool:
    low = surface.lower().strip()
    if low in _BLOCKED_SURFACES:
        return True
    if len(low) < _MIN_SURFACE_LEN and surface not in _ALLOWLISTED_SHORT:
        return True
    # All-lowercase words shorter than 5 are too risky.
    if low == surface and len(low) < 5:
        return True
    return False


def _is_unsafe_alias(alias: str, label: str, entity_type: str = "") -> bool:
    alias = _norm(alias)
    if not alias:
        return False
    if alias.lower() in _BROAD_TOPIC_ALIASES and alias.lower() != _norm(label).lower():
        return True
    if " " in alias:
        return False
    if canonicalize_entity_type(entity_type) == "person":
        return False
    label_parts = _norm(label).lower().split()
    alias_norm = alias.lower()
    return len(label_parts) > 1 and alias_norm == label_parts[-1] and alias_norm != label_parts[0]


def _overlay_entities(overlay_path: Path) -> list[tuple[str, str]]:
    """Return (surface, canonical_name) pairs from an overlay file."""

    if not overlay_path.exists():
        return []
    items = json.loads(overlay_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        return []
    label_pairs: list[tuple[str, str]] = []
    alias_pairs: list[tuple[str, str]] = []
    for item in items:
        if item.get("overlay_type") != "overlay_entity":
            continue
        label = _norm(str(item.get("label") or ""))
        if not label:
            continue
        label_pairs.append((label, label))
        entity_type = str(item.get("entity_type") or "")
        for alias in item.get("aliases") or []:
            alias = _norm(str(alias))
            if alias and alias != label and not _is_unsafe_alias(alias, label, entity_type):
                alias_pairs.append((alias, label))
    return [*label_pairs, *alias_pairs]


def _load_overlay_items(overlay_path: Path) -> list[dict]:
    if not overlay_path.exists():
        return []
    items = json.loads(overlay_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


class EntitySurfaceIndex:
    """Longest-match entity surface resolver."""

    def __init__(
        self,
        accepted_overlay_path: Path,
        promoted_overlay_path: Path,
        snapshot_overlay_path: Path,
        snapshot_manifest_path: Optional[Path] = None,
        graph_input: GraphInputLayer | None = None,
    ) -> None:
        self._surface_to_canonical: dict[str, str] = {}
        self._canonical_to_type: dict[str, str] = {}
        self._canonical_to_definition: dict[str, str] = {}

        for path in (accepted_overlay_path, promoted_overlay_path, snapshot_overlay_path):
            for surface, canonical in _overlay_entities(path):
                if not _is_blocked(surface):
                    self._surface_to_canonical.setdefault(surface, canonical)

        # Graph-backed input is deliberately supplemental.  Declared entities
        # and aliases above keep precedence; this layer only makes existing,
        # evidence-graph node labels reachable by the parser/planner.
        if graph_input is not None:
            for surface, canonical in graph_input.node_surfaces:
                if not _is_blocked(surface):
                    self._surface_to_canonical.setdefault(surface, canonical)

        # Definitions can come from any readable overlay source and are used
        # only as a fallback when no explicit entity type exists.
        for path in (accepted_overlay_path, promoted_overlay_path, snapshot_overlay_path):
            for item in _load_overlay_items(path):
                if item.get("overlay_type") != "overlay_definition":
                    continue
                subject = _norm(str(item.get("subject") or ""))
                definition = _norm(str(item.get("definition") or ""))
                if subject and definition:
                    self._canonical_to_definition.setdefault(subject, definition)

        # Add entity type hints from promoted overlay (most authoritative).
        if promoted_overlay_path.exists():
            for item in _load_overlay_items(promoted_overlay_path):
                if item.get("overlay_type") == "overlay_entity":
                    label = _norm(str(item.get("label") or ""))
                    etype = canonicalize_entity_type(str(item.get("entity_type") or ""))
                    if label and etype:
                        self._canonical_to_type[label] = etype

        # Add snapshot manifest titles.
        if snapshot_manifest_path and snapshot_manifest_path.exists():
            manifest = json.loads(snapshot_manifest_path.read_text(encoding="utf-8"))
            if isinstance(manifest, list):
                for row in manifest:
                    for key in ("title", "normalized_title"):
                        title = _norm(str(row.get(key) or ""))
                        if title and not _is_blocked(title):
                            self._surface_to_canonical.setdefault(title, title)

        # Pre-sort surfaces by length descending for longest-match.
        self._surfaces_sorted = sorted(
            self._surface_to_canonical.keys(), key=len, reverse=True
        )
        # Case-insensitive surface -> canonical, built once so find_in_text can
        # resolve a combined-regex match back to its canonical without redoing
        # per-surface lookups.
        self._canonical_by_lower_surface: dict[str, str] = {
            surface.lower(): canonical
            for surface, canonical in self._surface_to_canonical.items()
        }
        # One compiled regex over every known surface, built once here instead
        # of per-call: find_in_text used to re.compile + re.finditer once per
        # surface (thousands of entities) on every invocation, which dominated
        # per-request latency. A single alternation, ordered longest-first so
        # the engine's leftmost-first alternative selection reproduces the
        # original longest-match-wins semantics, does one scan instead.
        if self._surfaces_sorted:
            alternation = "|".join(re.escape(s) for s in self._surfaces_sorted)
            self._combined_surface_re: re.Pattern | None = re.compile(
                rf"(?<!\w)(?:{alternation})(?!\w)", re.IGNORECASE
            )
        else:
            self._combined_surface_re = None

    # ------------------------------------------------------------------ #

    def resolve(self, surface: str) -> Optional[str]:
        """Return the canonical name for ``surface``, or None."""
        norm_s = _norm(surface)
        return self._surface_to_canonical.get(norm_s)

    def entity_type(self, canonical: str) -> Optional[str]:
        norm_canonical = _norm(canonical)
        explicit = self._canonical_to_type.get(norm_canonical)
        if explicit:
            return explicit
        definition = self._canonical_to_definition.get(norm_canonical)
        if definition:
            return classify_entity_type(definition)
        return None

    def all_surfaces(self) -> list[str]:
        return list(self._surfaces_sorted)

    def known_surfaces_set(self) -> frozenset[str]:
        return frozenset(self._surface_to_canonical)

    def find_in_text(self, text: str) -> list[tuple[str, str, int, int]]:
        """Find all (surface, canonical, start, end) spans in ``text``, longest match."""

        if self._combined_surface_re is None:
            return []
        found: list[tuple[str, str, int, int]] = []
        for m in self._combined_surface_re.finditer(text):
            canonical = self._canonical_by_lower_surface[m.group(0).lower()]
            found.append((m.group(0), canonical, m.start(), m.end()))
        return found

    def has_surface(self, text: str) -> bool:
        """Quick check: does any known entity surface appear in ``text``?"""

        low = text.lower()
        return any(s.lower() in low for s in self._surfaces_sorted)
