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


def _overlay_entities(overlay_path: Path) -> list[tuple[str, str]]:
    """Return (surface, canonical_name) pairs from an overlay file."""

    if not overlay_path.exists():
        return []
    items = json.loads(overlay_path.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        return []
    pairs: list[tuple[str, str]] = []
    for item in items:
        if item.get("overlay_type") != "overlay_entity":
            continue
        label = _norm(str(item.get("label") or ""))
        if not label:
            continue
        pairs.append((label, label))
        for alias in item.get("aliases") or []:
            alias = _norm(str(alias))
            if alias and alias != label:
                pairs.append((alias, label))
    return pairs


class EntitySurfaceIndex:
    """Longest-match entity surface resolver."""

    def __init__(
        self,
        accepted_overlay_path: Path,
        promoted_overlay_path: Path,
        snapshot_overlay_path: Path,
        snapshot_manifest_path: Optional[Path] = None,
    ) -> None:
        self._surface_to_canonical: dict[str, str] = {}
        self._canonical_to_type: dict[str, str] = {}

        for path in (accepted_overlay_path, promoted_overlay_path, snapshot_overlay_path):
            for surface, canonical in _overlay_entities(path):
                if not _is_blocked(surface):
                    self._surface_to_canonical.setdefault(surface, canonical)

        # Add entity type hints from promoted overlay (most authoritative).
        if promoted_overlay_path.exists():
            items = json.loads(promoted_overlay_path.read_text(encoding="utf-8"))
            if isinstance(items, list):
                for item in items:
                    if item.get("overlay_type") == "overlay_entity":
                        label = _norm(str(item.get("label") or ""))
                        etype = str(item.get("entity_type") or "")
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

    # ------------------------------------------------------------------ #

    def resolve(self, surface: str) -> Optional[str]:
        """Return the canonical name for ``surface``, or None."""
        norm_s = _norm(surface)
        return self._surface_to_canonical.get(norm_s)

    def entity_type(self, canonical: str) -> Optional[str]:
        return self._canonical_to_type.get(_norm(canonical))

    def all_surfaces(self) -> list[str]:
        return list(self._surfaces_sorted)

    def known_surfaces_set(self) -> frozenset[str]:
        return frozenset(self._surface_to_canonical)

    def find_in_text(self, text: str) -> list[tuple[str, str, int, int]]:
        """Find all (surface, canonical, start, end) spans in ``text``, longest match."""

        found: list[tuple[str, str, int, int]] = []
        # Track which character positions are already covered.
        covered: list[tuple[int, int]] = []

        for surface in self._surfaces_sorted:
            canonical = self._surface_to_canonical[surface]
            pattern = r"(?<!\w)" + re.escape(surface) + r"(?!\w)"
            for m in re.finditer(pattern, text, re.IGNORECASE):
                start, end = m.start(), m.end()
                # Skip if already covered by a longer match.
                if any(cs <= start and end <= ce for cs, ce in covered):
                    continue
                found.append((m.group(0), canonical, start, end))
                covered.append((start, end))
        return found

    def has_surface(self, text: str) -> bool:
        """Quick check: does any known entity surface appear in ``text``?"""

        low = text.lower()
        return any(s.lower() in low for s in self._surfaces_sorted)
