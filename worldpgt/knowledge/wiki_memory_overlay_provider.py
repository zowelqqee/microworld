"""Read-only overlay provider for Wiki Candidate Memory Overlay v1.

Loads overlay JSON and provides deterministic query methods.
No embeddings. No fuzzy matching. No network access.

Entity resolution order:
  1. Exact label match (case-sensitive)
  2. Alias match (case-sensitive)
  3. Case-insensitive label match
  4. Case-insensitive alias match
  5. Page-title match (case-insensitive)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Optional


def _normalize(s: str) -> str:
    """Light normalization: lowercase, collapse whitespace, strip punctuation."""
    s = s.lower().strip()
    s = re.sub(r"[''`]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


class WikiMemoryOverlayProvider:
    """Provides deterministic query access to a loaded wiki memory overlay."""

    def __init__(self, overlay_path: str | Path | None = None) -> None:
        self._entities: list[dict] = []
        self._definitions: list[dict] = []
        self._relations: list[dict] = []
        self._context_links: list[dict] = []
        self._source_facts: list[dict] = []
        self._items_used: int = 0

        if overlay_path is not None:
            self.load(overlay_path)

    def load(self, path: str | Path) -> None:
        raw: list[dict] = json.loads(Path(path).read_text(encoding="utf-8"))
        self._entities = [i for i in raw if i.get("overlay_type") == "overlay_entity"]
        self._definitions = [i for i in raw if i.get("overlay_type") == "overlay_definition"]
        self._relations = [i for i in raw if i.get("overlay_type") == "overlay_relation"]
        self._context_links = [i for i in raw if i.get("overlay_type") == "overlay_context_link"]
        self._source_facts = [i for i in raw if i.get("overlay_type") == "overlay_source_fact"]

    # ------------------------------------------------------------------
    # Public query API
    # ------------------------------------------------------------------

    def get_entity(self, label_or_alias: str) -> Optional[dict]:
        """Resolve entity by label or alias (exact then case-insensitive)."""
        norm = _normalize(label_or_alias)
        # 1. Exact label
        for e in self._entities:
            if e.get("label") == label_or_alias:
                self._items_used += 1
                return e
        # 2. Exact alias
        for e in self._entities:
            for alias in e.get("aliases", []):
                if alias == label_or_alias:
                    self._items_used += 1
                    return e
        # 3. Case-insensitive label
        for e in self._entities:
            if _normalize(e.get("label", "")) == norm:
                self._items_used += 1
                return e
        # 4. Case-insensitive alias
        for e in self._entities:
            for alias in e.get("aliases", []):
                if _normalize(alias) == norm:
                    self._items_used += 1
                    return e
        # 5. Page-title match
        for e in self._entities:
            if _normalize(e.get("source_page", "")) == norm:
                self._items_used += 1
                return e
        return None

    def get_definition(self, subject: str) -> Optional[dict]:
        """Return overlay_definition for the given subject."""
        norm = _normalize(subject)
        for d in self._definitions:
            if _normalize(d.get("subject", "")) == norm:
                self._items_used += 1
                return d
        return None

    def get_relations(
        self,
        subject: str,
        predicate: Optional[str] = None,
    ) -> list[dict]:
        """Return overlay_relation items for subject, optionally filtered by predicate."""
        norm = _normalize(subject)
        out = [
            r for r in self._relations
            if _normalize(r.get("subject", "")) == norm
        ]
        if predicate is not None:
            out = [r for r in out if r.get("predicate") == predicate]
        self._items_used += len(out)
        return out

    def get_context_links(
        self,
        source_page: Optional[str] = None,
        target: Optional[str] = None,
        surface: Optional[str] = None,
    ) -> list[dict]:
        """Return overlay_context_link items matching any supplied filter."""
        out = list(self._context_links)
        if source_page is not None:
            sp_norm = _normalize(source_page)
            out = [l for l in out if _normalize(l.get("source_page", "")) == sp_norm]
        if target is not None:
            t_norm = _normalize(target)
            out = [l for l in out if _normalize(l.get("target", "")) == t_norm]
        if surface is not None:
            s_norm = _normalize(surface)
            out = [l for l in out if _normalize(l.get("surface", "")) == s_norm]
        self._items_used += len(out)
        return out

    def get_source_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        source_name: Optional[str] = None,
    ) -> list[dict]:
        """Return overlay_source_fact items matching any supplied filter."""
        out = list(self._source_facts)
        if subject is not None:
            s_norm = _normalize(subject)
            out = [f for f in out if _normalize(f.get("subject", "")) == s_norm]
        if predicate is not None:
            out = [f for f in out if f.get("predicate") == predicate]
        if source_name is not None:
            sn_norm = _normalize(source_name)
            out = [f for f in out if _normalize(f.get("source_name", "")) == sn_norm]
        self._items_used += len(out)
        return out

    def explain_link(self, source_page: str, target: str) -> list[dict]:
        """Return context links connecting source_page to target."""
        return self.get_context_links(source_page=source_page, target=target)

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    @property
    def items_used(self) -> int:
        return self._items_used

    def total_items(self) -> int:
        return (
            len(self._entities)
            + len(self._definitions)
            + len(self._relations)
            + len(self._context_links)
            + len(self._source_facts)
        )
