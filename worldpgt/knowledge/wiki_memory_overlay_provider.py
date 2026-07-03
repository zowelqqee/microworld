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
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^(?:the|a|an)\s+", "", s)
    return s


_BAD_SOURCE_PAGE_SUBJECT_PREFIXES = (
    "according ",
    "after ",
    "as ",
    "before ",
    "born ",
    "by ",
    "during ",
    "for ",
    "from ",
    "he ",
    "his ",
    "in ",
    "it ",
    "its ",
    "on ",
    "personal life ",
    "she ",
    "their ",
    "throughout ",
    "towards ",
    "when ",
)


def _tokens(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(s))


def _significant_tokens(s: str) -> set[str]:
    return {tok for tok in _tokens(s) if len(tok) > 1 and tok not in {"inc", "co"}}


def _bad_source_page_definition_subject(subject: str) -> bool:
    n = _normalize(subject)
    if not n:
        return True
    if any(
        f" {term} " in f" {n} "
        for term in ("son", "daughter", "wife", "husband", "mother", "father")
    ):
        return True
    if " asked " in f" {n} " or " if " in f" {n} ":
        return True
    if any(n.startswith(prefix) for prefix in _BAD_SOURCE_PAGE_SUBJECT_PREFIXES):
        return True
    if len(n.split()) > 8:
        return True
    if ";" in subject or "\x00" in subject:
        return True
    return False


def _meaningful_edge_tokens(s: str) -> tuple[str, str] | tuple[None, None]:
    parts = [
        tok
        for tok in _tokens(s)
        if tok not in {"sr", "jr", "ii", "iii", "iv", "inc", "co", "company"}
    ]
    if not parts:
        return None, None
    return parts[0], parts[-1]


def _is_bare_trailing_alias(subject: str, entity: dict) -> bool:
    subject_parts = _tokens(subject)
    if len(subject_parts) != 1:
        return False
    subject_token = subject_parts[0]
    for surface in (entity.get("label"), entity.get("source_page")):
        surface_parts = _tokens(str(surface or ""))
        if len(surface_parts) > 1 and subject_token == surface_parts[-1]:
            return subject_token != surface_parts[0]
    return False


def _bad_definition_text(definition: str) -> bool:
    d = (definition or "").strip()
    if not d or "\x00" in d:
        return True
    if "?" in d:
        return True
    if len(d.split()) > 24:
        return True
    return False


def _source_page_definition_score(definition: dict, entity: dict) -> int:
    """Rank source-page definitions without treating every copula as a bio."""
    subject = str(definition.get("subject") or "")
    def_text = str(definition.get("definition") or "")
    if _bad_source_page_definition_subject(subject) or _bad_definition_text(def_text):
        return -1
    if _is_bare_trailing_alias(subject, entity):
        return -1

    primary_surfaces = [
        str(entity.get("label") or ""),
        str(entity.get("source_page") or ""),
    ]
    subject_norm_for_compound = _normalize(subject)
    if " and " in f" {subject_norm_for_compound} " and not any(
        " and " in f" {_normalize(surface)} " for surface in primary_surfaces
    ):
        return -1
    entity_surfaces = [
        *primary_surfaces,
        *(str(a) for a in (entity.get("aliases") or [])),
    ]
    subject_norm = _normalize(subject)
    entity_norms = {_normalize(surface) for surface in entity_surfaces if surface}
    score = 0
    if subject_norm in entity_norms:
        score += 100

    subject_tokens = _significant_tokens(subject)
    best_overlap = 0
    for surface in primary_surfaces:
        surface_tokens = _significant_tokens(surface)
        if not surface_tokens:
            continue
        overlap = len(surface_tokens & subject_tokens)
        best_overlap = max(best_overlap, overlap)
        if surface_tokens <= subject_tokens or subject_tokens <= surface_tokens:
            score += 35
        surface_parts = _tokens(surface)
        subject_parts = _tokens(subject)
        if surface_parts and subject_parts and surface_parts[-1] == subject_parts[-1]:
            score += 25
        surface_first, surface_last = _meaningful_edge_tokens(surface)
        subject_first, subject_last = _meaningful_edge_tokens(subject)
        if surface_first and (surface_first, surface_last) == (subject_first, subject_last):
            score += 50
    score += best_overlap * 10
    if len(subject.split()) <= 5:
        score += 5
    return score


class WikiMemoryOverlayProvider:
    """Provides deterministic query access to a loaded wiki memory overlay."""

    def __init__(self, overlay_path: str | Path | None = None) -> None:
        self._entities: list[dict] = []
        self._definitions: list[dict] = []
        self._relations: list[dict] = []
        self._context_links: list[dict] = []
        self._source_facts: list[dict] = []
        self._items_used: int = 0
        self._definitions_by_norm: dict[str, list[dict]] = {}
        self._definitions_by_source_page_norm: dict[str, list[dict]] = {}

        if overlay_path is not None:
            self.load(overlay_path)

    def load(self, path: str | Path) -> None:
        raw: list[dict] = json.loads(Path(path).read_text(encoding="utf-8"))
        self._entities = [i for i in raw if i.get("overlay_type") == "overlay_entity"]
        self._definitions = [i for i in raw if i.get("overlay_type") == "overlay_definition"]
        self._relations = [i for i in raw if i.get("overlay_type") == "overlay_relation"]
        self._context_links = [i for i in raw if i.get("overlay_type") == "overlay_context_link"]
        self._source_facts = [i for i in raw if i.get("overlay_type") == "overlay_source_fact"]
        self._definitions_by_norm = {}
        self._definitions_by_source_page_norm = {}
        for definition in self._definitions:
            subject_norm = _normalize(definition.get("subject", ""))
            if subject_norm:
                self._definitions_by_norm.setdefault(subject_norm, []).append(definition)
            source_page_norm = _normalize(definition.get("source_page", ""))
            if source_page_norm:
                self._definitions_by_source_page_norm.setdefault(
                    source_page_norm, []
                ).append(definition)

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
        direct = self._definitions_by_norm.get(norm, [])
        if direct:
            self._items_used += 1
            return direct[0]
        entity = self.get_entity(subject)
        if entity:
            return self.get_definition_for_entity(entity, original_subject_norm=norm)
        return None

    def get_definition_for_entity(
        self,
        entity: dict,
        *,
        original_subject_norm: str | None = None,
    ) -> Optional[dict]:
        """Return the best definition for an entity card.

        Besides label/source-page/alias subject matches, pump overlays often
        store the lead definition under a fuller display name while the entity
        card keeps the page title. In that case, choose a clean definition from
        the same source page and reject obvious copula fragments.
        """
        candidate_subjects = [
            entity.get("label", ""),
            entity.get("source_page", ""),
            *(entity.get("aliases") or []),
        ]
        scored: list[tuple[int, int, dict]] = []
        for index, candidate in enumerate(candidate_subjects):
            candidate_norm = _normalize(candidate)
            if not candidate_norm or candidate_norm == original_subject_norm:
                continue
            direct = self._definitions_by_norm.get(candidate_norm, [])
            if direct:
                bonus = 100 if index < 2 else 10
                for definition in direct:
                    score = _source_page_definition_score(definition, entity)
                    if score >= 0:
                        scored.append((score + bonus, len(scored), definition))

        source_page_norm = _normalize(entity.get("source_page", ""))
        if source_page_norm:
            candidates = self._definitions_by_source_page_norm.get(source_page_norm, [])
            start = len(scored)
            scored.extend(
                (
                    score
                    + (80 if index == 0 else 40 if index < 3 else 20 if index < 6 else 0),
                    start + index,
                    definition,
                )
                for index, definition in enumerate(candidates)
                for score in [_source_page_definition_score(definition, entity)]
                if score >= 0
            )
        if not scored:
            return None
        scored.sort(key=lambda row: (-row[0], row[1]))
        self._items_used += 1
        return scored[0][2]

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

    # ------------------------------------------------------------------
    # Read-only bulk accessors (no filtering side effects).
    #
    # These return shallow copies of the loaded overlay slices so callers
    # (e.g. the cross-page graph builder) can read the full overlay without
    # reaching into private state. They do not modify overlay semantics.
    # ------------------------------------------------------------------

    def all_relations(self) -> list[dict]:
        return list(self._relations)

    def all_definitions(self) -> list[dict]:
        return list(self._definitions)

    def all_context_links(self) -> list[dict]:
        return list(self._context_links)

    def all_source_facts(self) -> list[dict]:
        return list(self._source_facts)

    def all_entities(self) -> list[dict]:
        return list(self._entities)
