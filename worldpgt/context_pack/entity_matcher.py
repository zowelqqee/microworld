"""Deterministic, string-based entity matcher for Working Context Pack v1.

No embeddings, no fuzzy/neural matching. It matches surfaces drawn ONLY from
the overlay (entity labels, overlay-present aliases, and the subject/object
terms that appear in definitions / relations / source facts) against the
question, using case-insensitive word-boundary matching. Longer surfaces win,
and tiny / generic tokens are never matched alone.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .types import MatchedEntity

# Pre-compiled pattern cache keyed by normalised surface string.
# Python's built-in re cache holds only 512 entries; with ~1700 overlay
# surfaces it evicts and recompiles on every match_entities call.  This
# unbounded dict persists for the process lifetime (overlays are read-only).
_PATTERN_CACHE: Dict[str, re.Pattern] = {}


def _get_pattern(norm_surface: str) -> re.Pattern:
    try:
        return _PATTERN_CACHE[norm_surface]
    except KeyError:
        p = re.compile(r"(?<![\w])" + re.escape(norm_surface) + r"(?![\w])")
        _PATTERN_CACHE[norm_surface] = p
        return p

# Generic tokens that must never be matched as entities on their own.
_GENERIC_STOPWORDS = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "is",
        "are",
        "to",
        "and",
        "or",
        "in",
        "on",
        "by",
        "for",
        "current",
        "stock",
        "price",
        "today",
        "latest",
        "news",
        "company",
        "ranking",
        "rankings",
        "businessman",
        "leader",
        "satellite",
    }
)

_MIN_SURFACE_LEN = 2


def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    # Drop possessive markers so "SpaceX's" matches the "SpaceX" surface.
    words = []
    for word in s.split():
        if word.endswith("'s") or word.endswith("`s"):
            word = word[:-2]
        elif word.endswith("s'") or word.endswith("s`"):
            word = word[:-1]
        word = word.replace("'", "").replace("`", "")
        if word:
            words.append(word)
    return " ".join(words)


def _is_generic(surface: str) -> bool:
    n = _norm(surface)
    if len(n) < _MIN_SURFACE_LEN:
        return True
    # A single short generic word is rejected; multi-word surfaces are kept.
    if " " not in n and n in _GENERIC_STOPWORDS:
        return True
    return False


def build_surface_index(overlay) -> List[Tuple[str, str, dict]]:
    """Return [(surface, kind, meta)] sorted by surface length descending.

    ``kind`` is "entity" for overlay entities/aliases, else "overlay_term".
    """
    surfaces: Dict[str, Tuple[str, dict]] = {}

    def _add(surface: str, kind: str, meta: dict) -> None:
        surface = (surface or "").strip()
        if not surface or _is_generic(surface):
            return
        key = _norm(surface)
        # Entity surfaces take precedence over plain terms.
        if key not in surfaces or (kind == "entity" and surfaces[key][0] != "entity"):
            surfaces[key] = (kind, {"surface": surface, **meta})

    for e in overlay.entities():
        _add(e.get("label", ""), "entity", {
            "name": e.get("label", ""),
            "entity_id": e.get("entity_id"),
            "entity_type": e.get("entity_type"),
        })
        for alias in e.get("aliases", []) or []:
            _add(alias, "entity", {
                "name": e.get("label", ""),
                "entity_id": e.get("entity_id"),
                "entity_type": e.get("entity_type"),
            })

    # Subject/object terms that are not entities (e.g. "rockets", "spacecraft").
    for d in overlay.definitions():
        _add(d.get("subject", ""), "overlay_term", {"name": d.get("subject", "")})
    for r in overlay.relations():
        _add(r.get("subject", ""), "overlay_term", {"name": r.get("subject", "")})
        _add(r.get("object", ""), "overlay_term", {"name": r.get("object", "")})
    for sf in overlay.source_facts():
        _add(sf.get("subject", ""), "overlay_term", {"name": sf.get("subject", "")})

    indexed = [(meta["surface"], kind, meta) for kind, meta in surfaces.values()]
    indexed.sort(key=lambda t: len(t[0]), reverse=True)
    return indexed


def match_entities(
    question: str,
    overlay,
    *,
    prebuilt_index: List[Tuple[str, str, dict]] | None = None,
) -> List[MatchedEntity]:
    surfaces = prebuilt_index if prebuilt_index is not None else build_surface_index(overlay)
    qlow = " " + _norm(question) + " "
    consumed_spans: List[Tuple[int, int]] = []
    matched: List[MatchedEntity] = []
    seen_names: set = set()

    for surface, kind, meta in surfaces:
        n = _norm(surface)
        if n not in qlow:
            continue
        # Word-boundary search over the normalized question.
        m = _get_pattern(n).search(qlow)
        if not m:
            continue
        start, end = m.start(), m.end()
        # Skip if this span is inside an already consumed (longer) match.
        if any(s <= start and end <= e for (s, e) in consumed_spans):
            continue
        name = meta.get("name") or surface
        dedup_key = (_norm(name), kind if kind == "overlay_term" else "entity")
        if dedup_key in seen_names:
            continue
        seen_names.add(dedup_key)
        consumed_spans.append((start, end))
        matched.append(
            MatchedEntity(
                surface=surface,
                name=name,
                matched_kind=kind,
                entity_id=meta.get("entity_id"),
                entity_type=meta.get("entity_type"),
            )
        )

    # Stable order: by first appearance position in the question.
    matched.sort(key=lambda me: qlow.find(_norm(me.surface)))
    return matched
