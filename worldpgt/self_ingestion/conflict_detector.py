"""Conflict / risk detector for Wikipedia Self-Ingestion v1.

Deterministic, rule-based detection of risky or conflicting candidates. Operates
on (a) free-text raw claims and (b) structured overlay items compared against the
existing overlay. Never weakens; only flags. No ML, no network.
"""

from __future__ import annotations

import re

from worldpgt.self_ingestion.types import (
    REASON_CONFLICTS_EXISTING,
    REASON_CURRENT_NO_AS_OF,
    REASON_ENTITY_TYPE_MISMATCH,
    REASON_INVERTED,
    REASON_PRIVATE,
    REASON_SOURCE_MISSING_VOLATILE,
    REASON_UNIVERSAL,
    REASON_VOLATILE_NEEDS_REVIEW,
    REASON_WEAK_LINK_PROMOTED,
)

# Known people / non-person entities (consistent with the overlay island).
_PEOPLE = (
    "elon musk", "jeff bezos", "bernard arnault", "michael bloomberg",
    "gwynne shotwell", "martin eberhard", "marc tarpenning", "larry ellison",
)
_NON_PERSON = (
    "spacex", "tesla", "forbes", "bloomberg news", "bloomberg lp", "neuralink",
    "the boring company", "xai", "openai", "nasa", "blue origin", "rocket",
    "rockets", "electric vehicle", "electric vehicles", "starlink", "falcon 9",
    "falcon 1", "oracle",
)

_PRIVATE_RE = re.compile(
    r"\b(private\s+email|personal\s+email|phone\s+number|home\s+address|"
    r"private\s+address|password|social\s+security|date\s+of\s+birth|@)\b",
    re.IGNORECASE,
)
_CURRENT_RE = re.compile(
    r"\b(stock\s+price|share\s+price|market\s+cap(?:italization)?|valuation|"
    r"currently|right\s+now|today|latest|richest|wealthiest)\b",
    re.IGNORECASE,
)
_AS_OF_RE = re.compile(r"\b(as of|according to)\b", re.IGNORECASE)
_WEAK_LINK_RE = re.compile(
    r"\b(weak\s+link|wiki\s+link|context\s+link)\b.*\b(prove|proves|means|owned)\b",
    re.IGNORECASE,
)
_UNIVERSAL_RE = re.compile(
    r"\b(all|every|any|each)\b\s+\w+.*\b(are|is)\b|"
    r"\b(are|is)\b\s+(all|every|any|each)\b",
    re.IGNORECASE,
)
_FOUNDED_RE = re.compile(r"\b(.+?)\s+founded\s+(.+?)[\.\,;]?$", re.IGNORECASE)


def _has(text: str, terms) -> bool:
    low = text.lower()
    return any(t in low for t in terms)


def detect_raw_claim(text: str) -> str | None:
    """Return a quarantine reason enum for a risky raw claim, else None."""
    t = (text or "").strip()
    low = t.lower()

    # Private / sensitive data.
    if _PRIVATE_RE.search(t):
        return REASON_PRIVATE

    # Weak-link-as-proof.
    if _WEAK_LINK_RE.search(t):
        return REASON_WEAK_LINK_PROMOTED

    # Inverted founder relation: "<non-person> founded <person>".
    m = _FOUNDED_RE.search(t)
    if m:
        subj, obj = m.group(1).strip().lower(), m.group(2).strip().lower()
        subj_is_person = any(p in subj for p in _PEOPLE)
        obj_is_person = any(p in obj for p in _PEOPLE)
        subj_is_org = any(o in subj for o in _NON_PERSON)
        if obj_is_person and (subj_is_org or not subj_is_person):
            return REASON_INVERTED

    # Universal / generalization (checked before generic current).
    if _UNIVERSAL_RE.search(t):
        return REASON_UNIVERSAL

    # Current / live measurement without as_of / source.
    if _CURRENT_RE.search(t) and not _AS_OF_RE.search(t):
        return REASON_CURRENT_NO_AS_OF

    return None


def detect_overlay_item(item: dict, existing: "ExistingOverlayIndex") -> str | None:
    """Return a quarantine/hold reason for a structured overlay item, else None."""
    otype = item.get("overlay_type", "")

    if otype == "overlay_source_fact":
        if not item.get("source_name") or not item.get("as_of"):
            return REASON_SOURCE_MISSING_VOLATILE
        if not item.get("requires_recheck", False) or item.get("stability") != "volatile":
            return REASON_SOURCE_MISSING_VOLATILE
        # Well-formed but volatile: must be held for human review, not auto-applied.
        return REASON_VOLATILE_NEEDS_REVIEW

    if otype == "overlay_entity":
        prev = existing.entity_type(item.get("label", ""))
        if prev is not None and prev != item.get("entity_type"):
            return REASON_ENTITY_TYPE_MISMATCH

    if otype == "overlay_definition":
        prev = existing.definition(item.get("subject", ""))
        if prev is not None and _norm(prev) != _norm(item.get("definition", "")):
            return REASON_CONFLICTS_EXISTING

    if otype == "overlay_context_link":
        if item.get("strength") != "weak" or item.get("trust") != "weak_context_only":
            return REASON_WEAK_LINK_PROMOTED

    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


class ExistingOverlayIndex:
    """Read-only index over the existing overlay for conflict checks."""

    def __init__(self, overlay_items: list[dict]) -> None:
        self._entity_type: dict[str, str] = {}
        self._definition: dict[str, str] = {}
        self._keys: set[tuple] = set()
        self._subj_pred: dict[tuple[str, str], set[str]] = {}
        for it in overlay_items:
            t = it.get("overlay_type", "")
            if t == "overlay_entity":
                self._entity_type[_norm(it.get("label", ""))] = it.get("entity_type", "")
                self._keys.add(("entity", _norm(it.get("label", ""))))
            elif t == "overlay_definition":
                self._definition[_norm(it.get("subject", ""))] = it.get("definition", "")
                self._keys.add(("definition", _norm(it.get("subject", ""))))
            elif t == "overlay_relation":
                key = ("relation", _norm(it.get("subject", "")), it.get("predicate", ""),
                       _norm(it.get("object", "")))
                self._keys.add(key)
                self._subj_pred.setdefault(
                    (_norm(it.get("subject", "")), it.get("predicate", "")), set()
                ).add(_norm(it.get("object", "")))
            elif t == "overlay_context_link":
                self._keys.add(("context_link", _norm(it.get("source_page", "")),
                                _norm(it.get("target", ""))))
            elif t == "overlay_source_fact":
                self._keys.add(("source_fact", _norm(it.get("subject", "")),
                                it.get("predicate", "")))

    def entity_type(self, label: str):
        return self._entity_type.get(_norm(label))

    def definition(self, subject: str):
        return self._definition.get(_norm(subject))

    def has_key(self, key: tuple) -> bool:
        return key in self._keys
