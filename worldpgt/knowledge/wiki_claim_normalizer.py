"""Deterministic normalization helpers for Wikipedia Ingestion v2.

Handles:
* entity id normalization (wikidata QID -> ``wikidata:<QID>``)
* deterministic stability classification
* deterministic risk classification

All rules are explicit and keyword-based. No learning, no embeddings.
"""

from __future__ import annotations

import re

from worldpgt.knowledge.entity_types import canonicalize_entity_type
from worldpgt.knowledge.wiki_ingestion_v2_types import (
    EntityType,
    Risk,
    Stability,
    WikiPageRecord,
)

# Strict patterns that signal a volatile, time-sensitive, source-dependent
# fact: a concrete *current measurement*, not merely topical vocabulary.
# (e.g. "Forbes publishes rankings" is NOT volatile; "net worth of US$1.1
# trillion" IS.)
_AMOUNT_RE = re.compile(r"US\$\s?[\d.,]+", re.IGNORECASE)
_NET_WORTH_RE = re.compile(r"net[\s-]?worth", re.IGNORECASE)
_ESTIMATED_RE = re.compile(r"\bestimat", re.IGNORECASE)
_WEALTHIEST_RE = re.compile(
    r"\b(?:wealthiest|richest)\s+(?:person|people|man|woman|individual)\b",
    re.IGNORECASE,
)
_RANKED_RE = re.compile(
    r"\brank(?:ed|ing)?\s+(?:number\s*|no\.?\s*|#)?\d+", re.IGNORECASE
)
_CURRENT_OFFICE_RE = re.compile(
    r"\bcurrent\s+(?:ceo|chief executive|president|prime minister|office holder)\b"
    r".*\b(?:is|are|was)\b",
    re.IGNORECASE,
)
_MARKET_CAP_RE = re.compile(
    r"market cap(?:italization)?\b.*(?:US\$|\d)", re.IGNORECASE
)
_PRICE_RE = re.compile(
    r"\b(?:current|stock|share)\s+price\b.*(?:US\$|\d)", re.IGNORECASE
)
_POPULATION_RE = re.compile(
    r"\bpopulation\b.*\b(?:is|of)\b.*\d", re.IGNORECASE
)


def is_volatile_text(text: str) -> bool:
    """True only for a concrete current/source-dependent measurement.

    This is intentionally strict: topical nouns such as "rankings" or "wealth
    indexes" do not, by themselves, make a claim volatile.
    """
    t = text or ""
    if _NET_WORTH_RE.search(t) and (_AMOUNT_RE.search(t) or _ESTIMATED_RE.search(t)):
        return True
    if _WEALTHIEST_RE.search(t):
        return True
    if _RANKED_RE.search(t):
        return True
    if _CURRENT_OFFICE_RE.search(t):
        return True
    if _MARKET_CAP_RE.search(t):
        return True
    if _PRICE_RE.search(t):
        return True
    if _POPULATION_RE.search(t):
        return True
    return False


# Tokens that signal a semi-stable relation (roles / associations / products).
_SEMI_STABLE_KEYWORDS: tuple[str, ...] = (
    "leadership",
    "leader",
    "ceo of",
    "known for",
    "founded",
    "co-founded",
    "produces",
    "develops",
    "publishes",
    "manufactures",
    "associated with",
)

# Predicates that are inherently semi-stable associations.
_SEMI_STABLE_PREDICATES: frozenset[str] = frozenset(
    {"known_for", "leader_of", "founded", "produces", "develops", "publishes"}
)

# Predicates that are inherently stable structural facts.
_STABLE_PREDICATES: frozenset[str] = frozenset({"is_a", "part_of"})


_QID_RE = re.compile(r"^Q\d+$")


def normalize_entity_id(wikidata_id: str | None, title: str) -> str:
    """Return a stable entity id.

    Prefer ``wikidata:<QID>`` when a valid QID exists, otherwise fall back to a
    normalized page slug. Never loses the human label (kept separately).
    """
    if wikidata_id:
        qid = wikidata_id.strip()
        if _QID_RE.match(qid):
            return f"wikidata:{qid}"
    return f"wiki:{normalize_title_slug(title)}"


def normalize_title_slug(title: str) -> str:
    """Normalize a page title into a consistent slug (deterministic)."""
    slug = title.strip().replace(" ", "_")
    # Collapse repeated underscores and strip surrounding punctuation noise.
    slug = re.sub(r"_+", "_", slug)
    return slug.strip("_")


def classify_stability(text: str, predicate: str | None = None) -> Stability:
    """Classify the stability of a claim deterministically.

    Structural predicates are stable; association predicates are semi_stable;
    a concrete current measurement is volatile; otherwise default to stable for
    definitions and semi_stable for keyword-bearing relations.
    """
    if predicate in _STABLE_PREDICATES:
        return "stable"
    if predicate in _SEMI_STABLE_PREDICATES:
        return "semi_stable"
    if is_volatile_text(text):
        return "volatile"
    lowered = (text or "").lower()
    if any(kw in lowered for kw in _SEMI_STABLE_KEYWORDS):
        return "semi_stable"
    return "stable"


def classify_risk(stability: Stability, predicate: str | None = None) -> Risk:
    """Map stability (and predicate) onto a deterministic risk level."""
    if stability == "volatile":
        return "high"
    if stability == "semi_stable":
        return "medium"
    # Stable structural facts / definitions / entity cards.
    return "low"


# ---------------------------------------------------------------------------
# Entity type inference
# ---------------------------------------------------------------------------

_LEAD_TYPE_PATTERNS: tuple[tuple[str, EntityType], ...] = (
    ("is a magazine", "publication"),
    ("is a publication", "publication"),
    ("is a newspaper", "publication"),
    ("is a businessman", "person"),
    ("is a businessperson", "person"),
    ("is a businesswoman", "person"),
    ("is an entrepreneur", "person"),
    ("is a politician", "person"),
    ("is a company", "organization"),
    ("is an american electric vehicle", "organization"),
    ("is an aerospace", "organization"),
    ("is a corporation", "organization"),
    ("is an organization", "organization"),
    ("is a concept", "concept"),
    ("is a practice", "concept"),
    ("is the ability", "concept"),
    ("is a technology", "technology"),
    ("is a vehicle", "vehicle"),
    ("is a type of vehicle", "vehicle"),
    ("is a product", "product"),
    ("is a program", "program"),
    ("is a project", "program"),
    ("is a city", "place"),
    ("is a country", "place"),
)

_INFOBOX_TYPE_HINTS: tuple[tuple[str, str, EntityType], ...] = (
    ("occupation", "businessman", "person"),
    ("occupation", "entrepreneur", "person"),
    ("born", "", "person"),
    ("industry", "", "organization"),
    ("type", "company", "organization"),
    ("type", "public", "organization"),
    ("founded", "", "organization"),
)

_CATEGORY_TYPE_HINTS: tuple[tuple[str, EntityType], ...] = (
    ("living people", "person"),
    ("businesspeople", "person"),
    ("people", "person"),
    ("companies", "organization"),
    ("manufacturers", "organization"),
    ("magazines", "publication"),
    ("publications", "publication"),
    ("technology", "technology"),
    ("vehicles", "vehicle"),
    ("programs", "program"),
    ("cities", "place"),
    ("countries", "place"),
)


def infer_entity_type(page: WikiPageRecord) -> EntityType:
    """Infer entity type deterministically from hint, lead, infobox, categories."""
    # 1. Explicit hint wins.
    hint = (page.entity_type_hint or "").strip().lower()
    canonical_hint = canonicalize_entity_type(hint)
    if canonical_hint is not None:
        return canonical_hint

    lead = page.lead_paragraph.lower()

    # 2. Lead-sentence patterns.
    for pattern, etype in _LEAD_TYPE_PATTERNS:
        if pattern in lead:
            return etype

    # 3. Infobox field hints.
    lowered_infobox = {k.lower(): str(v).lower() for k, v in page.infobox.items()}
    for field_name, needle, etype in _INFOBOX_TYPE_HINTS:
        if field_name in lowered_infobox:
            value = lowered_infobox[field_name]
            if needle == "" or needle in value:
                return etype

    # 4. Category hints.
    lowered_cats = [c.lower() for c in page.categories]
    for needle, etype in _CATEGORY_TYPE_HINTS:
        if any(needle in c for c in lowered_cats):
            return etype

    return "other"
