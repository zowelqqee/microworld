"""Entity type classifier — derives entity type from definition text.

Maps is_a / definition text to a canonical entity type via keyword rules.
No ML, no network. Deterministic.

Types
-----
person        — individual human being
organization  — company, corporation, agency, institution
publication   — magazine, newspaper, publisher, journal
product       — product, device, model
service       — internet service, constellation, network, platform, subscription, access
vehicle       — rocket, spacecraft, launch vehicle, car, truck, probe
program       — spaceflight program, project, mission, initiative
place         — city, country, region, island, location
concept       — theory, index, period, phenomenon
technology    — technology, system, method, protocol
other         — default when no rule fires
"""

from __future__ import annotations

import re

from worldpgt.knowledge.entity_types import CanonicalEntityType, canonicalize_entity_type

# ---------------------------------------------------------------------------
# Keyword tables (order matters: first match wins)
# ---------------------------------------------------------------------------

_RULES: list[tuple[CanonicalEntityType, frozenset[str]]] = [
    (
        "person",
        frozenset({
            "entrepreneur", "businessman", "businesswoman",
            "engineer", "scientist", "physicist", "chemist", "biologist",
            "astronaut", "cosmonaut", "pilot",
            "ceo", "chief executive", "executive",
            "founder", "co-founder",
            "investor", "venture capitalist",
            "author", "writer", "journalist",
            "politician", "president", "senator", "governor",
            "activist", "philanthropist",
            "born in", "born on",           # biographical markers
        }),
    ),
    (
        "organization",
        frozenset({
            "company", "corporation", "inc.", "corp.", "ltd.", "llc",
            "manufacturer", "manufacturer of",
            "aerospace company", "automotive company",
            "startup", "firm", "enterprise",
            "agency", "authority", "administration",
            "institution", "institute", "university", "foundation",
            "conglomerate", "holding", "group",
            "organization", "organisation",
            "studio",
            "provider", "operator",
        }),
    ),
    (
        "publication",
        frozenset({
            "magazine", "newspaper", "publication", "journal",
            "publisher", "news outlet",
        }),
    ),
    (
        "service",
        frozenset({
            "internet service", "constellation", "network", "platform",
            "subscription", "access",
        }),
    ),
    (
        "product",
        frozenset({
            "product", "offering", "device", "appliance",
            "model", "application", "app",
        }),
    ),
    (
        "vehicle",
        frozenset({
            "rocket", "launch vehicle", "orbital launch vehicle",
            "spacecraft", "space vehicle", "space capsule",
            "satellite", "probe", "lander", "rover",
            "reusable rocket", "reusable launch vehicle",
            "heavy-lift", "medium-lift", "smallsat",
            "truck", "semi truck", "pickup truck", "electric truck",
            "car", "electric car", "electric vehicle",
            "airplane", "aircraft", "drone",
            "submarine", "ship", "vessel",
        }),
    ),
    (
        "program",
        frozenset({
            "spaceflight program", "space program",
            "program", "project", "initiative",
            "mission", "space mission",
            "campaign", "operation",
        }),
    ),
    (
        "place",
        frozenset({
            "city", "town", "village",
            "country", "nation", "state",
            "region", "territory", "province",
            "island", "peninsula", "continent",
            "planet", "moon",
            "facility", "launch site", "spaceport", "launch complex",
            "headquarters", "campus",
            "ocean", "sea", "lake", "river",
        }),
    ),
    (
        "concept",
        frozenset({
            "theory", "concept", "principle",
            "index", "metric", "measure",
            "event", "period", "era",
            "phenomenon", "process",
            "currency", "cryptocurrency",
        }),
    ),
    (
        "technology",
        frozenset({
            "technology", "technique", "method",
            "system", "framework", "standard", "protocol",
        }),
    ),
]

# Pre-compiled: each rule becomes a single alternation regex for speed.
_COMPILED: list[tuple[CanonicalEntityType, re.Pattern[str]]] = []
for _etype, _keywords in _RULES:
    _pattern = "|".join(re.escape(kw) for kw in sorted(_keywords, key=len, reverse=True))
    _COMPILED.append((_etype, re.compile(r"\b(?:" + _pattern + r")\b", re.IGNORECASE)))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_entity_type(definition: str) -> CanonicalEntityType:
    """Return the entity type that best matches *definition* text.

    Tries each rule in priority order. Returns the first match.

    Args:
        definition: Free-text definition / is_a string from the overlay.

    Returns:
        A canonical entity type string.
    """
    if not definition:
        return "other"
    text = definition.strip()
    for etype, pattern in _COMPILED:
        if pattern.search(text):
            return etype
    return "other"


def infer_entity_type(entity_label: str, overlay_items: list[dict]) -> CanonicalEntityType:
    """Derive entity type from overlay items for *entity_label*.

    Searches overlay_items for overlay_definition or overlay_entity records
    whose subject/label matches entity_label. Feeds the definition text into
    classify_entity_type(). Falls back to "other" if nothing is found.

    Args:
        entity_label: Normalised entity name to look up.
        overlay_items: List of overlay item dicts (each has "overlay_type" key).

    Returns:
        Entity type string.
    """
    label_lower = entity_label.lower().strip()

    # 1. Explicit entity_type on overlay_entity items.
    for item in overlay_items:
        if item.get("overlay_type") == "overlay_entity":
            if item.get("label", "").lower().strip() == label_lower:
                existing = (item.get("entity_type") or "").strip()
                canonical_existing = canonicalize_entity_type(existing)
                if canonical_existing is not None:
                    return canonical_existing

    # 2. Definition text from overlay_definition items.
    for item in overlay_items:
        if item.get("overlay_type") == "overlay_definition":
            subject = (item.get("subject") or "").lower().strip()
            if subject == label_lower:
                definition = item.get("definition", "")
                if definition:
                    return classify_entity_type(definition)

    # 3. is_a relation from overlay_relation items.
    for item in overlay_items:
        if (
            item.get("overlay_type") == "overlay_relation"
            and item.get("predicate") in ("is_a", "type_of")
            and (item.get("subject") or "").lower().strip() == label_lower
        ):
            object_text = item.get("object", "")
            if object_text:
                return classify_entity_type(object_text)

    return "other"
