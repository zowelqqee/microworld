"""Delta Quality Firewall for Knowledge Pump v1.3.

Separates pump delta items into quality buckets:
- answerable: clean entities, definitions, and high-confidence relations
- weak_context: overlay_context_link items with trust=weak_context_only
- entity: entities and definitions (subset of answerable)
- rejected: items failing quality checks (self-relations, generic subjects, etc.)
- quarantine: volatile/current/date-heavy relation claims

This module is purely classifying – it never writes files, never calls the
network, and never modifies accepted/promoted/runtime memory.
"""

from __future__ import annotations

from typing import Any

# Subjects that are too generic to yield answerable knowledge as relation subjects.
# Only exact normalized matches are checked.
_GENERIC_SUBJECTS = frozenset({
    "company",
    "corporation",
    "vehicle",
    "electric vehicle",
    "product",
    "internet service provider",
    "satellite constellation",
    "business",
    "technology",
    "model",
    "organization",
})

# Evidence phrases that indicate the relation is negated or non-affirming.
_NEGATION_PHRASES = (
    "never developed",
    "not developed",
    "considered but never",
    "planned but not",
    "never built",
    "never produced",
    "not produced",
)

# Evidence markers indicating a claim is volatile, current, or date-heavy.
# All strings are lower-cased; they are checked against lower-cased evidence.
_VOLATILE_MARKERS = (
    "currently",
    "today",
    "latest",
    "2024",
    "2025",
    "2026",
    "signed a contract",
    "per month",
    "valuation",
    "stock price",
    "market cap",
    "revenue",
    "net worth",
    "worth $",
)

# is_a object phrases that are too broad or truncated to be answerable.
_BAD_IS_A_OBJECTS = frozenset({
    "artificial intelligence",
    "machine learning",
    "computer science",
    "information technology",
    "deep learning",
    "electric vehicle",
    "renewable energy",
    "social media",
    "e-commerce",
    "cloud computing",
})

# Nationality adjectives that prefix truncated is_a objects
# (e.g. "American infrastructure" is truncated; "American aerospace manufacturer" is fine).
_NATIONALITY_ADJECTIVES = frozenset({
    "american", "chinese", "european", "british", "german",
    "japanese", "korean", "indian", "canadian", "australian",
    "french", "italian", "spanish", "russian", "israeli",
})


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def is_weak_context(item: dict[str, Any]) -> bool:
    """Return True if this item is a weak context link (answerable-knowledge-quality boundary)."""
    return (
        item.get("overlay_type") == "overlay_context_link"
        and item.get("trust") == "weak_context_only"
    )


# Private alias kept for internal use.
_is_weak_context = is_weak_context


def _is_entity_or_definition(item: dict[str, Any]) -> bool:
    return item.get("overlay_type") in ("overlay_entity", "overlay_definition")


def _check_relation_quality(item: dict[str, Any]) -> tuple[str, str]:
    """Return (verdict, reason). verdict is 'accept', 'reject', or 'quarantine'."""
    if item.get("overlay_type") != "overlay_relation":
        return "accept", ""

    subj = _norm(item.get("subject", ""))
    obj = _norm(item.get("object", ""))
    pred = _norm(item.get("predicate", ""))
    evidence = _norm(item.get("evidence_text", ""))

    # Rule 1: Reject self-relations (subject == object).
    if subj and obj and subj == obj:
        return "reject", "self_relation"

    # Rule 3: Reject generic bad subjects.
    if subj in _GENERIC_SUBJECTS:
        return "reject", "generic_subject"

    # Rule 6: Reject negated/non-affirming development contexts.
    for phrase in _NEGATION_PHRASES:
        if phrase in evidence:
            return "reject", "negated_context"

    # Rule 7: Quarantine volatile/current/date-heavy claims.
    for marker in _VOLATILE_MARKERS:
        if marker in evidence:
            return "quarantine", "volatile_evidence"

    # Rule 4: Reject weak/truncated is_a objects.
    if pred == "is_a":
        if obj in _BAD_IS_A_OBJECTS:
            return "reject", "is_a_object_too_broad"
        words = obj.split()
        if len(words) >= 2 and words[0] in _NATIONALITY_ADJECTIVES:
            return "reject", "is_a_object_truncated_nationality"

    # Rule 5: headquartered_in – object must be present and not equal to subject.
    if pred == "headquartered_in" and (not obj or obj == subj):
        return "reject", "headquartered_in_bad_object"

    return "accept", ""


def classify_pump_delta(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Classify pump delta items into quality buckets.

    Returns a dict with keys:
        answerable  – entities + definitions + high-confidence relations
        weak_context – overlay_context_link items with weak_context_only trust
        entity      – entities and definitions (subset of answerable)
        rejected    – list of {"item": ..., "reason": ...} dicts
        quarantine  – list of {"item": ..., "reason": ...} dicts
        rejection_by_reason – {reason: count} covering rejected + quarantine
    """
    answerable: list[dict[str, Any]] = []
    weak_context: list[dict[str, Any]] = []
    entity: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    rejection_by_reason: dict[str, int] = {}

    for item in items:
        if _is_weak_context(item):
            weak_context.append(item)
            continue

        if _is_entity_or_definition(item):
            entity.append(item)
            answerable.append(item)
            continue

        verdict, reason = _check_relation_quality(item)
        if verdict == "reject":
            rejected.append({"item": item, "reason": reason})
            rejection_by_reason[reason] = rejection_by_reason.get(reason, 0) + 1
        elif verdict == "quarantine":
            quarantine.append({"item": item, "reason": reason})
            rejection_by_reason[reason] = rejection_by_reason.get(reason, 0) + 1
        else:
            answerable.append(item)

    return {
        "answerable": answerable,
        "weak_context": weak_context,
        "entity": entity,
        "rejected": rejected,
        "quarantine": quarantine,
        "rejection_by_reason": dict(sorted(rejection_by_reason.items())),
    }
