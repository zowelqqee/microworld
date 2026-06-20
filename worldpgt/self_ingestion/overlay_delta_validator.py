"""Overlay delta validator for Promote Overlay Delta v1.

Re-validates self-ingestion overlay delta items immediately before promotion.
Only safe, stable, non-conflicting, non-duplicate items are accepted. Every
risky / volatile / private / inverted / universal / weak-link-as-fact /
conflicting / malformed / duplicate item is blocked.

Deterministic, rule-based, offline only. No ML, no embeddings, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from worldpgt.self_ingestion.conflict_detector import ExistingOverlayIndex
from worldpgt.self_ingestion.ingestion_delta import item_key
from worldpgt.knowledge.temporal_classification import (
    classify_temporal_class,
    requires_as_of,
)

# --- block reason enums -------------------------------------------------
REASON_MALFORMED = "malformed_overlay_item"
REASON_MISSING_FIELD = "missing_required_field"
REASON_UNKNOWN_TYPE = "unknown_overlay_type"
REASON_HIGH_RISK = "high_risk"
REASON_VOLATILE = "volatile_or_source_qualified_fact"
REASON_REQUIRES_RECHECK = "requires_recheck_true"
REASON_PRIVATE = "private_or_sensitive_data"
REASON_INVERTED = "inverted_relation"
REASON_UNIVERSAL = "unsupported_universal_claim"
REASON_WEAK_LINK_PROMOTED = "weak_link_promoted_to_fact"
REASON_CONFLICTS_EXISTING = "conflicts_existing_fact"
REASON_ENTITY_TYPE_MISMATCH = "entity_type_mismatch"
REASON_DUPLICATE_EXISTING = "duplicate_existing_item"
REASON_DUPLICATE_IN_DELTA = "duplicate_in_delta"
REASON_SNAPSHOT_REQUIRES_AS_OF = "snapshot_requires_as_of"
REASON_AGGREGATE_REQUIRES_AS_OF = "aggregate_requires_as_of"
REASON_TEMPORAL_CLASS_REVIEW = "temporal_class_requires_review"

_SAFETY_REASONS = frozenset({
    REASON_HIGH_RISK, REASON_VOLATILE, REASON_REQUIRES_RECHECK, REASON_PRIVATE,
    REASON_INVERTED, REASON_UNIVERSAL, REASON_WEAK_LINK_PROMOTED,
    REASON_CONFLICTS_EXISTING, REASON_ENTITY_TYPE_MISMATCH,
    REASON_SNAPSHOT_REQUIRES_AS_OF, REASON_AGGREGATE_REQUIRES_AS_OF,
})

_KNOWN_TYPES = ("overlay_entity", "overlay_definition", "overlay_relation",
                "overlay_context_link", "overlay_source_fact")
_REQUIRED_FIELDS = {
    "overlay_entity": ("entity_id", "label", "entity_type"),
    "overlay_definition": ("subject", "definition"),
    "overlay_relation": ("subject", "predicate", "object", "evidence_text"),
    "overlay_context_link": ("source_page", "surface", "target", "strength", "trust"),
    "overlay_source_fact": ("subject", "predicate", "object", "source_name",
                            "as_of", "requires_recheck"),
}

_PEOPLE = (
    "elon musk", "jeff bezos", "bernard arnault", "michael bloomberg",
    "gwynne shotwell", "martin eberhard", "marc tarpenning", "larry ellison",
)
_NON_PERSON = (
    "spacex", "tesla", "forbes", "bloomberg news", "bloomberg lp", "neuralink",
    "the boring company", "xai", "openai", "nasa", "blue origin", "rocket",
    "rockets", "electric vehicle", "starlink", "falcon 9", "falcon 1", "oracle",
)
_PRIVATE_RE = re.compile(
    r"(private\s+email|personal\s+email|phone\s+number|home\s+address|password|"
    r"social\s+security|date\s+of\s+birth|@)",
    re.IGNORECASE,
)
_UNIVERSAL_RE = re.compile(
    r"\b(all|every|any|each)\b\s+\w+.*\b(are|is)\b|\b(are|is)\b\s+(all|every|any|each)\b",
    re.IGNORECASE,
)


@dataclass
class ValidationResult:
    accepted_count: int = 0
    rejected_count: int = 0
    blocked_count: int = 0
    accepted_item_ids: list[str] = field(default_factory=list)
    accepted_items: list[dict] = field(default_factory=list)
    rejected_items: list[dict] = field(default_factory=list)  # {id, reason, overlay_type}
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    safe_to_promote: bool = False

    def to_dict(self) -> dict:
        return {
            "accepted_count": self.accepted_count,
            "rejected_count": self.rejected_count,
            "blocked_count": self.blocked_count,
            "accepted_item_ids": self.accepted_item_ids,
            "rejected_items": self.rejected_items,
            "errors": self.errors,
            "warnings": self.warnings,
            "safe_to_promote": self.safe_to_promote,
        }


def _item_id(item: dict) -> str:
    return "::".join(str(p) for p in item_key(item))


def _text_blob(item: dict) -> str:
    return " ".join(str(v) for v in item.values() if isinstance(v, str))


def _classify(item: dict, existing: ExistingOverlayIndex, seen: set) -> str | None:
    """Return a block reason for one delta item, or None if safe."""
    if not isinstance(item, dict):
        return REASON_MALFORMED
    otype = item.get("overlay_type")
    if not otype:
        return REASON_MALFORMED
    if otype not in _KNOWN_TYPES:
        return REASON_UNKNOWN_TYPE
    for fld in _REQUIRED_FIELDS[otype]:
        if item.get(fld) in (None, ""):
            # requires_recheck may legitimately be False; treat presence only.
            if fld == "requires_recheck" and "requires_recheck" in item:
                continue
            return REASON_MISSING_FIELD

    # Volatile / source-qualified facts must never become stable overlay items.
    if otype == "overlay_source_fact" or item.get("stability") == "volatile":
        return REASON_VOLATILE

    temporal_class = item.get("temporal_class") or classify_temporal_class(
        item.get("predicate"),
        item.get("stability"),
        overlay_type=otype,
        claim_type=item.get("claim_type"),
    )
    if otype in {"overlay_definition", "overlay_relation"} and temporal_class is None:
        return REASON_TEMPORAL_CLASS_REVIEW
    if temporal_class == "snapshot" and not item.get("as_of"):
        return REASON_SNAPSHOT_REQUIRES_AS_OF
    if temporal_class == "aggregate" and not item.get("as_of"):
        return REASON_AGGREGATE_REQUIRES_AS_OF
    if item.get("requires_recheck") is True and not (
        temporal_class == "snapshot" and item.get("as_of")
    ):
        return REASON_REQUIRES_RECHECK
    if item.get("risk") == "high":
        return REASON_HIGH_RISK

    # Private / sensitive data anywhere in the item.
    if _PRIVATE_RE.search(_text_blob(item)):
        return REASON_PRIVATE

    # Weak links must remain weak/context-only.
    if otype == "overlay_context_link":
        if item.get("strength") != "weak" or item.get("trust") != "weak_context_only":
            return REASON_WEAK_LINK_PROMOTED

    # Relations: inverted founders & universal claims.
    if otype == "overlay_relation":
        subj = str(item.get("subject", "")).lower()
        obj = str(item.get("object", "")).lower()
        if item.get("predicate") == "founded":
            obj_is_person = any(p in obj for p in _PEOPLE)
            subj_is_person = any(p in subj for p in _PEOPLE)
            subj_is_org = any(o in subj for o in _NON_PERSON)
            if obj_is_person and (subj_is_org or not subj_is_person):
                return REASON_INVERTED
        if _UNIVERSAL_RE.search(f"{item.get('object','')} {item.get('evidence_text','')}"):
            return REASON_UNIVERSAL
        if item.get("stability") not in ("stable", "semi_stable"):
            return REASON_VOLATILE

    # Conflicts with the existing accepted overlay.
    if otype == "overlay_entity":
        prev = existing.entity_type(item.get("label", ""))
        if prev is not None and prev != item.get("entity_type"):
            return REASON_ENTITY_TYPE_MISMATCH
    if otype == "overlay_definition":
        prev = existing.definition(item.get("subject", ""))
        if prev is not None and _norm(prev) != _norm(item.get("definition", "")):
            return REASON_CONFLICTS_EXISTING

    # Duplicate collisions (semantic key).
    key = item_key(item)
    if existing.has_key(key):
        return REASON_DUPLICATE_EXISTING
    if key in seen:
        return REASON_DUPLICATE_IN_DELTA

    return None


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def validate_delta(delta_items: list, existing_items: list[dict]) -> ValidationResult:
    result = ValidationResult()
    if not isinstance(delta_items, list):
        result.errors.append("delta_proposal_not_a_list")
        result.safe_to_promote = False
        return result

    index = ExistingOverlayIndex(existing_items)
    seen: set = set()

    for item in delta_items:
        item_id = _item_id(item) if isinstance(item, dict) else f"malformed::{item!r}"
        reason = _classify(item, index, seen)
        if reason is None:
            result.accepted_count += 1
            result.accepted_item_ids.append(item_id)
            result.accepted_items.append(item)
            temporal_class = item.get("temporal_class") or classify_temporal_class(
                item.get("predicate") if isinstance(item, dict) else None,
                item.get("stability") if isinstance(item, dict) else None,
                overlay_type=item.get("overlay_type") if isinstance(item, dict) else None,
                claim_type=item.get("claim_type") if isinstance(item, dict) else None,
            )
            if temporal_class == "snapshot":
                result.warnings.append(f"{item_id}:snapshot_as_of_recheck_required")
            seen.add(item_key(item))
        else:
            result.rejected_count += 1
            if reason in _SAFETY_REASONS:
                result.blocked_count += 1
            result.rejected_items.append({
                "id": item_id,
                "overlay_type": item.get("overlay_type") if isinstance(item, dict) else None,
                "reason": reason,
            })
            if reason in (REASON_MALFORMED, REASON_MISSING_FIELD, REASON_UNKNOWN_TYPE):
                result.errors.append(f"{item_id}:{reason}")
            else:
                result.warnings.append(f"{item_id}:{reason}")

    # The whole proposal is safe to promote only if nothing was rejected and no
    # structural errors were found. (Accepted items are still promotable even if
    # this is False; this flag describes the proposal as a whole.)
    result.safe_to_promote = (
        result.rejected_count == 0 and not result.errors and result.accepted_count > 0
    )
    return result
