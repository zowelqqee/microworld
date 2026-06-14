"""Core dataclasses and deterministic enums for Wikipedia Self-Ingestion v1.

Rule-based and offline only. No ML, no embeddings, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# --- known vocabularies -------------------------------------------------
SOURCE_TYPES = ("local_wikipedia_like_text", "local_wikipedia_like_json", "unknown")
TRUST_TIERS = ("curated", "untrusted", "test_adversarial")

# --- ingestion delta classifications -----------------------------------
NEW_CANDIDATE = "new_candidate"
DUPLICATE_EXISTING = "duplicate_existing"
CONFLICT_EXISTING = "conflict_existing"
WEAKER_DUPLICATE = "weaker_duplicate"
VOLATILE_REQUIRES_SOURCE = "volatile_requires_source"
QUARANTINE = "quarantine"
REJECTED = "rejected"

# --- quarantine reason enums (deterministic) ---------------------------
REASON_CURRENT_NO_AS_OF = "current_fact_without_as_of"
REASON_PRIVATE = "private_or_sensitive_data"
REASON_INVERTED = "inverted_relation"
REASON_WEAK_LINK_PROMOTED = "weak_link_promoted_to_fact"
REASON_CONFLICTS_EXISTING = "conflicts_existing_fact"
REASON_UNIVERSAL = "unsupported_universal_claim"
REASON_SOURCE_MISSING_VOLATILE = "source_missing_for_volatile_fact"
REASON_ENTITY_TYPE_MISMATCH = "entity_type_mismatch"
REASON_UNKNOWN_FORMAT = "unknown_or_unsupported_format"
REASON_VOLATILE_NEEDS_REVIEW = "volatile_requires_source"

QUARANTINE_REASONS = (
    REASON_CURRENT_NO_AS_OF, REASON_PRIVATE, REASON_INVERTED,
    REASON_WEAK_LINK_PROMOTED, REASON_CONFLICTS_EXISTING, REASON_UNIVERSAL,
    REASON_SOURCE_MISSING_VOLATILE, REASON_ENTITY_TYPE_MISMATCH,
    REASON_UNKNOWN_FORMAT, REASON_VOLATILE_NEEDS_REVIEW,
)

# suggested actions
ACTION_REJECT = "reject"
ACTION_HUMAN_REVIEW = "human_review"
ACTION_NEEDS_SOURCE = "needs_source"
ACTION_NEEDS_AS_OF = "needs_as_of"
ACTION_KEEP_WEAK_ONLY = "keep_weak_only"

# reason -> (risk, suggested_action)
REASON_POLICY: dict[str, tuple[str, str]] = {
    REASON_INVERTED: ("high", ACTION_REJECT),
    REASON_PRIVATE: ("high", ACTION_REJECT),
    REASON_WEAK_LINK_PROMOTED: ("high", ACTION_REJECT),
    REASON_UNIVERSAL: ("high", ACTION_REJECT),
    REASON_CURRENT_NO_AS_OF: ("high", ACTION_NEEDS_AS_OF),
    REASON_SOURCE_MISSING_VOLATILE: ("high", ACTION_NEEDS_SOURCE),
    REASON_CONFLICTS_EXISTING: ("medium", ACTION_HUMAN_REVIEW),
    REASON_ENTITY_TYPE_MISMATCH: ("medium", ACTION_HUMAN_REVIEW),
    REASON_VOLATILE_NEEDS_REVIEW: ("high", ACTION_HUMAN_REVIEW),
    REASON_UNKNOWN_FORMAT: ("low", ACTION_HUMAN_REVIEW),
}


@dataclass
class SourceEntry:
    source_id: str
    path: str
    source_type: str
    trust_tier: str
    expected_domain: str
    allow_auto_accept: bool
    notes: str = ""


@dataclass
class ManifestLoadResult:
    sources: list[SourceEntry] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)  # {source_id, reason}


@dataclass
class NormalizedDocument:
    source_id: str
    title: str
    raw_text: str
    source_type: str
    trust_tier: str
    source_path: str
    chunks: list[str] = field(default_factory=list)
    read_errors: list[str] = field(default_factory=list)


@dataclass
class RawClaim:
    """A free-text claim that did not parse into a structured page."""

    source_id: str
    trust_tier: str
    text: str


@dataclass
class ClassifiedItem:
    """One overlay item produced from a source, with its delta classification."""

    source_id: str
    classification: str
    overlay_item: dict


@dataclass
class QuarantineItem:
    candidate_id: str
    text: str
    reason: str
    risk: str
    source_id: str
    suggested_action: str

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "text": self.text,
            "reason": self.reason,
            "risk": self.risk,
            "source_id": self.source_id,
            "suggested_action": self.suggested_action,
        }
