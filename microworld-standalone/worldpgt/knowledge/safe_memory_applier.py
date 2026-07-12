"""Safe Knowledge Memory Applier.

Builds an accepted knowledge memory artifact from accepted_auto facts and patterns,
and constructs a dry-run overlay ExplicitSenseMemory for validation.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- Live cue memory, guard rules, and engine behavior are NOT changed.
- Thresholds are NOT lowered.
- Validators are NOT weakened.
- No generic fallback is added.
- The overlay built here is used only for isolated dry-run validation.
"""

from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.knowledge.accepted_memory_types import AcceptedMemoryItem, ItemProvenance

_FACT_SOURCE_PATH = "worldpgt/experiments/knowledge_ingestion_v1_auto_review.json"
_PATTERN_SOURCE_PATH = "worldpgt/experiments/wiki_pattern_candidates_v1.json"

_EXCLUDED_REASON_PREFIXES = ("broad_", "conflicting_", "generic_")


def _normalize_value(v: str) -> str:
    return v.strip().lower()


def _dedup_key(item_kind: str, term: str, sense: str, item_type: str, value: str) -> str:
    return f"{item_kind}|{term}|{sense}|{item_type}|{_normalize_value(value)}"


def _make_item_id(item_kind: str, term: str, sense: str, item_type: str, value: str) -> str:
    key = _dedup_key(item_kind, term, sense, item_type, value)
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return f"ami-{digest[:12]}"


def _pattern_item_type(pattern_type: str) -> str:
    return f"{pattern_type}_pattern"


class SafeMemoryApplier:
    """Builds and validates accepted knowledge memory artifacts.

    Does not modify any live system state.
    """

    def __init__(self) -> None:
        self._items: list[AcceptedMemoryItem] = []
        self._fact_count = 0
        self._pattern_count = 0
        self._excluded_needs_review = 0
        self._excluded_rejected = 0
        self._excluded_high_risk = 0
        self._excluded_broad = 0
        self._dedup_count = 0

    # ------------------------------------------------------------------
    # Building from raw source files
    # ------------------------------------------------------------------

    def build_from_sources(
        self,
        auto_review_data: dict[str, Any],
        patterns_data: dict[str, Any],
    ) -> "SafeMemoryApplier":
        """Populate from raw auto-review and pattern source dicts."""
        self._items = []
        self._fact_count = self._pattern_count = 0
        self._excluded_needs_review = self._excluded_rejected = 0
        self._excluded_high_risk = self._excluded_broad = self._dedup_count = 0
        seen: set[str] = set()
        self._load_facts(auto_review_data, seen)
        self._load_patterns(patterns_data, seen)
        return self

    @classmethod
    def from_artifact(cls, artifact_data: dict[str, Any]) -> "SafeMemoryApplier":
        """Reconstruct a SafeMemoryApplier from a saved artifact dict.

        Only used for overlay-memory construction during validation.
        """
        obj = cls()
        for raw in artifact_data.get("items", []):
            prov_raw = raw.get("provenance", {})
            prov = ItemProvenance(
                source_artifact=prov_raw.get("source_artifact", ""),
                source_proposal_id=prov_raw.get("source_proposal_id"),
                source_pattern_id=prov_raw.get("source_pattern_id"),
            )
            obj._items.append(
                AcceptedMemoryItem(
                    item_id=raw["item_id"],
                    item_kind=raw["item_kind"],
                    term=raw["term"],
                    sense=raw["sense"],
                    item_type=raw["item_type"],
                    value=raw["value"],
                    source_id=raw.get("source_id", ""),
                    risk_level=raw.get("risk_level", "low"),
                    decision=raw.get("decision", "accepted_auto"),
                    provenance=prov,
                )
            )
        stats = artifact_data.get("stats", {})
        obj._fact_count = stats.get("fact_items", 0)
        obj._pattern_count = stats.get("pattern_items", 0)
        return obj

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[AcceptedMemoryItem]:
        return list(self._items)

    @property
    def stats(self) -> dict:
        by_term: dict[str, int] = {}
        by_sense: dict[str, int] = {}
        by_type: dict[str, int] = {}
        for it in self._items:
            by_term[it.term] = by_term.get(it.term, 0) + 1
            k = f"{it.term}:{it.sense}"
            by_sense[k] = by_sense.get(k, 0) + 1
            by_type[it.item_type] = by_type.get(it.item_type, 0) + 1
        return {
            "fact_items": self._fact_count,
            "pattern_items": self._pattern_count,
            "total_items": len(self._items),
            "excluded_needs_review": self._excluded_needs_review,
            "excluded_rejected_auto": self._excluded_rejected,
            "excluded_high_risk": self._excluded_high_risk,
            "excluded_broad_or_generic": self._excluded_broad,
            "deduplicated_count": self._dedup_count,
            "by_term": by_term,
            "by_sense": by_sense,
            "by_item_type": by_type,
        }

    # ------------------------------------------------------------------
    # Overlay memory builder (dry-run only)
    # ------------------------------------------------------------------

    def build_overlay_memory(self) -> ExplicitSenseMemory:
        """Return a fresh ExplicitSenseMemory with accepted positive_cue items applied.

        Only positive_cue items contribute to the scoring layer; all other
        item types are artifact metadata and are NOT applied here.
        The returned instance is independent of any live engine — for dry-run use only.
        """
        overlay_mem = ExplicitSenseMemory(include_builtin=True)

        builtin: dict[tuple[str, str], set[str]] = {}
        for term in overlay_mem.known_terms():
            for entry in overlay_mem.get_senses(term):
                builtin[(term, entry.sense_id)] = set(entry.cues)

        cues_by_ts: dict[tuple[str, str], list[str]] = defaultdict(list)
        for it in self._items:
            if it.item_type == "positive_cue":
                cues_by_ts[(it.term, it.sense)].append(_normalize_value(it.value))

        for (term, sense_id), new_cues in cues_by_ts.items():
            existing = builtin.get((term, sense_id), set())
            for entry in overlay_mem.get_senses(term):
                if entry.sense_id == sense_id:
                    for cue in new_cues:
                        if cue not in existing:
                            entry.cues.append(cue)
                            existing.add(cue)
        return overlay_mem

    # ------------------------------------------------------------------
    # Internal loaders
    # ------------------------------------------------------------------

    def _load_facts(self, data: dict[str, Any], seen: set[str]) -> None:
        for pr in data.get("proposal_reviews", []):
            term = pr["term"]
            sense = pr["sense"]
            proposal_id = pr.get("proposal_id", "")
            source_id = f"auto_review:{proposal_id}"
            for item in pr.get("items", []):
                dec = item.get("decision", "")
                if dec == "needs_review":
                    self._excluded_needs_review += 1
                    continue
                if dec == "rejected_auto":
                    self._excluded_rejected += 1
                    continue
                if dec != "accepted_auto":
                    continue
                if item.get("risk_level", "low") == "high":
                    self._excluded_high_risk += 1
                    continue
                reason = item.get("reason", "")
                if any(reason.startswith(p) for p in _EXCLUDED_REASON_PREFIXES):
                    self._excluded_broad += 1
                    continue
                item_type = item.get("item_type", "")
                value = item.get("value", "")
                key = _dedup_key("fact", term, sense, item_type, value)
                if key in seen:
                    self._dedup_count += 1
                    continue
                seen.add(key)
                self._items.append(
                    AcceptedMemoryItem(
                        item_id=_make_item_id("fact", term, sense, item_type, value),
                        item_kind="fact",
                        term=term,
                        sense=sense,
                        item_type=item_type,
                        value=value,
                        source_id=source_id,
                        risk_level=item.get("risk_level", "low"),
                        decision="accepted_auto",
                        provenance=ItemProvenance(
                            source_artifact=_FACT_SOURCE_PATH,
                            source_proposal_id=proposal_id,
                            source_pattern_id=None,
                        ),
                    )
                )
                self._fact_count += 1

    def _load_patterns(self, data: dict[str, Any], seen: set[str]) -> None:
        for pc in data.get("pattern_candidates", []):
            dec = pc.get("decision", "")
            if dec == "needs_review":
                self._excluded_needs_review += 1
                continue
            if dec == "rejected_auto":
                self._excluded_rejected += 1
                continue
            if dec != "accepted_auto":
                continue
            if pc.get("risk_level", "low") == "high":
                self._excluded_high_risk += 1
                continue
            term = pc.get("term", "")
            sense = pc.get("sense", "")
            pattern_type = pc.get("pattern_type", "")
            item_type = _pattern_item_type(pattern_type)
            value = pc.get("template", "")
            pattern_id = pc.get("pattern_id", "")
            source_id = pc.get("source_id", "")
            key = _dedup_key("pattern", term, sense, item_type, value)
            if key in seen:
                self._dedup_count += 1
                continue
            seen.add(key)
            self._items.append(
                AcceptedMemoryItem(
                    item_id=_make_item_id("pattern", term, sense, item_type, value),
                    item_kind="pattern",
                    term=term,
                    sense=sense,
                    item_type=item_type,
                    value=value,
                    source_id=source_id,
                    risk_level=pc.get("risk_level", "low"),
                    decision="accepted_auto",
                    provenance=ItemProvenance(
                        source_artifact=_PATTERN_SOURCE_PATH,
                        source_proposal_id=None,
                        source_pattern_id=pattern_id,
                    ),
                )
            )
            self._pattern_count += 1
