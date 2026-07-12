"""AcceptedKnowledgeMemoryProvider v1.

Read-only provider for the accepted knowledge memory artifact.
Loads accepted_knowledge_memory_v1.json and exposes deterministic,
indexed lookups. Builds an overlay ExplicitSenseMemory without
mutating any live memory object.

Rule-based and deterministic only.  No ML libraries.
No learned model parameters.  No language-model inference.  No back-prop.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- No live/default engine or memory path is changed.
- Thresholds are NOT lowered.
- Validators are NOT weakened.
- No generic fallback is added.
- Source memory objects are NOT mutated; a copied/combined instance is returned.
- Provider is optional: nothing imports or uses it by default.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Optional

from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.knowledge.accepted_memory_types import AcceptedMemoryItem, ItemProvenance

MEMORY_VERSION = "accepted_knowledge_memory_v1"


class AcceptedKnowledgeMemoryProvider:
    """Read-only provider backed by the accepted knowledge memory artifact.

    Usage::

        provider = AcceptedKnowledgeMemoryProvider("worldpgt/experiments/accepted_knowledge_memory_v1.json")
        overlay_mem = provider.build_overlay_memory()
        engine = ControlledContinuationEngine(memory=overlay_mem)

    The provider is deterministic: the same artifact always produces the same
    overlay memory and the same benchmark output.
    """

    def __init__(self, artifact_path: str) -> None:
        self._artifact_path = artifact_path
        self._items: list[AcceptedMemoryItem] = []
        self._by_term: dict[str, list[AcceptedMemoryItem]] = defaultdict(list)
        self._by_sense: dict[str, list[AcceptedMemoryItem]] = defaultdict(list)
        self._by_item_type: dict[str, list[AcceptedMemoryItem]] = defaultdict(list)
        self._by_value: dict[str, list[AcceptedMemoryItem]] = defaultdict(list)
        self._fact_count: int = 0
        self._pattern_count: int = 0
        self._terms: list[str] = []
        self._senses: list[str] = []
        self._load(artifact_path)

    # ------------------------------------------------------------------
    # Loading (private)
    # ------------------------------------------------------------------

    def _load(self, path: str) -> None:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)

        for raw in data.get("items", []):
            prov_raw = raw.get("provenance", {})
            prov = ItemProvenance(
                source_artifact=prov_raw.get("source_artifact", ""),
                source_proposal_id=prov_raw.get("source_proposal_id"),
                source_pattern_id=prov_raw.get("source_pattern_id"),
            )
            item = AcceptedMemoryItem(
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
            self._items.append(item)
            self._by_term[item.term].append(item)
            sense_key = f"{item.term}:{item.sense}"
            self._by_sense[sense_key].append(item)
            self._by_item_type[item.item_type].append(item)
            self._by_value[item.value.strip().lower()].append(item)

        artifact_stats = data.get("stats", {})
        self._fact_count = artifact_stats.get("fact_items", 0)
        self._pattern_count = artifact_stats.get("pattern_items", 0)
        self._terms = sorted(self._by_term.keys())
        self._senses = sorted(self._by_sense.keys())

    # ------------------------------------------------------------------
    # Read-only lookups
    # ------------------------------------------------------------------

    @property
    def items(self) -> list[AcceptedMemoryItem]:
        return list(self._items)

    @property
    def fact_count(self) -> int:
        return self._fact_count

    @property
    def pattern_count(self) -> int:
        return self._pattern_count

    @property
    def terms(self) -> list[str]:
        return list(self._terms)

    @property
    def senses(self) -> list[str]:
        return list(self._senses)

    def items_for_term(self, term: str) -> list[AcceptedMemoryItem]:
        return list(self._by_term.get(term, []))

    def items_for_sense(self, term: str, sense: str) -> list[AcceptedMemoryItem]:
        return list(self._by_sense.get(f"{term}:{sense}", []))

    def items_for_item_type(self, item_type: str) -> list[AcceptedMemoryItem]:
        return list(self._by_item_type.get(item_type, []))

    def items_for_value(self, value: str) -> list[AcceptedMemoryItem]:
        return list(self._by_value.get(value.strip().lower(), []))

    @property
    def stats(self) -> dict:
        return {
            "items_loaded": len(self._items),
            "fact_items": self._fact_count,
            "pattern_items": self._pattern_count,
            "terms": len(self._terms),
            "senses": len(self._senses),
            "by_item_type": {k: len(v) for k, v in sorted(self._by_item_type.items())},
        }

    # ------------------------------------------------------------------
    # Overlay memory builder
    # ------------------------------------------------------------------

    def build_overlay_memory(self) -> ExplicitSenseMemory:
        """Return a fresh ExplicitSenseMemory with accepted positive_cue items applied.

        Creates a new instance (include_builtin=True), then appends accepted
        positive cues that are not already present in the builtin set.

        The returned instance is completely independent of any live engine
        memory.  The original builtin ExplicitSenseMemory is never mutated.

        Only positive_cue items are applied to the scoring layer; all other
        item types are available for metadata lookups via the provider but
        are not injected into the returned memory.
        """
        overlay = ExplicitSenseMemory(include_builtin=True)

        builtin_cues: dict[tuple[str, str], set[str]] = {}
        for term in overlay.known_terms():
            for entry in overlay.get_senses(term):
                builtin_cues[(term, entry.sense_id)] = set(entry.cues)

        cues_by_ts: dict[tuple[str, str], list[AcceptedMemoryItem]] = defaultdict(list)
        for item in self._items:
            if item.item_type == "positive_cue":
                cues_by_ts[(item.term, item.sense)].append(item)

        for (term, sense_id), sense_items in cues_by_ts.items():
            existing = builtin_cues.get((term, sense_id), set())
            for entry in overlay.get_senses(term):
                if entry.sense_id == sense_id:
                    for item in sense_items:
                        v = item.value.strip().lower()
                        if v not in existing:
                            entry.cues.append(v)
                            existing.add(v)
        return overlay

    # ------------------------------------------------------------------
    # Trace helpers
    # ------------------------------------------------------------------

    def provider_trace_markers(self) -> list[str]:
        """Top-level trace markers indicating the provider is active."""
        return [
            "accepted_memory_provider=enabled",
            f"accepted_memory_version={MEMORY_VERSION}",
            f"accepted_memory_items_loaded={len(self._items)}",
        ]

    def item_trace_markers(self, item: AcceptedMemoryItem) -> list[str]:
        """Per-item trace markers for a specific item that influenced output."""
        return [
            "accepted_memory_provider=enabled",
            f"accepted_memory_version={MEMORY_VERSION}",
            f"accepted_memory_item_id={item.item_id}",
            f"accepted_memory_item_type={item.item_type}",
            f"accepted_memory_value={item.value}",
        ]

    def cue_item_for(
        self, term: str, sense_id: str, cue_value: str
    ) -> Optional[AcceptedMemoryItem]:
        """Return the AcceptedMemoryItem for a positive_cue, or None if not from provider."""
        for item in self._by_sense.get(f"{term}:{sense_id}", []):
            if item.item_type == "positive_cue" and item.value.strip().lower() == cue_value.strip().lower():
                return item
        return None

    def new_cues_for(
        self,
        baseline_mem: ExplicitSenseMemory,
        term: str,
        sense_id: str,
    ) -> set[str]:
        """Cues in the provider overlay that are NOT already in baseline_mem for this (term, sense)."""
        baseline_set: set[str] = set()
        for entry in baseline_mem.get_senses(term):
            if entry.sense_id == sense_id:
                baseline_set = set(entry.cues)
                break
        provider_cues = {
            item.value.strip().lower()
            for item in self._by_sense.get(f"{term}:{sense_id}", [])
            if item.item_type == "positive_cue"
        }
        return provider_cues - baseline_set
