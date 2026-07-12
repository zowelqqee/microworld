"""Knowledge overlay for dry-run benchmark simulation.

Loads accepted_auto items from the auto-review artifact and builds a temporary
overlay ExplicitSenseMemory for simulation only.

SAFETY CONTRACT:
- Does NOT write to sense_memory.py.
- Does NOT mutate any live ExplicitSenseMemory instance.
- Does NOT lower thresholds, bypass validators, or force continuations.
- Does NOT modify benchmark outputs.
- The overlay_memory returned is an independent instance created only for
  the dry-run and is discarded after use.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from worldpgt.continuation.sense_memory import ExplicitSenseMemory


class KnowledgeOverlay:
    """Reads accepted_auto items and builds a fresh overlay sense memory.

    Only positive_cues (and their normalized values) are injected into the
    scoring layer. Semantic frame hints, phrase hints, and actions are tracked
    for reporting but not auto-applied to templates in v1.
    """

    def __init__(self, auto_review_data: dict[str, Any]) -> None:
        # per-(term, sense_id) lists of new positive cues to inject
        self._positive_cues: dict[tuple[str, str], list[str]] = defaultdict(list)
        # per-(term, sense_id) lists of accepted semantic frame hints (metadata only)
        self._frame_hints: dict[tuple[str, str], list[str]] = defaultdict(list)
        # per-(term, sense_id) lists of accepted phrase hints (metadata only)
        self._phrase_hints: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._stats: dict[str, int] = {}
        self._load(auto_review_data)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self, data: dict[str, Any]) -> None:
        n_items = n_pos_cues = n_frames = n_phrases = 0
        for pr in data.get("proposal_reviews", []):
            term = pr["term"]
            sense = pr["sense"]
            for item in pr.get("items", []):
                if item.get("decision") != "accepted_auto":
                    continue
                n_items += 1
                itype = item.get("item_type", "")
                value = item.get("value", "")
                if itype == "positive_cue":
                    self._positive_cues[(term, sense)].append(value)
                    n_pos_cues += 1
                elif itype == "semantic_frame_hint":
                    self._frame_hints[(term, sense)].append(value)
                    n_frames += 1
                elif itype == "phrase_candidate_hint":
                    self._phrase_hints[(term, sense)].append(value)
                    n_phrases += 1
        self._stats = {
            "accepted_auto_items_loaded": n_items,
            "positive_cues_loaded": n_pos_cues,
            "semantic_frame_hints_loaded": n_frames,
            "phrase_candidate_hints_loaded": n_phrases,
        }

    # ------------------------------------------------------------------
    # Overlay memory builder
    # ------------------------------------------------------------------

    def build_overlay_memory(self) -> ExplicitSenseMemory:
        """Return a fresh ExplicitSenseMemory with overlay cues added.

        The returned instance is completely independent of any live engine
        instance; it is safe to use in a separate ControlledContinuationEngine
        for dry-run purposes and discarded afterwards.
        """
        overlay_mem = ExplicitSenseMemory(include_builtin=True)

        # Record which cues are already in the fresh builtin memory so we
        # only add genuinely new cues (avoids double-counting).
        builtin_cues: dict[tuple[str, str], set[str]] = {}
        for term in overlay_mem.known_terms():
            for entry in overlay_mem.get_senses(term):
                builtin_cues[(term, entry.sense_id)] = set(entry.cues)

        for (term, sense_id), new_cues in self._positive_cues.items():
            existing = builtin_cues.get((term, sense_id), set())
            # get_senses returns a list of the same SenseEntry objects held
            # inside overlay_mem._senses — mutating .cues here is safe because
            # overlay_mem is a fresh instance created above.
            for entry in overlay_mem.get_senses(term):
                if entry.sense_id == sense_id:
                    for cue in new_cues:
                        cue_lower = cue.lower()
                        if cue_lower not in existing:
                            entry.cues.append(cue_lower)
                            existing.add(cue_lower)
        return overlay_mem

    # ------------------------------------------------------------------
    # Accessors and trace helpers
    # ------------------------------------------------------------------

    @property
    def stats(self) -> dict[str, int]:
        return dict(self._stats)

    def new_cues_for(self, term: str, sense_id: str) -> list[str]:
        """Return overlay-only cues for the given term/sense (not in builtin)."""
        return list(self._positive_cues.get((term, sense_id), []))

    def overlay_trace_markers(self, n_new_cues_in_prompt: int) -> list[str]:
        """Standard trace marker strings to embed in output rows."""
        return [
            "knowledge_overlay=enabled",
            "overlay_source=knowledge_ingestion_v1_auto_review",
            f"overlay_item_count={self._stats.get('accepted_auto_items_loaded', 0)}",
            f"overlay_new_cues_matched_in_prompt={n_new_cues_in_prompt}",
        ]

    def cue_trace(self, term: str, sense_id: str, matched: list[str]) -> list[str]:
        """Per-cue trace markers for overlay-contributed scoring hits."""
        return [f"overlay_cue={cue} -> {term}:{sense_id}" for cue in matched]

    def all_terms(self) -> list[str]:
        keys = set(self._positive_cues) | set(self._frame_hints) | set(self._phrase_hints)
        return sorted({term for term, _ in keys})
