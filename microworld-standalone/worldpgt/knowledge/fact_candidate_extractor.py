"""Deterministic fact candidate extractor.

Uses fixed vocabulary / regex / phrase matching only. No ML.
Extracts typed candidate facts from curated snippets.
"""

from __future__ import annotations

import re
from typing import NamedTuple

from worldpgt.knowledge.types import CuratedSnippet, FactCandidate, FactType

# ---------------------------------------------------------------------------
# Per-sense extraction tables
# Each entry: (fact_type, value, is_multiword)
# ---------------------------------------------------------------------------

_SENSE_FACTS: dict[tuple[str, str], list[tuple[FactType, str, bool]]] = {
    ("bank", "financial_institution"): [
        ("positive_cue", "account", False),
        ("positive_cue", "deposit", False),
        ("positive_cue", "loan", False),
        ("positive_cue", "teller", False),
        ("positive_cue", "cash", False),
        ("positive_cue", "savings", False),
        ("positive_cue", "ATM", False),
        ("positive_cue", "interest", False),
        ("positive_cue", "check", False),
        ("positive_cue", "branch", False),
        ("typical_action", "deposit", False),
        ("typical_action", "withdraw", False),
        ("typical_action", "open an account", True),
        ("typical_action", "request a loan", True),
        ("typical_location", "branch", False),
        ("semantic_frame_hint", "financial_transaction", False),
        ("phrase_candidate_hint", "went to the bank to deposit", True),
    ],
    ("bank", "river_edge"): [
        ("positive_cue", "stream", False),
        ("positive_cue", "current", False),
        ("positive_cue", "reeds", False),
        ("positive_cue", "grass", False),
        ("positive_cue", "fishermen", False),
        ("positive_cue", "bridge", False),
        ("positive_cue", "floodwater", False),
        ("positive_cue", "soil", False),
        ("positive_cue", "river bank", True),
        ("typical_action", "fish", False),
        ("typical_action", "cast", False),
        ("typical_action", "erode", False),
        ("typical_location", "river", False),
        ("semantic_frame_hint", "natural_geography", False),
        ("phrase_candidate_hint", "stood on the bank", True),
    ],
    ("bat", "animal"): [
        ("positive_cue", "cave", False),
        ("positive_cue", "roost", False),
        ("positive_cue", "rafters", False),
        ("positive_cue", "insects", False),
        ("positive_cue", "echolocation", False),
        ("positive_cue", "wings", False),
        ("positive_cue", "membrane", False),
        ("positive_cue", "dusk", False),
        ("positive_cue", "mammal", False),
        ("typical_action", "hunt", False),
        ("typical_action", "roost", False),
        ("typical_action", "hang", False),
        ("typical_location", "cave", False),
        ("typical_location", "attic", False),
        ("semantic_frame_hint", "animal_behavior", False),
        ("phrase_candidate_hint", "flew out of the cave at dusk", True),
    ],
    ("bat", "sports_equipment"): [
        ("positive_cue", "pitcher", False),
        ("positive_cue", "ball", False),
        ("positive_cue", "batter", False),
        ("positive_cue", "plate", False),
        ("positive_cue", "swing", False),
        ("positive_cue", "player", False),
        ("positive_cue", "pitch", False),
        ("positive_cue", "aluminum", False),
        ("positive_cue", "baseball", False),
        ("positive_cue", "home run", True),
        ("typical_action", "swing", False),
        ("typical_action", "hit", False),
        ("typical_action", "strike", False),
        ("typical_location", "plate", False),
        ("semantic_frame_hint", "sports_equipment_use", False),
        ("phrase_candidate_hint", "swung the bat at the pitch", True),
    ],
    ("seal", "animal"): [
        ("positive_cue", "fish", False),
        ("positive_cue", "flippers", False),
        ("positive_cue", "ocean", False),
        ("positive_cue", "coast", False),
        ("positive_cue", "marine", False),
        ("positive_cue", "mammal", False),
        ("positive_cue", "pup", False),
        ("positive_cue", "orca", False),
        ("positive_cue", "bark", False),
        ("typical_action", "swim", False),
        ("typical_action", "dive", False),
        ("typical_action", "rest", False),
        ("typical_location", "coast", False),
        ("typical_location", "shore", False),
        ("semantic_frame_hint", "animal_behavior", False),
        ("phrase_candidate_hint", "dove below the surface", True),
    ],
    ("seal", "closure_stamp"): [
        ("positive_cue", "wax", False),
        ("positive_cue", "envelope", False),
        ("positive_cue", "document", False),
        ("positive_cue", "parcel", False),
        ("positive_cue", "flap", False),
        ("positive_cue", "certificate", False),
        ("positive_cue", "notary", False),
        ("positive_cue", "airtight", False),
        ("positive_cue", "stamped", False),
        ("typical_action", "press", False),
        ("typical_action", "break", False),
        ("typical_action", "stamp", False),
        ("typical_location", "document", False),
        ("semantic_frame_hint", "document_closure", False),
        ("phrase_candidate_hint", "sealed it shut", True),
        ("phrase_candidate_hint", "pressed the seal onto the document", True),
    ],
    ("crane", "bird"): [
        ("positive_cue", "neck", False),
        ("positive_cue", "legs", False),
        ("positive_cue", "reeds", False),
        ("positive_cue", "lake", False),
        ("positive_cue", "wetlands", False),
        ("positive_cue", "marsh", False),
        ("positive_cue", "wings", False),
        ("positive_cue", "migration", False),
        ("positive_cue", "flock", False),
        ("typical_action", "wade", False),
        ("typical_action", "dance", False),
        ("typical_action", "migrate", False),
        ("typical_location", "wetlands", False),
        ("typical_location", "lake", False),
        ("semantic_frame_hint", "bird_behavior", False),
        ("phrase_candidate_hint", "stood in the reeds by the lake", True),
    ],
    ("crane", "machine"): [
        ("positive_cue", "hook", False),
        ("positive_cue", "load", False),
        ("positive_cue", "cable", False),
        ("positive_cue", "boom", False),
        ("positive_cue", "operator", False),
        ("positive_cue", "tower crane", True),
        ("positive_cue", "outriggers", False),
        ("positive_cue", "construction site", True),
        ("positive_cue", "steel beams", True),
        ("typical_action", "lift", False),
        ("typical_action", "lower", False),
        ("typical_action", "swing", False),
        ("typical_location", "construction site", True),
        ("semantic_frame_hint", "heavy_machinery_operation", False),
        ("phrase_candidate_hint", "lowered the load with the hook", True),
    ],
    ("rock", "stone"): [
        ("positive_cue", "cliff", False),
        ("positive_cue", "boulder", False),
        ("positive_cue", "trail", False),
        ("positive_cue", "mineral", False),
        ("positive_cue", "geology", False),
        ("positive_cue", "ground", False),
        ("positive_cue", "hillside", False),
        ("typical_action", "climb", False),
        ("typical_action", "roll", False),
        ("typical_location", "mountain", False),
        ("typical_location", "cliff", False),
        ("semantic_frame_hint", "natural_material", False),
        ("phrase_candidate_hint", "climbed over the rocks on the trail", True),
    ],
    ("rock", "music"): [
        ("positive_cue", "band", False),
        ("positive_cue", "concert", False),
        ("positive_cue", "crowd", False),
        ("positive_cue", "stage", False),
        ("positive_cue", "guitar", False),
        ("positive_cue", "drums", False),
        ("positive_cue", "album", False),
        ("positive_cue", "genre", False),
        ("typical_action", "play", False),
        ("typical_action", "perform", False),
        ("typical_location", "stage", False),
        ("semantic_frame_hint", "music_performance", False),
        ("phrase_candidate_hint", "played to a crowd on stage", True),
    ],
    ("spring", "season"): [
        ("positive_cue", "thaw", False),
        ("positive_cue", "flowers", False),
        ("positive_cue", "mornings", False),
        ("positive_cue", "rain", False),
        ("positive_cue", "migration", False),
        ("positive_cue", "bloom", False),
        ("positive_cue", "warm", False),
        ("typical_action", "bloom", False),
        ("typical_action", "thaw", False),
        ("typical_location", "outdoors", False),
        ("semantic_frame_hint", "seasonal_transition", False),
        ("phrase_candidate_hint", "arrived with the first warm mornings", True),
    ],
    ("spring", "coil"): [
        ("positive_cue", "latch", False),
        ("positive_cue", "handle", False),
        ("positive_cue", "device", False),
        ("positive_cue", "coil", False),
        ("positive_cue", "metal", False),
        ("positive_cue", "elastic", False),
        ("positive_cue", "mattress", False),
        ("positive_cue", "compressed", False),
        ("typical_action", "snap", False),
        ("typical_action", "compress", False),
        ("typical_action", "stretch", False),
        ("typical_location", "device", False),
        ("semantic_frame_hint", "mechanical_device", False),
        ("phrase_candidate_hint", "snapped back into place", True),
    ],
}

# Phrases to scan for in text to extract phrase_candidate_hints dynamically
_PHRASE_PATTERNS: list[tuple[str, str, str]] = [
    # (term, sense, pattern_substring)
    ("bank", "financial_institution", "open an account"),
    ("bank", "river_edge", "stood on the bank"),
    ("bat", "animal", "at dusk"),
    ("bat", "sports_equipment", "swung the bat"),
    ("seal", "animal", "dove below"),
    ("seal", "closure_stamp", "sealed it shut"),
    ("crane", "bird", "in the reeds"),
    ("crane", "machine", "construction site"),
    ("rock", "stone", "on the trail"),
    ("rock", "music", "on stage"),
    ("spring", "season", "warm mornings"),
    ("spring", "coil", "snapped back"),
]


class FactCandidateExtractor:
    """Extracts typed fact candidates from curated snippets deterministically."""

    def extract(self, snippets: list[CuratedSnippet]) -> list[FactCandidate]:
        candidates: list[FactCandidate] = []
        for snippet in snippets:
            candidates.extend(self._extract_one(snippet))
        return candidates

    def _extract_one(self, snippet: CuratedSnippet) -> list[FactCandidate]:
        key = (snippet.term, snippet.sense)
        table = _SENSE_FACTS.get(key, [])
        results: list[FactCandidate] = []
        seen: set[tuple[FactType, str]] = set()
        for fact_type, value, is_multiword in table:
            dedup_key = (fact_type, value.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            # Verify the value (or a close variant) appears in the snippet text
            confidence = self._confidence(value, snippet.text, snippet.trust_seed)
            results.append(
                FactCandidate(
                    source_id=snippet.source_id,
                    term=snippet.term,
                    sense=snippet.sense,
                    fact_type=fact_type,
                    value=value,
                    confidence=confidence,
                    is_broad=False,  # normalizer decides
                    is_multiword=is_multiword,
                )
            )
        return results

    @staticmethod
    def _confidence(value: str, text: str, trust_seed: float) -> float:
        text_lower = text.lower()
        val_lower = value.lower()
        if val_lower in text_lower:
            return round(trust_seed, 3)
        # Partial word match (e.g. "swim" matches "swimming")
        core = val_lower.split()[0] if " " in val_lower else val_lower
        if re.search(r"\b" + re.escape(core), text_lower):
            return round(trust_seed * 0.85, 3)
        return round(trust_seed * 0.6, 3)
