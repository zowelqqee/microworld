"""Normalizes extracted fact candidates and applies conflict analysis."""

from __future__ import annotations

import re
from collections import defaultdict

from worldpgt.knowledge.types import FactCandidate, NormalizedFact

# Broad single-word cues that are rejected unless part of a narrow multiword phrase
_BROAD_DENYLIST: frozenset[str] = frozenset({
    "water", "light", "thing", "object", "place", "time",
    "people", "near", "under", "over", "good", "bad",
    "big", "small", "use", "used", "make", "made",
    "area", "surface", "side", "part", "end", "top",
    "move", "go", "get", "see", "come", "take",
    "long", "large", "high", "low", "new", "old",
})

# Stopwords stripped when normalizing values
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "by", "to",
    "and", "or", "is", "are", "was", "were", "be", "been",
    "it", "its", "this", "that", "their", "they",
    "may", "can", "will", "would", "has", "have", "had",
})

# Plural→singular simple suffix rules (applied only to single words)
_PLURAL_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"ches$"), "ch"),
    (re.compile(r"shes$"), "sh"),
    (re.compile(r"sses$"), "ss"),
    (re.compile(r"xes$"), "x"),
    (re.compile(r"zes$"), "z"),
    (re.compile(r"ies$"), "y"),
    (re.compile(r"ves$"), "f"),
    # Only strip trailing s when it doesn't create a double-s ending or garble the root
    (re.compile(r"(?<!s)s$"), ""),
]

# Multiword phrases that are narrow exceptions to broad single-word rejection
_NARROW_MULTIWORD_EXCEPTIONS: frozenset[str] = frozenset({
    "ocean water",
    "construction site",
    "wax seal",
    "river bank",
    "steel beams",
    "tower crane",
    "home run",
})


def _normalize_value(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^\w\s]", "", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _singularize(word: str) -> str:
    for pattern, replacement in _PLURAL_RULES:
        candidate = pattern.sub(replacement, word)
        if candidate != word and len(candidate) > 2:
            return candidate
    return word


def _is_broad(value: str, is_multiword: bool) -> bool:
    if is_multiword:
        # Multiword narrow exceptions are never broad
        if value in _NARROW_MULTIWORD_EXCEPTIONS:
            return False
        # Multiword phrases that contain only stopwords + broad words stay broad
        words = value.split()
        non_stop = [w for w in words if w not in _STOPWORDS]
        if all(w in _BROAD_DENYLIST for w in non_stop):
            return True
        return False
    return value in _BROAD_DENYLIST


class FactNormalizer:
    """Normalizes candidates and assigns risk based on conflict analysis."""

    def normalize(self, candidates: list[FactCandidate]) -> list[NormalizedFact]:
        # First pass: normalize values
        normalized: list[NormalizedFact] = []
        for c in candidates:
            norm_val = _normalize_value(c.value)
            # Singularize single-word non-multiword values
            if not c.is_multiword and " " not in norm_val:
                norm_val = _singularize(norm_val)
            broad = _is_broad(norm_val, c.is_multiword)
            rejected = broad and c.fact_type in ("positive_cue", "anti_cue")
            rejection_reason = "broad_cue_denylist" if rejected else ""
            normalized.append(
                NormalizedFact(
                    source_id=c.source_id,
                    term=c.term,
                    sense=c.sense,
                    fact_type=c.fact_type,
                    value=norm_val,
                    confidence=c.confidence,
                    is_broad=broad,
                    is_multiword=c.is_multiword,
                    rejected=rejected,
                    rejection_reason=rejection_reason,
                )
            )

        # Second pass: conflict detection — cue appears under multiple senses
        # Group by (term, fact_type, value) → set of senses
        cue_senses: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        for nf in normalized:
            if nf.fact_type in ("positive_cue", "anti_cue") and not nf.rejected:
                cue_senses[(nf.term, nf.fact_type, nf.value)].add(nf.sense)

        # Third pass: annotate conflicts and assign risk_level
        result: list[NormalizedFact] = []
        for nf in normalized:
            conflict_key = (nf.term, nf.fact_type, nf.value)
            conflict_s = sorted(cue_senses.get(conflict_key, set()) - {nf.sense})
            nf.conflict_senses = conflict_s
            nf.risk_level = self._risk(nf)
            result.append(nf)

        return result

    @staticmethod
    def _risk(nf: NormalizedFact) -> str:
        if nf.rejected:
            return "high"
        if nf.conflict_senses:
            return "high"
        if nf.is_broad:
            return "high"
        if nf.confidence < 0.5:
            return "medium"
        if nf.is_multiword and nf.fact_type == "phrase_candidate_hint":
            return "low"
        if nf.fact_type in ("semantic_frame_hint",):
            return "low"
        return "low"
