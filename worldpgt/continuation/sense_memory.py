"""Explicit, inspectable sense memory with deterministic lexical scoring.

No ML libraries. Scores are cue_overlap_count * trust over lowercase token/substring matches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from worldpgt.continuation.types import SenseEntry

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_NEGATION_WORDS = {"not", "no", "never", "without", "neither", "nor"}
_NEGATION_WINDOW = 3


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class SenseEvidence:
    """Detailed deterministic cue evidence for one prompt and ambiguous term."""

    positive_scores: dict[str, float]
    adjusted_scores: dict[str, float]
    negated_cues: dict[str, list[str]]
    anti_cues: dict[str, list[str]]
    guard_failures: dict[str, list[str]]
    positive_cues: dict[str, list[str]]
    conflict_detected: bool
    evidence_notes: list[str] = field(default_factory=list)


def _cue_positions(tokens: list[str], cue: str) -> list[int]:
    cue_tokens = _tokenize(cue)
    if not cue_tokens:
        return []
    if len(cue_tokens) == 1:
        needle = cue_tokens[0]
        return [idx for idx, token in enumerate(tokens) if token == needle]

    positions = []
    width = len(cue_tokens)
    for idx in range(0, len(tokens) - width + 1):
        if tokens[idx : idx + width] == cue_tokens:
            positions.append(idx)
    return positions


def _has_negation_before(tokens: list[str], cue_position: int) -> bool:
    start = max(0, cue_position - _NEGATION_WINDOW)
    return any(token in _NEGATION_WORDS for token in tokens[start:cue_position])


class ExplicitSenseMemory:
    """Stores SenseEntry objects per term and scores them against prompts deterministically."""

    def __init__(self, include_builtin: bool = True) -> None:
        self._senses: dict[str, list[SenseEntry]] = {}
        self._anti_cues: dict[tuple[str, str], list[str]] = {}
        if include_builtin:
            self._load_builtin_senses()

    def add_sense(
        self,
        term: str,
        sense_id: str,
        cues: list[str],
        continuations: list[str],
        trust: float = 1.0,
    ) -> SenseEntry:
        entry = SenseEntry(
            term=term.lower(),
            sense_id=sense_id,
            cues=[c.lower() for c in cues],
            continuations=list(continuations),
            trust=trust,
        )
        self._senses.setdefault(entry.term, []).append(entry)
        return entry

    def add_anti_cues(self, term: str, sense_id: str, phrases: list[str]) -> None:
        key = (term.lower(), sense_id)
        self._anti_cues.setdefault(key, []).extend(phrase.lower() for phrase in phrases)

    def get_senses(self, term: str) -> list[SenseEntry]:
        return list(self._senses.get(term.lower(), []))

    def known_terms(self) -> list[str]:
        return list(self._senses.keys())

    def find_ambiguous_terms(self, prompt: str) -> list[str]:
        """Known terms present in the prompt, ordered by first appearance."""
        tokens = _tokenize(prompt)
        found: list[tuple[int, str]] = []
        for term in self.known_terms():
            if term in tokens:
                found.append((tokens.index(term), term))
        found.sort(key=lambda pair: pair[0])
        return [term for _, term in found]

    def matched_cues(self, prompt: str, term: str) -> dict[str, list[str]]:
        """Per sense_id, non-negated cues of that sense found in the prompt."""
        return self.score_senses_with_evidence(prompt, term).positive_cues

    def score_senses(self, prompt: str, term: str) -> dict[str, float]:
        """score = cue_overlap_count * trust; 0.0 when no cues match."""
        return self.score_senses_with_evidence(prompt, term).adjusted_scores

    def score_senses_with_evidence(self, prompt: str, term: str) -> SenseEvidence:
        """Score senses while separating positive and negated cue evidence."""
        tokens = _tokenize(prompt)
        positive_cues: dict[str, list[str]] = {}
        negated_cues: dict[str, list[str]] = {}
        anti_cues: dict[str, list[str]] = {}
        guard_failures: dict[str, list[str]] = {}
        positive_scores: dict[str, float] = {}
        adjusted_scores: dict[str, float] = {}
        evidence_notes: list[str] = []

        for entry in self.get_senses(term):
            positives = []
            negated = []
            for cue in entry.cues:
                positions = _cue_positions(tokens, cue)
                if not positions:
                    continue
                if any(not _has_negation_before(tokens, position) for position in positions):
                    positives.append(cue)
                else:
                    negated.append(cue)

            positive_cues[entry.sense_id] = positives
            negated_cues[entry.sense_id] = negated
            positive_scores[entry.sense_id] = float(len(positives)) * entry.trust
            matched_anti_cues = self._matched_anti_cues(tokens, entry.term, entry.sense_id)
            anti_cues[entry.sense_id] = matched_anti_cues
            failures = self._guard_failures(tokens, entry.term, entry.sense_id, positives)
            guard_failures[entry.sense_id] = failures
            if matched_anti_cues or failures:
                adjusted_scores[entry.sense_id] = 0.0
            else:
                adjusted_scores[entry.sense_id] = positive_scores[entry.sense_id]
            for cue in positives:
                evidence_notes.append(f"positive_cue={cue} -> {entry.sense_id}")
            for cue in negated:
                evidence_notes.append(f"negated_cue={cue} -> {entry.sense_id}")
            for cue in matched_anti_cues:
                evidence_notes.append(f"anti_cue={cue} -> {entry.sense_id}")
            for failure in failures:
                evidence_notes.append(f"guard_failure={failure} -> {entry.sense_id}")

        senses_with_positive_evidence = [
            sense_id for sense_id, score in adjusted_scores.items() if score > 0.0
        ]
        conflict_detected = len(senses_with_positive_evidence) > 1
        if conflict_detected:
            evidence_notes.append("conflict_detected")

        return SenseEvidence(
            positive_scores=positive_scores,
            adjusted_scores=adjusted_scores,
            negated_cues=negated_cues,
            anti_cues=anti_cues,
            guard_failures=guard_failures,
            positive_cues=positive_cues,
            conflict_detected=conflict_detected,
            evidence_notes=evidence_notes,
        )

    def _matched_anti_cues(self, tokens: list[str], term: str, sense_id: str) -> list[str]:
        matched = []
        for phrase in self._anti_cues.get((term, sense_id), []):
            if _cue_positions(tokens, phrase):
                matched.append(phrase)
        return matched

    def _guard_failures(
        self,
        tokens: list[str],
        term: str,
        sense_id: str,
        positive_cues: list[str],
    ) -> list[str]:
        if term == "bat" and sense_id == "sports_equipment":
            strong_cues = {"baseball", "swing", "swung", "hit", "cracked", "game"}
            if "player" in positive_cues and not any(cue in strong_cues for cue in positive_cues + tokens):
                return ["player_alone_insufficient"]
        if term == "bank" and sense_id == "financial_institution":
            weak_cues = {"cash", "card"}
            strong_cues = {
                "money",
                "loan",
                "account",
                "teller",
                "deposit",
                "mortgage",
                "credit",
                "customer",
            }
            if positive_cues and all(cue in weak_cues for cue in positive_cues):
                if not any(cue in strong_cues for cue in positive_cues):
                    return ["cash_or_card_alone_insufficient"]
        return []

    def _load_builtin_senses(self) -> None:
        self.add_sense(
            "bank",
            "financial_institution",
            ["money", "loan", "account", "teller", "deposit", "cash", "card", "mortgage", "credit", "customer"],
            ["open an account", "deposit money", "ask for a loan"],
        )
        self.add_sense(
            "bank",
            "river_edge",
            ["river", "fisherman", "water", "shore", "mud", "stream", "current", "boat"],
            ["cast his line", "watched the current", "sat near the water"],
        )
        self.add_sense(
            "bat",
            "animal",
            ["cave", "flew", "flying", "wings", "night", "animal", "hanging"],
            ["flew into the dark cave", "hung from the ceiling", "searched for insects"],
        )
        self.add_sense(
            "bat",
            "sports_equipment",
            ["baseball", "player", "hit", "cracked", "swing", "game"],
            ["hit the ball", "cracked after the swing", "was dropped near home plate"],
        )
        self.add_sense(
            "seal",
            "animal",
            ["ocean", "fish", "zoo", "flippers", "animal", "water"],
            ["swam through the cold water", "balanced on the rock", "looked for fish"],
        )
        self.add_sense(
            "seal",
            "closure_stamp",
            ["envelope", "document", "stamp", "wax", "official", "package"],
            ["closed the envelope", "marked the document", "protected the package"],
        )
        self.add_sense(
            "crane",
            "bird",
            ["bird", "wings", "marsh", "flew", "nest", "lake"],
            ["flew over the marsh", "stood near the lake", "spread its wings"],
        )
        self.add_sense(
            "crane",
            "machine",
            ["construction", "building", "lifted", "steel", "operator", "site"],
            ["lifted the steel beam", "moved above the construction site", "carried the load"],
        )
        self.add_sense(
            "spring",
            "season",
            ["flowers", "april", "warm", "weather", "garden", "rain"],
            ["brought warmer days", "filled the garden with flowers", "came after winter"],
        )
        self.add_sense(
            "spring",
            "coil",
            ["metal", "compressed", "mechanism", "tension", "bounce", "device"],
            ["compressed under pressure", "snapped back into place", "stored mechanical energy"],
        )
        self.add_sense(
            "rock",
            "stone",
            ["mountain", "stone", "river", "heavy", "ground", "cliff"],
            ["rolled down the hill", "lay near the river", "broke into smaller stones"],
        )
        self.add_sense(
            "rock",
            "music",
            ["band", "guitar", "concert", "song", "drummer", "stage"],
            ["filled the stadium", "played through the speakers", "started with a loud guitar riff"],
        )
        self.add_anti_cues(
            "bank",
            "financial_institution",
            ["not a place for money", "no money", "without money", "not for cash"],
        )
        self.add_anti_cues(
            "bank",
            "river_edge",
            ["not near the river", "no water", "without water"],
        )
        self.add_anti_cues(
            "bat",
            "animal",
            ["not flying", "not an animal", "no wings"],
        )
        self.add_anti_cues(
            "crane",
            "bird",
            ["not a bird", "no wings"],
        )
        self.add_anti_cues(
            "crane",
            "machine",
            ["not construction", "no operator"],
        )
