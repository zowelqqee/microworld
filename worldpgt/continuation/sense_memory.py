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
        continuation_templates: dict[str, list[str]] | None = None,
        trust: float = 1.0,
    ) -> SenseEntry:
        entry = SenseEntry(
            term=term.lower(),
            sense_id=sense_id,
            cues=[c.lower() for c in cues],
            continuations=list(continuations),
            continuation_templates={
                key: list(values)
                for key, values in (continuation_templates or {}).items()
            },
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
        if term == "bat" and sense_id == "animal":
            if any(cue in positive_cues for cue in {"cave", "wings"}) and (
                "logo" in tokens or "painted" in tokens
            ):
                return ["depicted_animal_cue_insufficient"]
        if term == "crane" and sense_id == "machine":
            if "steel" in positive_cues and "sculpture" in tokens:
                return ["depicted_machine_cue_insufficient"]
        if term == "spring" and sense_id == "coil":
            if "metal" in positive_cues and "sign" in tokens:
                return ["depicted_coil_cue_insufficient"]
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
            [
                "money",
                "loan",
                "account",
                "teller",
                "deposit",
                "cash",
                "card",
                "mortgage",
                "credit",
                "customer",
                "client",
                "counter",
                "manager",
                "lobby",
            ],
            ["open an account", "deposit money", "ask for a loan"],
            {
                "infinitive_after_to": ["open an account", "speak with the teller"],
                "clause_after_and": ["the teller processed the request"],
                "clause_after_as": ["the teller checked the paperwork"],
                "clause_after_until_subject": ["the teller called her forward"],
                "clause_after_before_subject": ["the teller checked the paperwork"],
                "clause_after_when_subject": ["the teller called her forward"],
                "neutral_extension": ["and completed the transaction"],
            },
        )
        self.add_sense(
            "bank",
            "river_edge",
            [
                "river",
                "fisherman",
                "water",
                "shore",
                "mud",
                "stream",
                "current",
                "boat",
                "bridge",
                "reeds",
                "path",
            ],
            ["cast his line", "watched the current", "sat near the water"],
            {
                "infinitive_after_to": ["watch the current"],
                "clause_after_and": ["the water moved past the reeds"],
                "clause_after_as": ["carried it downstream"],
                "clause_after_until_subject": ["the boat touched the mud"],
                "clause_after_before_subject": ["the water reached the shore"],
                "clause_after_when_subject": ["the current slowed"],
                "neutral_extension": ["and watched the current"],
            },
        )
        self.add_sense(
            "bat",
            "animal",
            [
                "cave",
                "flew",
                "flying",
                "wings",
                "night",
                "animal",
                "hanging",
                "attic",
                "dusk",
                "eaves",
                "light",
            ],
            ["flew into the dark cave", "hung from the ceiling", "searched for insects"],
            {
                "infinitive_after_to": ["search for insects"],
                "clause_after_and": ["it spread its wings", "it searched for insects"],
                "clause_after_as": ["it flew into the dark"],
                "clause_after_until_subject": ["it stirred at dusk"],
                "clause_after_before_subject": ["spread its wings"],
                "clause_after_when_subject": ["flew into the dark"],
                "neutral_extension": ["and searched for insects", "and flew into the dark"],
            },
        )
        self.add_sense(
            "bat",
            "sports_equipment",
            [
                "baseball",
                "player",
                "hit",
                "cracked",
                "swing",
                "game",
                "batter",
                "plate",
                "dugout",
                "swung",
                "lighter",
            ],
            ["hit the ball", "cracked after the swing", "was dropped near home plate"],
            {
                "infinitive_after_to": ["hit the ball"],
                "clause_after_and": ["the player took a swing"],
                "clause_after_as": ["the pitcher released the ball"],
                "clause_after_until_subject": ["the pitcher threw the ball"],
                "clause_after_before_subject": ["the player took a swing", "he steadied himself"],
                "clause_after_when_subject": ["the player took a swing"],
                "neutral_extension": ["and hit the ball", "and the player dropped the bat"],
            },
        )
        self.add_sense(
            "seal",
            "animal",
            [
                "ocean",
                "fish",
                "zoo",
                "flippers",
                "animal",
                "water",
                "trainer",
                "treat",
                "slid",
                "pier",
                "tourists",
                "splash",
                "swimming",
            ],
            ["swam through the cold water", "balanced on the rock", "looked for fish"],
            {
                "infinitive_after_to": ["catch another fish"],
                "clause_after_and": ["it dove below the surface", "it surfaced for air"],
                "clause_after_as": ["it followed the fish"],
                "clause_after_until_subject": ["it reached the deeper water"],
                "clause_after_before_subject": ["dove below the surface"],
                "clause_after_when_subject": ["dove below the surface"],
                "neutral_extension": ["and swam through the cold water", "and dove below the surface"],
            },
        )
        self.add_sense(
            "seal",
            "closure_stamp",
            [
                "envelope",
                "document",
                "stamp",
                "wax",
                "official",
                "package",
                "clerk",
                "closing",
                "parcel",
                "flap",
                "label",
            ],
            ["closed the envelope", "marked the document", "protected the package"],
            {
                "infinitive_after_to": ["close the envelope"],
                "clause_after_and": ["the wax hardened"],
                "clause_after_as": ["the wax cooled"],
                "clause_after_until_subject": ["the wax hardened"],
                "clause_after_before_subject": ["the wax hardened"],
                "clause_after_when_subject": ["the wax hardened"],
                "neutral_extension": ["and the wax hardened", "and marked the document"],
            },
        )
        self.add_sense(
            "crane",
            "bird",
            [
                "bird",
                "wings",
                "marsh",
                "flew",
                "nest",
                "lake",
                "dawn",
                "reeds",
                "neck",
                "photographer",
            ],
            ["flew over the marsh", "stood near the lake", "spread its wings"],
            {
                "infinitive_after_to": ["step through the reeds"],
                "clause_after_and": ["it spread its wings", "it crossed the shallows"],
                "clause_after_as": ["it crossed the shallows"],
                "clause_after_until_subject": ["it reached the lake"],
                "clause_after_before_subject": ["spread its wings"],
                "clause_after_when_subject": ["crossed the shallows"],
                "neutral_extension": ["and spread its wings", "and crossed the shallows"],
            },
        )
        self.add_sense(
            "crane",
            "machine",
            [
                "construction",
                "building",
                "lifted",
                "steel",
                "operator",
                "site",
                "foreman",
                "crew",
                "hook",
                "load",
                "lift",
            ],
            ["lifted the steel beam", "moved above the construction site", "carried the load"],
            {
                "infinitive_after_to": ["lift the load"],
                "clause_after_and": ["the operator raised the load"],
                "clause_after_as": ["the operator checked the cables"],
                "clause_after_until_subject": ["the operator lowered the hook"],
                "clause_after_before_subject": ["the operator checked the cables"],
                "clause_after_when_subject": ["the operator raised the load"],
                "clause_after_while_subject": ["the operator checked the cables"],
                "neutral_extension": ["and carried the load", "and the operator checked the cables"],
            },
        )
        self.add_sense(
            "spring",
            "season",
            [
                "flowers",
                "april",
                "warm",
                "weather",
                "garden",
                "rain",
                "thaw",
                "mornings",
                "warmed",
            ],
            ["brought warmer days", "filled the garden with flowers", "came after winter"],
            {
                "infinitive_after_to": ["bring warmer days"],
                "clause_after_and": ["the garden filled with flowers", "the flowers opened"],
                "clause_after_as": ["the weather warmed"],
                "clause_after_until_subject": ["the rain softened the ground"],
                "clause_after_before_subject": ["the weather warmed"],
                "clause_after_when_subject": ["the weather warmed"],
                "neutral_extension": ["and brought warmer days"],
            },
        )
        self.add_sense(
            "spring",
            "coil",
            [
                "metal",
                "compressed",
                "mechanism",
                "tension",
                "bounce",
                "device",
                "latch",
                "handle",
            ],
            ["compressed under pressure", "snapped back into place", "stored mechanical energy"],
            {
                "infinitive_after_to": ["store mechanical energy"],
                "clause_after_and": ["the mechanism snapped back", "the tension released"],
                "clause_after_as": ["the device compressed it"],
                "clause_after_until_subject": ["the latch released"],
                "clause_after_before_subject": ["the mechanism snapped back"],
                "clause_after_when_subject": ["the mechanism snapped back"],
                "clause_after_after_subject": ["the mechanism snapped back into place"],
                "neutral_extension": ["and snapped back into place"],
            },
        )
        self.add_sense(
            "rock",
            "stone",
            ["mountain", "stone", "river", "heavy", "ground", "cliff", "trail", "boulder"],
            ["rolled down the hill", "lay near the river", "broke into smaller stones"],
            {
                "infinitive_after_to": ["roll down the hill"],
                "clause_after_and": ["it scraped the ground", "it rolled downhill"],
                "clause_after_as": ["it slid down the slope"],
                "clause_after_until_subject": ["it reached the ground"],
                "clause_after_before_subject": ["scraped the ground"],
                "clause_after_when_subject": ["scraped the ground"],
                "neutral_extension": ["and lay near the river"],
            },
        )
        self.add_sense(
            "rock",
            "music",
            ["band", "guitar", "concert", "song", "drummer", "stage", "crowd", "louder", "venue"],
            ["filled the stadium", "played through the speakers", "started with a loud guitar riff"],
            {
                "infinitive_after_to": ["fill the room"],
                "clause_after_and": ["the band started playing"],
                "clause_after_as": ["the guitarist started the riff"],
                "clause_after_until_subject": ["the drummer counted in"],
                "clause_after_before_subject": ["the band started playing"],
                "clause_after_when_subject": ["the guitar started playing"],
                "neutral_extension": ["and filled the stadium"],
            },
        )
        self.add_anti_cues(
            "bank",
            "financial_institution",
            [
                "not a place for money",
                "no money",
                "without money",
                "not for cash",
                "not beside a river",
            ],
        )
        self.add_anti_cues(
            "bank",
            "river_edge",
            [
                "not near the river",
                "no water",
                "without water",
                "not a place for money",
                "not about credit",
            ],
        )
        self.add_anti_cues(
            "bat",
            "animal",
            ["not flying", "not an animal", "no wings"],
        )
        self.add_anti_cues(
            "bat",
            "sports_equipment",
            ["not an animal", "not flying"],
        )
        self.add_anti_cues(
            "seal",
            "animal",
            ["not a stamp"],
        )
        self.add_anti_cues(
            "seal",
            "closure_stamp",
            ["not an animal"],
        )
        self.add_anti_cues(
            "crane",
            "bird",
            ["not a bird", "no wings", "not at a construction site"],
        )
        self.add_anti_cues(
            "crane",
            "machine",
            ["not construction", "no operator"],
        )
        self.add_anti_cues(
            "spring",
            "season",
            ["not a metal device"],
        )
        self.add_anti_cues(
            "spring",
            "coil",
            ["not warm weather"],
        )
        self.add_anti_cues(
            "rock",
            "stone",
            ["not a song"],
        )
        self.add_anti_cues(
            "rock",
            "music",
            ["not a stone"],
        )
