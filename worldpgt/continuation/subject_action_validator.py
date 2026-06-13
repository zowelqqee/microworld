"""Role-aware subject/action validation for realized continuations.

Weightless and deterministic. Detects grammatically valid but role-invalid
subject/action pairs where a *body part* is made the subject of a *whole-animal*
action (e.g. ``its wings searched for insects``). Body parts may only perform
body-part actions (spread, folded, flicked, ...); whole-animal actions must use
the animal as actor (``it flew``, ``the seal swam``).

This module only *detects* and *describes* the violation. Repair (rewriting the
phrase or routing to audit) is performed in ``surface_repair`` so that all
surface fixes share one entry point. No ML, no weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

# Possessive / determiner that can introduce a body-part subject.
_DETERMINERS = {"its", "his", "her", "their", "the"}

# Body parts that cannot be the subject of a whole-animal action.
BODY_PARTS = {
    "wings", "wing", "flippers", "flipper", "tail", "tails",
    "fins", "fin", "paws", "paw", "claws", "claw", "beak", "beaks",
    "snout", "muzzle",
}

# Whole-animal actions: only the animal (not a body part) may perform these.
WHOLE_ANIMAL_ACTIONS = {
    "searched", "search", "searches",
    "hunted", "hunt", "hunts",
    "swam", "swim", "swims",
    "chased", "chase", "chases",
    "flew", "fly", "flies",
    "dove", "dive", "dives", "dived",
    "surfaced", "surface", "surfaces",
    "crawled", "crawl", "crawls",
    "walked", "walk", "walks",
    "ran", "run", "runs",
}

# Body-part actions that are always allowed.
ALLOWED_BODY_ACTIONS = {
    "spread", "spreads",
    "folded", "fold", "folds",
    "moved", "move", "moves",
    "flicked", "flick", "flicks",
    "beat", "beats",
    "carried", "carry", "carries",
    "propelled", "propel", "propels",
    "tucked", "tuck", "tucks",
    "opened", "open", "opens",
    "closed", "close", "closes",
    "gripped", "grip", "grips",
    "lifted", "lift", "lifts",
}

# Deterministic body-part -> safe body-part clause used to repair a violation.
# Each clause keeps the body part as subject doing a body-part-appropriate
# action; ``it`` refers to the animal. No new named entities are introduced.
BODY_PART_SAFE_CLAUSE = {
    "wings": "spread wide",
    "wing": "spread wide",
    "flippers": "moved through the water",
    "flipper": "moved through the water",
    "fins": "moved through the water",
    "fin": "moved through the water",
    "tail": "flicked behind it",
    "tails": "flicked behind it",
    "paws": "gripped the ground",
    "paw": "gripped the ground",
    "claws": "gripped the ground",
    "claw": "gripped the ground",
    "beak": "opened wide",
    "beaks": "opened wide",
    "snout": "lifted into the air",
    "muzzle": "lifted into the air",
}

_DET_GROUP = "|".join(sorted(_DETERMINERS))
_PART_GROUP = "|".join(sorted(BODY_PARTS, key=len, reverse=True))
# det + body part + verb token (verb checked against the action sets below).
_PAIR_RE = re.compile(
    rf"\b(?P<det>{_DET_GROUP})\s+(?P<part>{_PART_GROUP})\s+(?P<verb>[a-z']+)\b",
    re.IGNORECASE,
)


@dataclass
class SubjectActionResult:
    """Outcome of role-aware subject/action validation for one text."""

    ok: bool
    determiner: str | None = None
    body_part: str | None = None
    verb: str | None = None
    reasons: list[str] = field(default_factory=list)


def validate_subject_action(text: str) -> SubjectActionResult:
    """Detect a body-part subject performing a whole-animal action.

    Returns ``ok=True`` when no such violation exists (including when the body
    part performs an allowed body-part action). When a violation is found, the
    offending determiner / body part / verb are reported for repair.
    """
    for match in _PAIR_RE.finditer(text):
        verb = match.group("verb").lower()
        if verb in ALLOWED_BODY_ACTIONS:
            continue
        if verb in WHOLE_ANIMAL_ACTIONS:
            return SubjectActionResult(
                ok=False,
                determiner=match.group("det").lower(),
                body_part=match.group("part").lower(),
                verb=verb,
                reasons=[
                    "subject_action_invalid",
                    f"bad_subject_action={match.group('part').lower()}+{verb}",
                ],
            )
    return SubjectActionResult(ok=True, reasons=["subject_action_ok"])


def safe_body_part_clause(body_part: str) -> str | None:
    """Deterministic body-part-appropriate clause for a given body part."""
    return BODY_PART_SAFE_CLAUSE.get(body_part.lower())
