"""Deterministic coreference / repeated-object repair for continuations.

Local, phrase-level only. Two rules, applied to the appended phrase:

1. Prey repetition (animal senses): ``<verb> another <prey>`` / ``<verb> the
   <prey>`` where the prey noun already appears in the prompt becomes
   ``<verb> its prey``  (``catch another fish`` -> ``catch its prey``).

2. Repeated object: ``<verb> the <noun>`` where the noun already appears in the
   prompt becomes ``<verb> it``  (``close the envelope`` -> ``close it``).

These never invent new entities; they replace a redundant repeated noun with a
pronoun the reader already resolves. No ML, no weights.
"""

from __future__ import annotations

import re

_TOKEN_RE = re.compile(r"[a-z0-9']+")

# Animal-food nouns that read better as "its prey" on the second mention.
PREY_NOUNS = {"fish", "prey"}

_ANIMAL_SENSES = {"animal", "bird"}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _phrase_part(prompt: str, text: str) -> tuple[str, str]:
    """Return ``(base, phrase_part)`` splitting the prompt prefix from the tail."""
    base = prompt.rstrip()
    if text.lower().startswith(base.lower()):
        return base, text[len(base):].strip()
    return "", text.strip()


def repair_repetition(
    prompt: str,
    text: str,
    sense_id: str,
) -> tuple[str, str | None]:
    """Return ``(text, rule)``; ``rule`` is ``None`` when nothing was changed."""
    base, phrase_part = _phrase_part(prompt, text)
    if not base or not phrase_part:
        return text, None

    prompt_nouns = set(_tokens(prompt))

    # Rule 1: prey repetition for animal senses.
    if sense_id in _ANIMAL_SENSES:
        prey_group = "|".join(sorted(PREY_NOUNS))
        prey_re = re.compile(
            rf"\b(?P<verb>[a-z']+)\s+(?:another|the|more)\s+(?P<prey>{prey_group})\b",
            re.IGNORECASE,
        )
        match = prey_re.search(phrase_part)
        if match is not None and match.group("prey").lower() in prompt_nouns:
            new_phrase = (
                phrase_part[: match.start()]
                + f"{match.group('verb')} its prey"
                + phrase_part[match.end():]
            ).strip()
            return f"{base} {new_phrase}".strip(), "coreference_prey"

    # Rule 2: repeated object noun -> pronoun "it".
    obj_re = re.compile(r"\b(?P<verb>[a-z']+)\s+the\s+(?P<noun>[a-z']+)\b", re.IGNORECASE)
    for match in obj_re.finditer(phrase_part):
        noun = match.group("noun").lower()
        verb = match.group("verb").lower()
        if verb in {"and", "or", "but", "the", "a", "an"}:
            continue
        if noun in prompt_nouns:
            new_phrase = (
                phrase_part[: match.start()]
                + f"{match.group('verb')} it"
                + phrase_part[match.end():]
            ).strip()
            return f"{base} {new_phrase}".strip(), "coreference_pronoun"

    return text, None
