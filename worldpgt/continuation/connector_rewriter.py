"""Deterministic connector-grammar repair for realized continuations.

Minimal-repair only. When the prompt ends in a subordinate-connector phrase whose
object is a noun event (``before the swing``, ``after sunset``) and the appended
phrase starts a *new main clause* with its own pronoun subject, the two clauses
run together without punctuation:

    "before the swing he steadied himself"

The fix is the smallest possible: insert a comma at the clause boundary:

    "before the swing, he steadied himself"

No words are added or reordered. No ML, no weights.
"""

from __future__ import annotations

import re

from worldpgt.continuation.realization import (
    ENDING_AFTER_SUBJECT,
    ENDING_BEFORE_SUBJECT,
    ENDING_UNTIL_SUBJECT,
    ENDING_WHEN_SUBJECT,
    ENDING_WHILE_SUBJECT,
)

# Subordinate-connector endings where a following main clause needs a comma.
_SUBJECT_CONNECTOR_ENDINGS = {
    ENDING_BEFORE_SUBJECT,
    ENDING_AFTER_SUBJECT,
    ENDING_UNTIL_SUBJECT,
    ENDING_WHEN_SUBJECT,
    ENDING_WHILE_SUBJECT,
}

# Bare pronoun subjects that begin a new main clause (not possessives like "its").
_MAIN_CLAUSE_SUBJECTS = {"he", "she", "it", "they", "we", "i"}

_FIRST_WORD_RE = re.compile(r"[a-z']+", re.IGNORECASE)


def _phrase_part(prompt: str, text: str) -> str:
    base = prompt.rstrip()
    if text.lower().startswith(base.lower()):
        return text[len(base):].strip()
    return text.strip()


def repair_connector_grammar(
    prompt: str,
    text: str,
    connector_type: str,
) -> tuple[str, str | None]:
    """Return ``(text, rule)``; ``rule`` is ``None`` when nothing was changed."""
    if connector_type not in _SUBJECT_CONNECTOR_ENDINGS:
        return text, None
    if "," in _phrase_part(prompt, text):
        return text, None

    phrase_part = _phrase_part(prompt, text)
    match = _FIRST_WORD_RE.match(phrase_part)
    if match is None:
        return text, None
    if match.group(0).lower() not in _MAIN_CLAUSE_SUBJECTS:
        return text, None

    base = prompt.rstrip()
    if not text.lower().startswith(base.lower()):
        return text, None
    repaired = f"{base}, {phrase_part}".strip()
    return repaired, "connector_comma"
