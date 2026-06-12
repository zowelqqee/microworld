"""Deterministic continuation realization helpers.

This layer chooses a small typed phrase for the already-selected sense. It does
not score senses, call ML systems, or change audit policy.
"""

from __future__ import annotations

import re

from worldpgt.continuation.types import SenseEntry

ENDING_INFINITIVE = "infinitive_after_to"
ENDING_AND = "clause_after_and"
ENDING_AS = "clause_after_as"
ENDING_UNTIL_SUBJECT = "clause_after_until_subject"
ENDING_UNTIL_INCOMPLETE = "clause_after_until_incomplete"
ENDING_BEFORE_SUBJECT = "clause_after_before_subject"
ENDING_WHEN_SUBJECT = "clause_after_when_subject"
ENDING_AFTER_SUBJECT = "clause_after_after_subject"
ENDING_WHILE_SUBJECT = "clause_after_while_subject"
ENDING_NEUTRAL = "neutral_extension"

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SUBJECT_CONNECTORS = {
    "until": ENDING_UNTIL_SUBJECT,
    "before": ENDING_BEFORE_SUBJECT,
    "when": ENDING_WHEN_SUBJECT,
    "after": ENDING_AFTER_SUBJECT,
    "while": ENDING_WHILE_SUBJECT,
}
_DUPLICATE_ACTIONS = [
    "spread its wings",
    "hit",
    "opened an account",
    "lifted",
    "compressed",
    "swam",
    "marked the document",
]


def _tokens(prompt: str) -> list[str]:
    return _TOKEN_RE.findall(prompt.lower())


def classify_prompt_ending(prompt: str) -> str:
    tokens = _tokens(prompt)
    if not tokens:
        return ENDING_NEUTRAL
    if tokens[-1] == "to":
        return ENDING_INFINITIVE
    if tokens[-1] == "and":
        return ENDING_AND
    if tokens[-1] == "until":
        return ENDING_UNTIL_INCOMPLETE
    subject_connector = _ending_subject_connector(tokens)
    if subject_connector is not None:
        return subject_connector
    if "as" in tokens[-4:]:
        return ENDING_AS
    return ENDING_NEUTRAL


def compose_continuation(prompt: str, sense_entry: SenseEntry, ending_type: str) -> str:
    templates = sense_entry.continuation_templates
    candidates = templates.get(ending_type) or templates.get(ENDING_NEUTRAL) or sense_entry.continuations
    phrase = _select_non_repeating_phrase(prompt, candidates)
    if not phrase:
        return ""
    phrase = _trim_repeated_subject(prompt, phrase)
    phrase = _trim_connector_mismatch(ending_type, phrase)
    return f"{prompt.rstrip()} {phrase}".strip()


def _ending_subject_connector(tokens: list[str]) -> str | None:
    for idx in range(max(0, len(tokens) - 4), len(tokens) - 1):
        connector = tokens[idx]
        if connector in _SUBJECT_CONNECTORS and len(tokens) - idx >= 3:
            return _SUBJECT_CONNECTORS[connector]
    return None


def _select_non_repeating_phrase(prompt: str, candidates: list[str]) -> str:
    if not candidates:
        return ""
    prompt_lower = prompt.lower()
    for candidate in candidates:
        if not _would_repeat_action(prompt_lower, candidate.lower()):
            return candidate
    return candidates[0]


def _would_repeat_action(prompt_lower: str, phrase_lower: str) -> bool:
    for action in _DUPLICATE_ACTIONS:
        if action in prompt_lower and _phrase_mentions_action(phrase_lower, action):
            return True
        if action == "hit" and re.search(r"\bhit\b", prompt_lower) and re.search(r"\bhit\b", phrase_lower):
            return True
        if action == "opened an account" and "opened an account" in prompt_lower:
            if "open an account" in phrase_lower:
                return True
    return False


def _phrase_mentions_action(phrase_lower: str, action: str) -> bool:
    if action == "opened an account":
        return "open an account" in phrase_lower or "opened an account" in phrase_lower
    return action in phrase_lower


def _trim_repeated_subject(prompt: str, phrase: str) -> str:
    prompt_tokens = _tokens(prompt)
    phrase_tokens = _tokens(phrase)
    if len(prompt_tokens) < 2 or len(phrase_tokens) < 2:
        return phrase
    prompt_subject = prompt_tokens[-2:]
    phrase_subject = phrase_tokens[:2]
    if prompt_subject == phrase_subject:
        original_words = phrase.split()
        return " ".join(original_words[2:])
    return phrase


def _trim_connector_mismatch(ending_type: str, phrase: str) -> str:
    if ending_type in {
        ENDING_UNTIL_SUBJECT,
        ENDING_BEFORE_SUBJECT,
        ENDING_WHEN_SUBJECT,
        ENDING_AFTER_SUBJECT,
        ENDING_WHILE_SUBJECT,
    }:
        return re.sub(r"^(and|then)\s+", "", phrase, flags=re.IGNORECASE)
    return phrase
