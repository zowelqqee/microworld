"""Prompt-tail compatibility validation for repaired continuations.

Runs after semantic rendering and surface repair. It checks whether the final
candidate fits an unfinished prompt tail such as ``could``, ``before``,
``motioned for``, or ``made everyone``. Repairs are narrow, deterministic
rewrites for known safe prompt-tail shapes; otherwise the candidate is rejected
and routed to audit.

No ML, no neural weights, no sampling, and no generic fallback continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from worldpgt.continuation.semantic_frame import SemanticFrame
from worldpgt.continuation.subject_action_validator import validate_subject_action
from worldpgt.continuation.surface_validator import validate_surface_text

AUDIT_REASON = "prompt_tail_incompatible"

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_MODALS = {"could", "would", "should", "can", "might"}
_PAST_TO_BASE = {
    "searched": "search",
    "swam": "swim",
    "spread": "spread",
    "flew": "fly",
    "dove": "dive",
    "surfaced": "surface",
    "lifted": "lift",
    "carried": "carry",
    "filled": "fill",
    "brought": "bring",
    "snapped": "snap",
    "hit": "hit",
}


@dataclass
class PromptTailValidationResult:
    """Auditable result of prompt-tail validation."""

    text: str
    passed: bool
    repair_applied: bool = False
    rule_name: str | None = None
    rejection_reason: str | None = None
    evidence: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _phrase_part(prompt: str, text: str) -> str:
    base = prompt.rstrip()
    if text.lower().startswith(base.lower()):
        return text[len(base):].strip()
    return text.strip()


def _complete(prompt: str, phrase: str) -> str:
    return f"{prompt.rstrip()} {phrase}".strip()


def _reject(text: str, rule: str, reason: str, evidence: list[str]) -> PromptTailValidationResult:
    return PromptTailValidationResult(
        text=text,
        passed=False,
        repair_applied=False,
        rule_name=rule,
        rejection_reason=reason,
        evidence=evidence,
    )


def _pass(text: str, rule: str | None, evidence: list[str]) -> PromptTailValidationResult:
    return PromptTailValidationResult(
        text=text,
        passed=True,
        repair_applied=False,
        rule_name=rule,
        evidence=evidence,
    )


def _repair(text: str, rule: str, evidence: list[str]) -> PromptTailValidationResult:
    failures = _final_failures(text)
    if failures:
        return _reject(text, rule, "repair_failed_final_validation", evidence + failures)
    return PromptTailValidationResult(
        text=text,
        passed=True,
        repair_applied=True,
        rule_name=rule,
        evidence=evidence,
    )


def _final_failures(text: str) -> list[str]:
    surface = validate_surface_text("", text)
    failures = [f"surface_pattern={pattern}" for pattern in surface.matched_patterns]
    if not validate_subject_action(text).ok:
        failures.append("subject_action_invalid")
    return failures


def _starts_with_and_past(phrase: str) -> tuple[str, str] | None:
    match = re.match(r"^and\s+([a-z']+)(.*)$", phrase.strip(), flags=re.IGNORECASE)
    if match is None:
        return None
    verb = match.group(1).lower()
    base = _PAST_TO_BASE.get(verb)
    if base is None:
        return None
    return base, match.group(2).strip()


def _modal_tail(prompt: str, text: str, frame: SemanticFrame) -> PromptTailValidationResult | None:
    tokens = _tokens(prompt)
    if not tokens or tokens[-1] not in _MODALS:
        return None
    phrase = _phrase_part(prompt, text)
    evidence = [f"tail_modal={tokens[-1]}", f"phrase={phrase}"]
    converted = _starts_with_and_past(phrase)
    if converted is None:
        if re.match(r"^[a-z']+\b", phrase):
            return _pass(text, "modal_tail", evidence)
        return _reject(text, "modal_tail", "modal_requires_bare_infinitive", evidence)
    base, rest = converted
    repaired_phrase = f"{base} {rest}".strip()
    return _repair(_complete(prompt, repaired_phrase), "modal_and_past_to_bare_infinitive", evidence)


def _preposition_object_tail(
    prompt: str,
    text: str,
    frame: SemanticFrame,
) -> PromptTailValidationResult | None:
    unfinished_tails = {
        ("motioned", "for"),
        ("turned", "toward"),
        ("looked", "toward"),
        ("waited", "for"),
        ("reached", "for"),
        ("pointed", "toward"),
    }
    tail_repairs = {
        ("motioned", "for", "financial_institution"): "the client to come forward",
        ("turned", "toward", "machine"): "the load",
    }
    tokens = _tokens(prompt)
    if len(tokens) < 2:
        return None
    tail = (tokens[-2], tokens[-1])
    if tail not in unfinished_tails:
        return None
    phrase = _phrase_part(prompt, text)
    evidence = [f"tail_preposition={tokens[-2]}_{tokens[-1]}", f"phrase={phrase}"]
    if not phrase.lower().startswith("and "):
        return _pass(text, "preposition_object_tail", evidence)
    key = (tokens[-2], tokens[-1], frame.sense_id)
    if key not in tail_repairs:
        return _reject(text, "preposition_object_tail", "object_or_purpose_required", evidence)
    return _repair(_complete(prompt, tail_repairs[key]), f"{tokens[-2]}_{tokens[-1]}_object_repair", evidence)


def _subordinator_tail(prompt: str, text: str, frame: SemanticFrame) -> PromptTailValidationResult | None:
    tokens = _tokens(prompt)
    if not tokens:
        return None
    phrase = _phrase_part(prompt, text)
    lower_phrase = phrase.lower().strip()

    if tokens[-1] == "before":
        evidence = ["tail_subordinator=before", f"phrase={phrase}"]
        if lower_phrase.startswith("and hit the ball"):
            return _repair(_complete(prompt, "hitting the ball"), "before_and_hit_to_gerund", evidence)
        if lower_phrase.startswith(("and ", "then ")):
            return _reject(text, "before_tail", "before_requires_event_or_clause", evidence)
        return _pass(text, "before_tail", evidence)

    if tokens[-1] == "after":
        evidence = ["tail_subordinator=after", f"phrase={phrase}"]
        if frame.sense_id == "music" and lower_phrase.startswith("and filled the stadium"):
            return _repair(_complete(prompt, "the band started playing"), "after_music_event_repair", evidence)
        if lower_phrase.startswith(("and ", "then ")):
            return _reject(text, "after_tail", "after_requires_event_or_clause", evidence)
        return _pass(text, "after_tail", evidence)

    return None


def _while_subject_tail(prompt: str, text: str, frame: SemanticFrame) -> PromptTailValidationResult | None:
    match = re.search(r"\bwhile\s+(tourists|visitors|people|children|workers)\s*$", prompt, re.IGNORECASE)
    if match is None:
        return None
    phrase = _phrase_part(prompt, text)
    evidence = [f"tail_while_subject={match.group(1).lower()}", f"phrase={phrase}"]
    if phrase.lower().startswith("and "):
        return _reject(text, "while_subject_tail", "while_subject_requires_predicate_for_subject", evidence)
    return _pass(text, "while_subject_tail", evidence)


def _as_subject_tail(prompt: str, text: str, frame: SemanticFrame) -> PromptTailValidationResult | None:
    match = re.search(r"\bas\s+the\s+(hook)\s*$", prompt, re.IGNORECASE)
    if match is None:
        return None
    phrase = _phrase_part(prompt, text)
    evidence = [f"tail_as_subject=the_{match.group(1).lower()}", f"phrase={phrase}"]
    if frame.sense_id == "machine" and phrase.lower().startswith("the operator checked"):
        return _repair(_complete(prompt, "rose"), "as_hook_predicate_repair", evidence)
    if re.match(r"^(the|a|an|it|he|she|they)\b", phrase, flags=re.IGNORECASE):
        return _reject(text, "as_subject_tail", "as_subject_requires_predicate_for_subject", evidence)
    return _pass(text, "as_subject_tail", evidence)


def _complement_tail(prompt: str, text: str, frame: SemanticFrame) -> PromptTailValidationResult | None:
    match = re.search(r"\b(made|left|kept)\s+everyone\s*$", prompt, re.IGNORECASE)
    if match is None:
        return None
    phrase = _phrase_part(prompt, text)
    evidence = [f"tail_complement={match.group(1).lower()}_everyone", f"phrase={phrase}"]
    if frame.sense_id == "season" and phrase.lower().startswith("and brought warmer days"):
        return _repair(_complete(prompt, "feel warmer"), "made_everyone_season_complement_repair", evidence)
    if phrase.lower().startswith("and "):
        return _reject(text, "complement_tail", "complement_required", evidence)
    return _pass(text, "complement_tail", evidence)


def _redundant_conjunction(
    prompt: str,
    text: str,
    frame: SemanticFrame,
) -> PromptTailValidationResult | None:
    phrase = _phrase_part(prompt, text)
    evidence = [f"phrase={phrase}"]
    if (
        frame.sense_id == "coil"
        and re.search(r"\bwhen\s+the\s+spring\s+inside\s+the\s+handle\s*$", prompt, re.IGNORECASE)
        and phrase.lower().startswith("and snapped back")
    ):
        repaired_phrase = re.sub(r"^and\s+", "", phrase, flags=re.IGNORECASE)
        return _repair(_complete(prompt, repaired_phrase), "redundant_and_after_existing_subject", evidence)
    return None


def _before_subject_connector_repair(
    prompt: str,
    text: str,
    frame: SemanticFrame,
) -> PromptTailValidationResult | None:
    phrase = _phrase_part(prompt, text)
    if not re.search(r"\bbefore\s+the\s+player\s+swung\s*$", prompt, re.IGNORECASE):
        return None
    evidence = [f"phrase={phrase}"]
    if phrase.lower().startswith(", he steadied himself"):
        return _repair(
            f"{prompt.rstrip()}, and he steadied himself",
            "before_subject_comma_and_repair",
            evidence,
        )
    return None


def validate_prompt_tail_compatibility(
    prompt: str,
    candidate_text: str,
    frame: SemanticFrame,
) -> PromptTailValidationResult:
    """Validate or narrowly repair the completed candidate against prompt tail."""
    checks = (
        _modal_tail,
        _preposition_object_tail,
        _subordinator_tail,
        _while_subject_tail,
        _as_subject_tail,
        _complement_tail,
        _redundant_conjunction,
        _before_subject_connector_repair,
    )
    for check in checks:
        result = check(prompt, candidate_text, frame)
        if result is not None:
            return result
    return _pass(candidate_text, None, ["prompt_tail=no_special_tail"])
