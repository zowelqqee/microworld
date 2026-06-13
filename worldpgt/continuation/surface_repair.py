"""Deterministic surface-repair layer.

Runs *after* ``semantic_renderer_v2`` has selected a single best candidate and
*before* final emission. It applies minimal, rule-based fixes for grammar / role
/ surface-realization bugs the renderer's phrase library can still produce:

    selected candidate
        -> attachment-drift check        (subject drift / dangling clause)
        -> subject/action repair          (body-part subject doing animal action)
        -> connector-grammar repair       (missing comma at clause boundary)
        -> coreference / repetition repair (repeated object noun -> pronoun/prey)
        -> final validation               (surface + role + drift + repetition)
        -> safe repaired text  OR  audit (no_safe_repaired_candidate)

Every change is a fixed, deterministic string transform: no words are invented,
no ML, no neural weights, no sampling. When a candidate cannot be repaired
without drift or a remaining bug, the result is marked unsafe and the engine
emits an audit instead of a risky continuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from worldpgt.continuation.connector_rewriter import repair_connector_grammar
from worldpgt.continuation.coreference_repair import repair_repetition
from worldpgt.continuation.semantic_frame import SemanticFrame, _ACTOR_NOUNS
from worldpgt.continuation.subject_action_validator import (
    safe_body_part_clause,
    validate_subject_action,
)
from worldpgt.continuation.surface_validator import validate_surface_text

AUDIT_REASON = "no_safe_repaired_candidate"

_TOKEN_RE = re.compile(r"[a-z0-9']+")
_SUBORDINATE_CONNECTORS = {"until", "before", "when", "after", "while"}
# Positional / stative verbs naturally belong to an inanimate subject. When one
# is appended across an intervening human actor its subject becomes ambiguous.
_POSITIONAL_VERBS = {
    "lay", "lie", "lies", "lying",
    "rested", "rest", "rests",
    "sat", "sit", "sits",
    "stood", "stand", "stands",
    "remained", "remain", "remains",
}
_CLAUSE_SUBJECT_WORDS = {
    "the", "a", "an", "it", "he", "she", "they", "we", "i",
    "its", "his", "her", "their",
}
_ACTOR_SET = set(_ACTOR_NOUNS)

_DRIFT_WORDS = {
    "said", "asked", "told", "replied", "shouted", "whispered",
    "mother", "father", "boyfriend", "girlfriend", "pregnancy",
    "wife", "husband", "daughter", "son",
}
_DRIFT_CHARS = ('"', "'", "“", "”", "‘", "’", "\n")


@dataclass
class SurfaceRepairResult:
    """Auditable outcome of the surface-repair layer for one candidate."""

    text: str
    applied: bool
    safe: bool
    rules: list[str] = field(default_factory=list)
    audit_reason: str | None = None
    reasons: list[str] = field(default_factory=list)


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _phrase_part(prompt: str, text: str) -> str:
    base = prompt.rstrip()
    if text.lower().startswith(base.lower()):
        return text[len(base):].strip()
    return text.strip()


def _has_attachment_drift(prompt: str, text: str) -> bool:
    """Dangling ``and <bare-verb>`` clause attaching across a second actor.

    The appended phrase starts ``and <positional-verb>`` with no explicit subject,
    while the prompt has already introduced a *second* actor inside a subordinate
    clause (``until the climber ...``). The positional verb's natural (inanimate)
    subject then sits across an intervening human actor, leaving the subject
    ambiguous -> drift. A simple compound predicate over a single subject (no
    subordinate second actor), or a non-positional action verb, is left untouched.
    """
    phrase_part = _phrase_part(prompt, text).lower()
    match = re.match(r"^and\s+(?P<next>[a-z']+)\b", phrase_part)
    if match is None:
        return False
    if match.group("next") in _CLAUSE_SUBJECT_WORDS:
        return False  # "and the wax ...", "and it ..." -> explicit subject, fine.
    if match.group("next") not in _POSITIONAL_VERBS:
        return False  # "and filled the stadium" -> attaches to the local actor.

    prompt_tokens = _tokens(prompt)
    for idx, token in enumerate(prompt_tokens[:-1]):
        if token not in _SUBORDINATE_CONNECTORS:
            continue
        for follow in prompt_tokens[idx + 1: idx + 4]:
            if follow in _ACTOR_SET:
                return True
    return False


def _repair_subject_action(
    prompt: str,
    text: str,
    frame: SemanticFrame,
) -> tuple[str, str | None, bool]:
    """Return ``(text, rule, safe)``. ``safe=False`` means route to audit."""
    result = validate_subject_action(text)
    if result.ok:
        return text, None, True

    base = prompt.rstrip()
    boundary = f"{result.determiner} {result.body_part}"
    safe_clause = safe_body_part_clause(result.body_part or "")
    animal_sense = frame.sense_id in {"animal", "bird"}

    # Repair only when the animal reading is selected and the body part sits at
    # the prompt boundary (so we substitute only the appended action clause).
    if animal_sense and safe_clause and base.lower().endswith(boundary):
        repaired = f"{base} {safe_clause}".strip()
        return repaired, "subject_action_bodypart_rewrite", True

    return text, None, False


def _has_drift(prompt: str, text: str) -> bool:
    phrase_part = _phrase_part(prompt, text)
    if any(ch in phrase_part for ch in _DRIFT_CHARS):
        return True
    return any(tok in _DRIFT_WORDS for tok in _tokens(phrase_part))


def _has_repeated_object(prompt: str, text: str) -> bool:
    phrase_part = _phrase_part(prompt, text)
    prompt_nouns = set(_tokens(prompt))
    for match in re.finditer(r"\b([a-z']+)\s+the\s+([a-z']+)\b", phrase_part, re.IGNORECASE):
        verb, noun = match.group(1).lower(), match.group(2).lower()
        if verb in {"and", "or", "but", "the", "a", "an"}:
            continue
        if noun in prompt_nouns:
            return True
    return False


def repair_surface_candidate(
    prompt: str,
    candidate_text: str,
    frame: SemanticFrame,
) -> SurfaceRepairResult:
    """Apply deterministic surface repairs, then re-validate.

    Returns a :class:`SurfaceRepairResult`. ``safe=False`` signals the engine to
    emit an audit with ``audit_reason=no_safe_repaired_candidate`` instead of the
    (un-repairable) continuation.
    """
    rules: list[str] = []
    reasons: list[str] = []
    text = candidate_text

    # 1. Attachment drift -> cannot be repaired without inventing a subject.
    if _has_attachment_drift(prompt, text):
        reasons.append("attachment_drift")
        return SurfaceRepairResult(
            text=candidate_text,
            applied=False,
            safe=False,
            rules=rules,
            audit_reason=AUDIT_REASON,
            reasons=reasons,
        )

    # 2. Subject / action role repair.
    text, sa_rule, sa_safe = _repair_subject_action(prompt, text, frame)
    if not sa_safe:
        reasons.append("subject_action_validator=rejected")
        return SurfaceRepairResult(
            text=candidate_text,
            applied=False,
            safe=False,
            rules=rules,
            audit_reason=AUDIT_REASON,
            reasons=reasons,
        )
    if sa_rule is not None:
        rules.append(sa_rule)
        reasons.append("subject_action_validator=repaired")
    else:
        reasons.append("subject_action_validator=passed")

    # 3. Connector-grammar repair (comma at clause boundary).
    text, connector_rule = repair_connector_grammar(prompt, text, frame.connector_type)
    if connector_rule is not None:
        rules.append(connector_rule)
        reasons.append("connector_rewriter=applied")

    # 4. Coreference / repetition repair.
    text, coref_rule = repair_repetition(prompt, text, frame.sense_id)
    if coref_rule is not None:
        rules.append(coref_rule)
        reasons.append("coreference_repair=applied")

    # 5. Final validation after repair.
    failures: list[str] = []
    surface = validate_surface_text(prompt, text)
    if not surface.ok:
        failures.extend(f"surface_pattern={p}" for p in surface.matched_patterns)
    if not validate_subject_action(text).ok:
        failures.append("subject_action_invalid")
    if _has_drift(prompt, text):
        failures.append("story_drift")
    if _has_repeated_object(prompt, text):
        failures.append("repeated_object")

    if failures:
        reasons.extend(failures)
        return SurfaceRepairResult(
            text=candidate_text,
            applied=False,
            safe=False,
            rules=rules,
            audit_reason=AUDIT_REASON,
            reasons=reasons,
        )

    applied = bool(rules)
    if applied:
        reasons.insert(0, "surface_repair=applied")
        reasons.extend(f"surface_repair_rule={rule}" for rule in rules)
    else:
        reasons.insert(0, "surface_repair=clean")

    return SurfaceRepairResult(
        text=text,
        applied=applied,
        safe=True,
        rules=rules,
        audit_reason=None,
        reasons=reasons,
    )
