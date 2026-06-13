"""Answer validator for AnswerPlanner v1.

Checks rendered answers for safety, correctness, and quality.

Rule-based and deterministic only.  No ML libraries.
No learned model parameters.  No language-model inference.  No back-prop.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_MAX_ANSWER_CHARS = 400

# Spacing / concatenation artifacts that indicate a rendering defect.
# Operate on the original casing of the answer.
_SURFACE_REGEXES = (
    (re.compile(r"Itis"), "missing_space_it_is"),
    (re.compile(r"releasesmechanical"), "missing_space_releases_mechanical"),
    (re.compile(r",[A-Za-z]"), "missing_space_after_comma"),
    (re.compile(r"\.[A-Z]"), "missing_space_after_period"),
)

# Known concatenation artifacts (also covered by the comma/period regexes,
# listed explicitly for clarity).  Matched lowercase.
_KNOWN_GLUE_ARTIFACTS = (
    "baseball,associated",
    "releasesmechanical",
    "device,associated",
)

# Wrong action-agency phrasings that SemanticLanguageRealizer v2 must avoid:
# rock music does not "play", a rock does not "climb", a season does not
# "bloom", and "thaws"/"rains" are awkward plural cues.  Matched lowercase.
_BAD_AGENCY_PATTERNS = (
    "it can play and perform",
    "it can climb",
    "it can bloom and thaw",
    "common signs are thaws",
    "mornings and rains",
)

_BANNED_ANSWER_PHRASES = (
    "it depends on many things",
    "there are several possibilities",
    "in general",
    "various contexts",
    "something related",
    "many different",
    "it could mean",
    "broadly speaking",
)

# Surface-quality patterns that indicate a rendering defect.
_SURFACE_ISSUES = (
    "a animal is",        # wrong article before vowel
    "a rock music is",    # article before mass noun
    "a spring season is", # article before uncountable season label
    "typicalactions",     # missing space artifact
    "andparcel",          # missing space artifact
)

_WRONG_SENSE_SIGNALS: dict[str, dict[str, list[str]]] = {
    "bank": {
        "financial_institution": ["river", "stream", "current", "reed", "shore", "mud"],
        "river_edge": ["teller", "deposit", "loan", "account", "credit", "mortgage"],
    },
    "bat": {
        "animal": ["pitcher", "baseball", "swing", "batter", "plate", "dugout"],
        "sports_equipment": ["cave", "flying", "echolocation", "roost", "nocturnal"],
    },
    "seal": {
        "animal": ["envelope", "wax", "document", "stamp", "parcel", "notary"],
        "closure_stamp": ["flipper", "ocean", "mammal", "marine", "bark", "coast"],
    },
    "crane": {
        "bird": ["hook", "construction", "operator", "boom", "cable", "load"],
        "machine": ["marsh", "wings", "migration", "reeds", "nest", "neck"],
    },
    "rock": {
        "stone": ["band", "concert", "guitar", "stage", "drum", "album"],
        "music": ["boulder", "cliff", "mineral", "trail", "geology", "hillside"],
    },
    "spring": {
        "season": ["coil", "latch", "compressed", "metal spring", "mattress"],
        "coil": ["flowers", "april", "warm weather", "thaw", "bloom", "winter"],
    },
}


@dataclass
class ValidationResult:
    passed: bool
    reasons: list[str] = field(default_factory=list)
    quality_flagged: bool = False
    quality_reason: str = ""


def validate(
    answer: str,
    decision: str,
    term: str | None,
    sense_id: str | None,
    expected_contains: list[str] | None = None,
    intent: str | None = None,
) -> ValidationResult:
    """Validate an answer string. Returns ValidationResult."""
    reasons: list[str] = []

    # 1. Non-empty when decision=answer
    if decision == "answer" and not answer.strip():
        return ValidationResult(
            passed=False,
            reasons=["empty_answer_for_answer_decision"],
            quality_flagged=True,
            quality_reason="empty_answer",
        )

    # 2. Audit answers must not hallucinate a continuation
    if decision == "audit":
        lower = answer.lower()
        hallucination_signals = [
            "means financial",
            "means river",
            "is a bat",
            "is a seal",
            "is a crane",
            "is a rock",
            "is a spring",
        ]
        for sig in hallucination_signals:
            if sig in lower:
                return ValidationResult(
                    passed=False,
                    reasons=[f"audit_answer_hallucinating:{sig}"],
                    quality_flagged=True,
                    quality_reason="audit_hallucination",
                )
        return ValidationResult(passed=True)

    lower = answer.lower()

    # 3. No banned answer phrases
    for phrase in _BANNED_ANSWER_PHRASES:
        if phrase in lower:
            return ValidationResult(
                passed=False,
                reasons=[f"generic_fallback_phrase:{phrase!r}"],
                quality_flagged=True,
                quality_reason=f"generic_fallback:{phrase!r}",
            )

    # 4. Answer must not mention wrong-sense signals
    # (skip for distinguish_senses: those answers intentionally cover both senses)
    if term and sense_id and intent != "distinguish_senses":
        wrong_signals = _WRONG_SENSE_SIGNALS.get(term, {}).get(sense_id, [])
        for sig in wrong_signals:
            if sig in lower:
                return ValidationResult(
                    passed=False,
                    reasons=[f"wrong_sense_signal:{sig!r}"],
                    quality_flagged=True,
                    quality_reason=f"wrong_sense:{sig!r}",
                )

    # 5. Answer must be short enough
    if len(answer) > _MAX_ANSWER_CHARS:
        return ValidationResult(
            passed=False,
            reasons=[f"answer_too_long:{len(answer)}"],
            quality_flagged=True,
            quality_reason="too_long",
        )

    # 5b. Surface quality checks
    for issue in _SURFACE_ISSUES:
        if issue in lower:
            return ValidationResult(
                passed=False,
                reasons=[f"surface_issue:{issue!r}"],
                quality_flagged=True,
                quality_reason=f"surface_issue:{issue!r}",
            )
    if "  " in answer:
        return ValidationResult(
            passed=False,
            reasons=["surface_issue:'double_space'"],
            quality_flagged=True,
            quality_reason="surface_issue:'double_space'",
        )
    if intent != "distinguish_senses" and lower.count("associated with") > 1:
        return ValidationResult(
            passed=False,
            reasons=["surface_issue:'repeated_associated_with'"],
            quality_flagged=True,
            quality_reason="surface_issue:'repeated_associated_with'",
        )

    # 5c. define_sense answers must use SemanticLanguageRealizer v1 phrasing,
    # not the flat "It is associated with ..." evidence-list form.
    if intent == "define_sense" and "it is associated with" in lower:
        return ValidationResult(
            passed=False,
            reasons=["surface_issue:'define_sense_it_is_associated_with'"],
            quality_flagged=True,
            quality_reason="surface_issue:'define_sense_it_is_associated_with'",
        )

    # 5c-contrast. distinguish_senses answers must use ContrastRealizer v1,
    # not the flat "associated with ..." evidence-list form.
    if intent == "distinguish_senses" and "associated with" in lower:
        return ValidationResult(
            passed=False,
            reasons=["surface_issue:'distinguish_associated_with'"],
            quality_flagged=True,
            quality_reason="surface_issue:'distinguish_associated_with'",
        )

    # 5c-glue. Known concatenation artifacts.
    for artifact in _KNOWN_GLUE_ARTIFACTS:
        if artifact in lower:
            return ValidationResult(
                passed=False,
                reasons=[f"surface_issue:{artifact!r}"],
                quality_flagged=True,
                quality_reason=f"surface_issue:{artifact!r}",
            )

    # 5c-bis. Wrong action-agency phrasings.
    for phrase in _BAD_AGENCY_PATTERNS:
        if phrase in lower:
            return ValidationResult(
                passed=False,
                reasons=[f"agency_issue:{phrase!r}"],
                quality_flagged=True,
                quality_reason=f"agency_issue:{phrase!r}",
            )

    # 5d. Spacing / concatenation artifacts.
    for pattern, name in _SURFACE_REGEXES:
        if pattern.search(answer):
            return ValidationResult(
                passed=False,
                reasons=[f"surface_issue:{name!r}"],
                quality_flagged=True,
                quality_reason=f"surface_issue:{name!r}",
            )

    # 6. expected_contains check
    if expected_contains:
        for keyword in expected_contains:
            kw = keyword.strip().lower()
            if kw and kw not in lower:
                reasons.append(f"missing_expected_keyword:{kw!r}")
        if reasons:
            return ValidationResult(
                passed=False,
                reasons=reasons,
                quality_flagged=True,
                quality_reason="missing_keywords",
            )

    return ValidationResult(passed=True, reasons=reasons)
