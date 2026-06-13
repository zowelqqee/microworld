"""Answer validator for AnswerPlanner v1.

Checks rendered answers for safety, correctness, and quality.

Rule-based and deterministic only.  No ML libraries.
No learned model parameters.  No language-model inference.  No back-prop.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_MAX_ANSWER_CHARS = 400

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
