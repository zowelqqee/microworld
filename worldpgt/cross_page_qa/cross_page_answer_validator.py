"""Answer validator for Cross-page Entity QA v1.

Checks decision/keyword correctness against the expected CSV row, plus a set of
hard safety quality gates. Deterministic. No ML, no network. Never weakens — it
only flags.
"""

from __future__ import annotations

import re

from worldpgt.cross_page_qa.types import ValidationResult

_MAX_ANSWER_CHARS = 800
_AUDIT_PREFIX = "i cannot answer from the wiki overlay because"

# Open-domain / hedging wording that must never appear.
_BANNED_PHRASES = (
    "i think", "i guess", "i believe", "probably", "maybe ", "perhaps",
    "it depends", "in general", "as an ai", "i'm not sure",
)

# Current/live factual wording that must never be asserted as fact.
_CURRENT_FACT_PHRASES = (
    "current stock price is", "stock price is", "current valuation is",
    "valuation is us$", "market cap is", "latest revenue is",
    "current price is",
)

# People who must never be claimed as having been "founded".
_PEOPLE = (
    "elon musk", "jeff bezos", "bernard arnault", "michael bloomberg",
    "gwynne shotwell", "martin eberhard", "marc tarpenning",
)
# Entities/concepts that must never be claimed to have founded a person.
_NON_PERSON_FOUNDERS = (
    "spacex", "tesla", "forbes", "bloomberg news", "bloomberg lp", "neuralink",
    "the boring company", "xai", "openai", "nasa", "blue origin", "rocket",
    "rockets", "electric vehicle", "electric vehicles", "starlink",
)


def validate(row: dict, decision: str, answer: str, intent: str) -> ValidationResult:
    expected_decision = row.get("expected_decision", "")
    expected_contains = row.get("expected_answer_contains", "")

    # Decision must match expectation.
    if decision != expected_decision:
        return ValidationResult(False, False, "decision_mismatch")

    if decision == "audit":
        # Audit answers carry the audit prefix and no factual content.
        if not answer.strip().lower().startswith(_AUDIT_PREFIX):
            return ValidationResult(True, True, "audit_missing_prefix")
        return ValidationResult(True, False, "")

    # ----- answer rows -----
    low = answer.lower()

    # Keyword containment.
    if expected_contains:
        for kw in (k.strip().lower() for k in expected_contains.split(";") if k.strip()):
            if kw not in low:
                return ValidationResult(False, False, "missing_keyword")

    # ----- hard safety gates (quality_flagged on violation) -----
    if not answer.strip():
        return ValidationResult(True, True, "empty_answer")

    for phrase in _BANNED_PHRASES:
        if phrase in low:
            return ValidationResult(True, True, f"banned_phrase:{phrase.strip()}")

    for phrase in _CURRENT_FACT_PHRASES:
        if phrase in low:
            return ValidationResult(True, True, f"current_fact:{phrase}")

    # No inverted founder claims: "<non-person> founded <person>".
    for org in _NON_PERSON_FOUNDERS:
        for person in _PEOPLE:
            if re.search(rf"\b{re.escape(org)}\s+founded\s+{re.escape(person)}\b", low):
                return ValidationResult(True, True, "inverted_founder")
    # No "<person> was founded by ...".
    for person in _PEOPLE:
        if re.search(rf"\b{re.escape(person)}\s+was\s+founded\s+by\b", low):
            return ValidationResult(True, True, "person_founded_by")

    # Weak-link answers must flag the non-factual nature (negated stable-relation
    # wording, or an explicit "cannot prove"). Both renderers always do this.
    if "weak context" in low or "weak contextual" in low:
        flagged_ok = (
            "stable factual relation" in low
            or "cannot prove" in low
            or "weak_context_only" in low
        )
        if not flagged_ok:
            return ValidationResult(True, True, "weak_link_not_flagged")

    # Source-qualified answers must flag volatility / recheck.
    if "source-qualified" in low or "source qualified" in low:
        if "recheck" not in low and "volatile" not in low:
            return ValidationResult(True, True, "source_fact_not_flagged")

    if len(answer) > _MAX_ANSWER_CHARS:
        return ValidationResult(True, True, "answer_too_long")

    return ValidationResult(True, False, "")
