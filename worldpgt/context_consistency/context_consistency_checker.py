"""Consistency checker for the Context-Pack QA Consistency Gate v1.

Compares one QA result against its read-only Working Context Pack and assigns a
consistency status. Pure and deterministic; it reads the pack (built/validated
by the existing context_pack package) and never mutates anything.
"""

from __future__ import annotations

from typing import List, Tuple

from worldpgt.context_pack.types import STABLE_STABILITIES, WorkingContextPack
from worldpgt.context_pack.context_pack_validator import (
    ContextPackValidationResult,
)

from .types import (
    STATUS_ANSWER_SUPPORTED,
    STATUS_ANSWER_WITHOUT_SUPPORT,
    STATUS_AUDIT_DESPITE_SAFE,
    STATUS_AUDIT_SUPPORTED,
    STATUS_CURRENT_LIVE_FALSE,
    STATUS_INVERSION_FALSE,
    STATUS_PRIVATE_FALSE,
    STATUS_SAFE_POLICY_ANSWER,
    STATUS_UNIVERSAL_FALSE,
    STATUS_VALIDATION_FAILED,
    STATUS_VOLATILE_FALSE,
    STATUS_WEAK_LINK_FALSE,
    ContextConsistencyResult,
    LoadedQAResult,
)

# Risk-flag tokens.
RISK_INVERSION = "relation_inversion"
RISK_UNIVERSAL = "unsupported_universal_claim"
RISK_PRIVATE = "private_or_sensitive"
RISK_CURRENT_LIVE = "current_or_live"
RISK_WEAK_ONLY = "weak_only_path"
RISK_VOLATILE = "volatile_source_qualified"

# Markers that identify a safe-policy answer (a safety rule, not a factual claim).
_SAFE_POLICY_MARKERS = (
    "weak context link",
    "weak_context_only",
    "not treated as a stable",
    "not a stable factual relation",
    "is not treated as a stable factual relation",
    "cannot prove",
    "does not establish",
    "does not support",
    "is not a stable fact",
    "not a stable or current fact",
    "volatile",
    "source-qualified",
    "source qualified",
    "requires recheck",
    "should be rechecked",
    "require rechecking",
    "requires rechecking",
    "time-sensitive estimate",
    "no current",
    "no live data",
    "not available",
    "contextual mention",
)


def compute_risk_flags(pack: WorkingContextPack) -> List[str]:
    flags: List[str] = []
    for b in pack.blocked_paths:
        if b.reason == RISK_INVERSION and RISK_INVERSION not in flags:
            flags.append(RISK_INVERSION)
        elif b.reason == RISK_UNIVERSAL and RISK_UNIVERSAL not in flags:
            flags.append(RISK_UNIVERSAL)
        elif b.reason == "weak_only_path" and RISK_WEAK_ONLY not in flags:
            flags.append(RISK_WEAK_ONLY)
    for n in pack.safety_notes:
        if n.startswith("private_or_sensitive") and RISK_PRIVATE not in flags:
            flags.append(RISK_PRIVATE)
        if n.startswith("current_or_live") and RISK_CURRENT_LIVE not in flags:
            flags.append(RISK_CURRENT_LIVE)
    if pack.source_facts and RISK_VOLATILE not in flags:
        if any(sf.stability == "volatile" or sf.requires_recheck for sf in pack.source_facts):
            flags.append(RISK_VOLATILE)
    return flags


def has_explicit_support(pack: WorkingContextPack) -> bool:
    """Explicit stable/semi-stable support: a candidate path, a non-weak stable
    direct relation, or a definition for a matched entity."""
    if pack.candidate_paths:
        return True
    if any(
        r.stability in STABLE_STABILITIES and r.trust != "weak_context_only"
        for r in pack.direct_relations
    ):
        return True
    if pack.definitions:
        return True
    return False


def is_safe_policy_answer(answer: str) -> bool:
    text = (answer or "").lower()
    return any(marker in text for marker in _SAFE_POLICY_MARKERS)


def _hard_block_status(flags: List[str]) -> Tuple[str, str]:
    """Map a hard-block risk flag to its false-support status + reason."""
    if RISK_INVERSION in flags:
        return STATUS_INVERSION_FALSE, "inverted relation claim treated as answer support"
    if RISK_UNIVERSAL in flags:
        return STATUS_UNIVERSAL_FALSE, "unsupported universal claim treated as answer support"
    if RISK_PRIVATE in flags:
        return STATUS_PRIVATE_FALSE, "private/sensitive question treated as answer support"
    return "", ""


def check_row(
    qa: LoadedQAResult,
    pack: WorkingContextPack,
    validation: ContextPackValidationResult,
) -> ContextConsistencyResult:
    flags = compute_risk_flags(pack)
    decision = (qa.qa_decision or "").strip().lower()

    def _result(status: str, reason: str) -> ContextConsistencyResult:
        return ContextConsistencyResult(
            id=qa.id,
            source_suite=qa.source_suite,
            question=qa.question,
            qa_decision=qa.qa_decision,
            qa_intent=qa.qa_intent,
            qa_answer=qa.qa_answer,
            qa_correct=qa.qa_correct,
            context_safe_for_answering=pack.safe_for_answering,
            context_candidate_paths_count=len(pack.candidate_paths),
            context_blocked_paths_count=len(pack.blocked_paths),
            context_weak_links_count=len(pack.weak_links),
            context_source_facts_count=len(pack.source_facts),
            context_missing_hints_count=len(pack.missing_knowledge_hints),
            validation_passed=validation.passed,
            consistency_status=status,
            failure_reason=reason,
            risk_flags=list(flags),
        )

    # Context pack itself is invalid -> fail the row regardless of QA.
    if not validation.passed:
        return _result(
            STATUS_VALIDATION_FAILED,
            "context pack failed safety validation: " + "; ".join(validation.failures),
        )

    if decision == "answer":
        return _check_answer(qa, pack, flags, _result)
    # Treat anything that is not an explicit answer (audit/reject/blank) as audit.
    return _check_audit(qa, pack, flags, _result)


def _check_answer(qa, pack, flags, _result):
    # A safe-policy explanation is allowed even when the topic is dangerous,
    # because it states a safety rule rather than asserting the claim.
    if is_safe_policy_answer(qa.qa_answer):
        return _result(
            STATUS_SAFE_POLICY_ANSWER,
            "",
        )

    # Hard blocks: an answer that is NOT a safe-policy explanation must never be
    # backed by an inverted / universal / private context.
    hb_status, hb_reason = _hard_block_status(flags)
    if hb_status:
        return _result(hb_status, hb_reason)

    # Current/live without an explicit source-qualified fact AND without explicit
    # definitional/relational support. (A stable definition of a concept such as
    # "net worth" is legitimate support and is not a live-value answer.)
    if (
        RISK_CURRENT_LIVE in flags
        and not pack.source_facts
        and not has_explicit_support(pack)
    ):
        return _result(
            STATUS_CURRENT_LIVE_FALSE,
            "current/live question answered without explicit source-qualified fact",
        )

    # Weak-only path cannot support a factual answer.
    if RISK_WEAK_ONLY in flags and not has_explicit_support(pack):
        return _result(
            STATUS_WEAK_LINK_FALSE,
            "weak-only context link treated as factual answer support",
        )

    # Volatile source-qualified fact cannot support a stable answer.
    if RISK_VOLATILE in flags and not has_explicit_support(pack):
        return _result(
            STATUS_VOLATILE_FALSE,
            "volatile source-qualified fact treated as stable answer support",
        )

    if has_explicit_support(pack):
        return _result(STATUS_ANSWER_SUPPORTED, "")

    return _result(
        STATUS_ANSWER_WITHOUT_SUPPORT,
        "QA answered but the context pack shows no explicit stable/semi-stable support",
    )


def _check_audit(qa, pack, flags, _result):
    # Audit aligned with not-safe context is the expected, consistent case.
    if not pack.safe_for_answering:
        return _result(STATUS_AUDIT_SUPPORTED, "")
    # QA audited even though the context pack looks safe. This is a warning: the
    # context pack alone is more permissive than QA (e.g. direction-agnostic
    # paths for inverted phrasings). QA being more conservative is not a safety
    # failure, so we surface it as a warning rather than failing.
    return _result(
        STATUS_AUDIT_DESPITE_SAFE,
        "QA audited though the context pack shows an explicit path; QA is more "
        "conservative (reported as warning, not weakened)",
    )
