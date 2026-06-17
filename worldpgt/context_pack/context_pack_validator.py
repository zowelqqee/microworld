"""Validator for Working Context Pack v1.

Pure, deterministic safety checks over a built pack. It does not touch the
overlay or runtime; it only asserts that a pack respects the safety invariants.
"""

from __future__ import annotations

import re

from .types import (
    STABILITY_VOLATILE,
    STABLE_STABILITIES,
    TRUST_WEAK,
    ContextPackValidationResult,
    WorkingContextPack,
)

_CURRENT_LIVE_RE = re.compile(
    r"\b(current|currently|latest|today|right now|stock price|share price|"
    r"market cap|valuation|net worth|revenue|earnings)\b",
    re.I,
)
_PRIVATE_RE = re.compile(
    r"\b(phone number|home address|private email|date of birth|favorite|"
    r"favourite|social security|passport)\b",
    re.I,
)


def validate_pack(pack: WorkingContextPack) -> ContextPackValidationResult:
    failures = []
    checks = {}

    # 1. No weak link present among direct stable relations.
    weak_in_direct = [
        r for r in pack.direct_relations
        if r.trust == TRUST_WEAK or r.stability == "weak"
    ]
    checks["no_weak_link_in_direct_relations"] = not weak_in_direct
    if weak_in_direct:
        failures.append("weak link present in direct stable relations")

    # 2. No source-qualified volatile fact marked as a stable relation.
    volatile_in_direct = [
        r for r in pack.direct_relations
        if r.stability == STABILITY_VOLATILE or r.trust.startswith("source_qualified")
    ]
    checks["no_volatile_fact_as_stable_relation"] = not volatile_in_direct
    if volatile_in_direct:
        failures.append("source-qualified volatile fact marked as stable relation")

    # 3. safe_for_answering must be False when only weak links exist (no stable
    #    candidate path and no stable direct relation).
    has_stable_support = bool(pack.candidate_paths) or any(
        r.stability in STABLE_STABILITIES and r.trust != TRUST_WEAK
        for r in pack.direct_relations
    )
    checks["weak_only_not_answerable"] = has_stable_support or not pack.safe_for_answering
    if pack.safe_for_answering and not has_stable_support:
        failures.append("safe_for_answering true without an explicit stable support path")

    # 4. Current/live questions without an explicit source fact are not answerable.
    if _CURRENT_LIVE_RE.search(pack.question) and not pack.source_facts:
        checks["current_live_not_answerable"] = not pack.safe_for_answering
        if pack.safe_for_answering:
            failures.append("current/live question answerable without source-qualified fact")
    else:
        checks["current_live_not_answerable"] = True

    # 5. Private/sensitive questions produce no stable context.
    if _PRIVATE_RE.search(pack.question):
        no_stable_ctx = not pack.direct_relations and not pack.candidate_paths and not pack.safe_for_answering
        checks["private_no_stable_context"] = no_stable_ctx
        if not no_stable_ctx:
            failures.append("private/sensitive question produced stable context")
    else:
        checks["private_no_stable_context"] = True

    # 6. Inverted-relation claims are blocked or at least safety-noted.
    inversion_blocked = any(b.reason == "relation_inversion" for b in pack.blocked_paths)
    inversion_noted = any("relation_inversion" in n for n in pack.safety_notes)
    # Only required to hold when an inversion was actually detected (recorded as
    # a block or note). This check passes vacuously otherwise.
    checks["inversion_blocked_or_noted"] = True
    if (inversion_blocked or inversion_noted) and pack.safe_for_answering:
        checks["inversion_blocked_or_noted"] = False
        failures.append("inverted relation claim marked safe_for_answering")

    # 7. safe_for_general_runtime must be False.
    checks["safe_for_general_runtime_false"] = pack.safe_for_general_runtime is False
    if pack.safe_for_general_runtime:
        failures.append("safe_for_general_runtime must be False")

    return ContextPackValidationResult(
        passed=not failures, failures=failures, checks=checks
    )
