"""Opportunity classifier for the Self-Learning Loop v1.

Maps mined :class:`LearningSignal` objects into :class:`LearningOpportunity`
proposals, and derives pattern / analyzer-rule proposals. Every output is a
proposal: nothing here writes memory, activates a rule, or promotes anything.

Safety invariants enforced here:

- Unsafe claims (private, inverted, weak-link-as-fact, current/live, universal)
  are NEVER classified as ``missing_stable_knowledge`` or as safe knowledge.
- No opportunity proposes a direct accepted-memory write.
- All pattern and analyzer-rule proposals carry ``activation_status =
  proposal_only``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .types import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    PROPOSAL_ONLY,
    REQUIRES_HUMAN_REVIEW,
    REQUIRES_INGESTION,
    REQUIRES_REGRESSION_GATE,
    CAT_ADVERSARIAL_TEST_CANDIDATE,
    CAT_ANALYZER_RULE_CANDIDATE,
    CAT_CURRENT_CLAIM_GUARD_CASE,
    CAT_EXTRACTION_PATTERN_CANDIDATE,
    CAT_MISSING_STABLE_KNOWLEDGE,
    CAT_RELATION_INVERSION_GUARD_CASE,
    CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE,
    CAT_VOLATILE_REVIEW_CANDIDATE,
    CAT_WEAK_LINK_SAFETY_CASE,
    AnalyzerRuleProposal,
    LearningOpportunity,
    LearningSignal,
    PatternProposal,
)

# --------------------------------------------------------------------------- #
# Quarantine reason -> classification.
# (category, priority, expected_policy, activation_status, suggested_next_step)
# --------------------------------------------------------------------------- #
_QUARANTINE_REASON_MAP: Dict[str, Tuple[str, str, str, str, str]] = {
    "current_fact_without_as_of": (
        CAT_CURRENT_CLAIM_GUARD_CASE,
        PRIORITY_HIGH,
        "must_audit_or_quarantine_current_claim_without_as_of",
        REQUIRES_REGRESSION_GATE,
        "add as a current-claim guard regression case expecting audit; never answer live claims",
    ),
    "weak_link_promoted_to_fact": (
        CAT_WEAK_LINK_SAFETY_CASE,
        PRIORITY_HIGH,
        "keep_weak_context_links_weak_never_treat_as_stable_fact",
        REQUIRES_REGRESSION_GATE,
        "add adversarial regression rows; keep weak links weak; do not promote into a stable relation",
    ),
    "inverted_relation": (
        CAT_RELATION_INVERSION_GUARD_CASE,
        PRIORITY_HIGH,
        "audit_inverted_relation_never_answer",
        REQUIRES_REGRESSION_GATE,
        "add a relation-inversion guard regression case expecting audit",
    ),
    "private_or_sensitive_data": (
        CAT_ADVERSARIAL_TEST_CANDIDATE,
        PRIORITY_HIGH,
        "reject_private_or_sensitive_data",
        REQUIRES_REGRESSION_GATE,
        "add a high-priority adversarial regression case expecting reject; never answer private data",
    ),
    "unsupported_universal_claim": (
        CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE,
        PRIORITY_MEDIUM,
        "audit_unsupported_universal_reversal",
        REQUIRES_REGRESSION_GATE,
        "add an unsupported-universal guard regression case expecting audit",
    ),
    "volatile_requires_source": (
        CAT_VOLATILE_REVIEW_CANDIDATE,
        PRIORITY_MEDIUM,
        "route_to_human_review_not_auto_accept",
        REQUIRES_HUMAN_REVIEW,
        "hold volatile source/as_of fact for human review; never auto-accept as stable",
    ),
    "entity_type_mismatch": (
        CAT_VOLATILE_REVIEW_CANDIDATE,
        PRIORITY_MEDIUM,
        "route_conflict_to_human_review_not_auto_accept",
        REQUIRES_HUMAN_REVIEW,
        "hold conflicting entity-type item for human review; never auto-accept",
    ),
    "conflicts_existing_fact": (
        CAT_VOLATILE_REVIEW_CANDIDATE,
        PRIORITY_MEDIUM,
        "route_conflict_to_human_review_not_auto_accept",
        REQUIRES_HUMAN_REVIEW,
        "hold conflicting item for human review; never auto-accept",
    ),
}

# Danger screen: if a free-text question/claim looks unsafe, force it into the
# right guard category instead of ever calling it missing stable knowledge.
# (category, priority, activation_status)
_DANGER_PATTERNS: List[Tuple[re.Pattern, Tuple[str, str, str]]] = [
    (
        re.compile(r"\bprivate\s+email\b|\b[\w.+-]+@[\w-]+\.\w+", re.I),
        (CAT_ADVERSARIAL_TEST_CANDIDATE, PRIORITY_HIGH, REQUIRES_REGRESSION_GATE),
    ),
    (
        re.compile(r"\b(did|is)\s+\w+\s+(found|the founder of)\b", re.I),
        (CAT_RELATION_INVERSION_GUARD_CASE, PRIORITY_HIGH, REQUIRES_REGRESSION_GATE),
    ),
    (
        re.compile(r"\b(all|every)\b.*\b(are|is)\b.*\bproducts?\b|\bare all\b", re.I),
        (
            CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE,
            PRIORITY_MEDIUM,
            REQUIRES_REGRESSION_GATE,
        ),
    ),
    (
        re.compile(
            r"\bstock price\b|\brichest\b|\bmarket cap\b|\btoday\b|\bright now\b|\blatest revenue\b",
            re.I,
        ),
        (CAT_CURRENT_CLAIM_GUARD_CASE, PRIORITY_HIGH, REQUIRES_REGRESSION_GATE),
    ),
    (
        re.compile(r"\bweak link\b.*\b(prove|owned)\b", re.I),
        (CAT_WEAK_LINK_SAFETY_CASE, PRIORITY_HIGH, REQUIRES_REGRESSION_GATE),
    ),
]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".?! ")


def _danger_category(text: str) -> Optional[Tuple[str, str, str]]:
    for pattern, classification in _DANGER_PATTERNS:
        if pattern.search(text or ""):
            return classification
    return None


def classify_signals(signals: List[LearningSignal]) -> List[LearningOpportunity]:
    opportunities: List[LearningOpportunity] = []
    seen: set = set()
    counter = {"n": 0}

    def _emit(opp: LearningOpportunity) -> None:
        key = (opp.category, _normalize(opp.question_or_claim))
        if key in seen:
            return
        seen.add(key)
        opportunities.append(opp)

    def _new_id() -> str:
        counter["n"] += 1
        return f"slo-{counter['n']:04d}"

    for sig in signals:
        if sig.signal_type == "quarantine_item":
            opp = _from_quarantine(sig, _new_id)
            if opp is not None:
                _emit(opp)
        elif sig.signal_type == "qa_audit":
            opp = _from_qa_audit(sig, _new_id)
            if opp is not None:
                _emit(opp)

    return opportunities


def _from_quarantine(
    sig: LearningSignal, new_id
) -> Optional[LearningOpportunity]:
    reason = (sig.reason or "").strip()
    mapping = _QUARANTINE_REASON_MAP.get(reason)
    if mapping is None:
        # Unknown reason: never treat as safe knowledge; hold for human review.
        category, priority, policy, status, step = (
            CAT_VOLATILE_REVIEW_CANDIDATE,
            PRIORITY_MEDIUM,
            "route_to_human_review_not_auto_accept",
            REQUIRES_HUMAN_REVIEW,
            "hold quarantined item for human review; never auto-accept",
        )
    else:
        category, priority, policy, status, step = mapping
    return LearningOpportunity(
        id=new_id(),
        category=category,
        priority=priority,
        source_artifact=sig.source_artifact,
        source_row_id=sig.source_row_id,
        question_or_claim=sig.text,
        observed_decision=sig.observed_decision,
        expected_policy=policy,
        reason=f"quarantined: {reason or 'unknown_reason'} (risk={sig.risk})",
        suggested_next_step=step,
        activation_status=status,
    )


def _from_qa_audit(sig: LearningSignal, new_id) -> Optional[LearningOpportunity]:
    text = sig.text or ""
    danger = _danger_category(text)
    if danger is not None:
        # A dangerous-looking audited question is a guard case, never missing
        # stable knowledge.
        category, priority, status = danger
        return LearningOpportunity(
            id=new_id(),
            category=category,
            priority=priority,
            source_artifact=sig.source_artifact,
            source_row_id=sig.source_row_id,
            question_or_claim=text,
            observed_decision=sig.observed_decision,
            expected_policy="audit_or_reject_unsafe_query",
            reason="audited QA query matched a safety guard pattern",
            suggested_next_step="add as a safety regression case; do not answer",
            activation_status=status,
        )
    # Benign audited question: the system correctly audits, but a stable fact /
    # local source document may be missing.
    return LearningOpportunity(
        id=new_id(),
        category=CAT_MISSING_STABLE_KNOWLEDGE,
        priority=PRIORITY_MEDIUM,
        source_artifact=sig.source_artifact,
        source_row_id=sig.source_row_id,
        question_or_claim=text,
        observed_decision=sig.observed_decision,
        expected_policy="audit_until_local_source_added",
        reason="audit caused by missing explicit stable path",
        suggested_next_step=(
            "add local source document or explicit relation candidate, then route "
            "through ingestion/quarantine/promotion"
        ),
        activation_status=REQUIRES_INGESTION,
    )


# --------------------------------------------------------------------------- #
# Pattern proposals (extraction patterns).
# --------------------------------------------------------------------------- #
_RELATION_PATTERN_TEMPLATES: Dict[str, str] = {
    "produces": "{ORG} produces {PRODUCT}",
    "develops": "{ORG} develops {PRODUCT}",
    "makes": "{ORG} makes {PRODUCT}",
    "leader_of": "{PERSON} is the leader of {ORG}",
    "founded": "{PERSON} founded {ORG}",
    "known_for": "{PERSON} is known for {ORG}",
}


def propose_extraction_patterns(
    signals: List[LearningSignal],
) -> List[PatternProposal]:
    out: List[PatternProposal] = []
    seen: set = set()
    for sig in signals:
        if sig.signal_type != "overlay_relation":
            continue
        relation = (sig.text or "").strip()
        if not relation or relation in seen:
            continue
        seen.add(relation)
        pattern = _RELATION_PATTERN_TEMPLATES.get(
            relation, "{SUBJECT} " + relation + " {OBJECT}"
        )
        out.append(
            PatternProposal(
                proposal_type=CAT_EXTRACTION_PATTERN_CANDIDATE,
                pattern=pattern,
                relation=relation,
                risk=sig.risk or "medium",
                requires_tests=True,
                activation_status=PROPOSAL_ONLY,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# Analyzer-rule proposals.
# --------------------------------------------------------------------------- #
# guard category -> (intent, surface_patterns, expected_decision)
_ANALYZER_RULE_SPECS: Dict[str, Tuple[str, List[str], str]] = {
    CAT_RELATION_INVERSION_GUARD_CASE: (
        "relation_inversion_guard",
        ["Did {ORG} found {PERSON}?", "Is {ORG} the founder of {PERSON}?"],
        "audit",
    ),
    CAT_UNSUPPORTED_UNIVERSAL_GUARD_CASE: (
        "unsupported_universal_guard",
        ["If {ORG} makes {PRODUCT}, are all {PRODUCT} {ORG} products?"],
        "audit",
    ),
    CAT_CURRENT_CLAIM_GUARD_CASE: (
        "current_claim_guard",
        [
            "What is {ENTITY}'s stock price today?",
            "Who is the richest person right now?",
        ],
        "audit",
    ),
    CAT_ADVERSARIAL_TEST_CANDIDATE: (
        "private_data_guard",
        ["What is {PERSON}'s private email?"],
        "audit",
    ),
    CAT_WEAK_LINK_SAFETY_CASE: (
        "weak_link_guard",
        ["Does a weak link prove {ORG} is owned by {PERSON}?"],
        "audit",
    ),
}


def propose_analyzer_rules(
    opportunities: List[LearningOpportunity],
) -> List[AnalyzerRuleProposal]:
    out: List[AnalyzerRuleProposal] = []
    seen: set = set()
    by_category: Dict[str, str] = {}
    for opp in opportunities:
        by_category.setdefault(opp.category, opp.id)
    for category, source_opp_id in by_category.items():
        spec = _ANALYZER_RULE_SPECS.get(category)
        if spec is None or category in seen:
            continue
        seen.add(category)
        intent, surface_patterns, expected_decision = spec
        out.append(
            AnalyzerRuleProposal(
                proposal_type=CAT_ANALYZER_RULE_CANDIDATE,
                intent=intent,
                surface_patterns=list(surface_patterns),
                expected_decision=expected_decision,
                activation_status=PROPOSAL_ONLY,
                source_opportunity_id=source_opp_id,
            )
        )
    return out
