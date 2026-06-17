"""Request classifier for Candidate Knowledge Request v1.

Maps missing-knowledge opportunities into :class:`CandidateKnowledgeRequest`
proposals, applying its own safety screening so that unsafe content never
becomes a stable knowledge request:

- Private / personal data questions are skipped (safety note).
- Relation-inversion and unsupported-universal questions are skipped (safety
  note) — they must keep auditing, not gain a stable fact.
- Current / live / source-qualified questions become HIGH-risk
  ``missing_source_qualified_fact`` requests that require human review and
  source qualification, and must remain audited until supported.
- Genuine entity / product / organization relation gaps become medium-risk
  relation requests that must not be inferred from weak context links.

Every request is request-only and carries an ``activation_status`` from
``ALLOWED_ACTIVATION_STATUSES`` (no auto_apply / auto_ingest / auto_promote).
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from .types import (
    DANGEROUS_OPPORTUNITY_CATEGORIES,
    REQUIRES_LOCAL_SOURCE_DOC,
    REQUIRES_QUARANTINE_REVIEW,
    REQUIRES_REGRESSION_GATE,
    REQ_MISSING_ENTITY_RELATION,
    REQ_MISSING_ORGANIZATION_RELATION,
    REQ_MISSING_POLICY_EXPLANATION,
    REQ_MISSING_PRODUCT_RELATION,
    REQ_MISSING_SOURCE_QUALIFIED_FACT,
    REQ_UNSUPPORTED_OR_TOO_FAR_PATH,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
    CandidateKnowledgeRequest,
    MissingKnowledgeInput,
    SkippedInput,
)

# --------------------------------------------------------------------------- #
# Content screening patterns.
# --------------------------------------------------------------------------- #
_PRIVATE_RE = re.compile(
    r"\b(phone number|home address|private email|date of birth|favorite|favourite|"
    r"social security|passport)\b",
    re.I,
)
_INVERSION_RE = re.compile(
    r"(lead|leads|led|found|founded|founder of)\s+(elon\s+musk|musk)\b"
    r"|does\s+tesla\s+lead\s+(elon\s+)?musk"
    r"|does\s+\w+\s+lead\s+(elon\s+)?musk",
    re.I,
)
_UNIVERSAL_RE = re.compile(
    r"\b(are all|is every|every electric car|all rockets|all .* are)\b", re.I
)
_SOURCE_QUALIFIED_RE = re.compile(
    r"\b(net worth|valuation|stock|share price|market cap|revenue|earnings|"
    r"ranking|richest|subscriber|current|currently|latest|today|right now|"
    r"as of now|price)\b",
    re.I,
)
_POLICY_RE = re.compile(
    r"^is\s+\w+.*\b(a company|a product|a stable physical property|the same as)\b"
    r"|\bprove\b",
    re.I,
)

# Object-type keyword sets for relation classification.
_PRODUCT_KEYWORDS = (
    "rocket",
    "rockets",
    "battery",
    "batteries",
    "lithium-ion",
    "electric car",
    "electric cars",
    "satellite",
    "starlink",
    "roadster",
    "falcon",
    "crew dragon",
    "gigafactory",
    "vehicle",
    "spacecraft",
)
_ORG_KEYWORDS = (
    "nasa",
    "forbes",
    "bloomberg",
    "reuters",
    "spacex",
    "tesla",
    "oracle",
    "neuralink",
    "boring company",
    "xai",
    "company",
    "news",
    "l.p.",
    "lp",
)

_CONNECT_RE = re.compile(r"how is (.+?) connected to (.+?)[\?\.]*\s*$", re.I)
_PATH_RE = re.compile(r"what path connects (.+?) to (.+?)[\?\.]*\s*$", re.I)

_WEAK_LINK_CONSTRAINT = "do_not_infer_stable_relation_from_weak_context_links"
_AUDIT_CONSTRAINT = "keep_current_audit_behavior_until_explicit_stable_evidence"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".?! ")


def _slug(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", (text or "").lower()).strip("_")
    return s or "topic"


def _parse_pair(question: str) -> Tuple[str, str]:
    for rx in (_CONNECT_RE, _PATH_RE):
        m = rx.search(question or "")
        if m:
            return m.group(1).strip(), m.group(2).strip()
    return "", ""


def _object_type(text: str) -> str:
    low = (text or "").lower()
    for kw in _PRODUCT_KEYWORDS:
        if kw in low:
            return "product"
    for kw in _ORG_KEYWORDS:
        if kw in low:
            return "org"
    return "entity"


def classify_requests(
    inputs: List[MissingKnowledgeInput],
) -> Tuple[List[CandidateKnowledgeRequest], List[SkippedInput]]:
    requests: List[CandidateKnowledgeRequest] = []
    skipped: List[SkippedInput] = []
    seen: set = set()
    counter = {"n": 0}

    def _next_id() -> str:
        counter["n"] += 1
        return f"ckr-{counter['n']:04d}"

    for inp in inputs:
        # Defense in depth: a dangerous opportunity category never becomes a
        # stable knowledge request.
        if inp.category in DANGEROUS_OPPORTUNITY_CATEGORIES:
            skipped.append(
                SkippedInput(
                    opportunity_id=inp.id,
                    question=inp.question,
                    opportunity_category=inp.category,
                    reason="dangerous_opportunity_category",
                    note="dangerous category routed to safety notes; never a stable request",
                )
            )
            continue

        built = _classify_one(inp, _next_id)
        if isinstance(built, SkippedInput):
            skipped.append(built)
            continue

        # Deduplicate by normalized question + candidate relation/path.
        key = (
            _normalize(built.question),
            _normalize(built.candidate_relation),
            _normalize(built.suggested_relation_path),
        )
        if key in seen:
            counter["n"] -= 1  # reclaim the id for a real next request
            continue
        seen.add(key)
        requests.append(built)

    return requests, skipped


def _classify_one(inp: MissingKnowledgeInput, next_id) -> object:
    q = inp.question or ""

    # 1. Private / personal data -> skip, never request a doc with private data.
    if _PRIVATE_RE.search(q):
        return SkippedInput(
            opportunity_id=inp.id,
            question=q,
            opportunity_category=inp.category,
            reason="private_or_sensitive_data",
            note="private/personal data must not become a knowledge request",
        )

    # 2. Relation inversion -> skip, keep auditing.
    if _INVERSION_RE.search(q):
        return SkippedInput(
            opportunity_id=inp.id,
            question=q,
            opportunity_category=inp.category,
            reason="relation_inversion",
            note="inverted relation must keep auditing; no stable request",
        )

    # 3. Unsupported universal -> skip, keep auditing.
    if _UNIVERSAL_RE.search(q):
        return SkippedInput(
            opportunity_id=inp.id,
            question=q,
            opportunity_category=inp.category,
            reason="unsupported_universal",
            note="universal reversal must keep auditing; no stable request",
        )

    subject, obj = _parse_pair(q)

    # 4. Source-qualified / current-ish -> HIGH risk, requires review + source.
    if _SOURCE_QUALIFIED_RE.search(q):
        return _build(
            inp,
            next_id(),
            category=REQ_MISSING_SOURCE_QUALIFIED_FACT,
            subject=subject,
            obj=obj,
            relation="source_qualified_fact",
            needed_evidence="source_qualified_fact_with_as_of_and_source",
            risk=RISK_HIGH,
            activation=REQUIRES_QUARANTINE_REVIEW,
            extra_constraints=[
                "require_explicit_as_of_and_source",
                "never_render_as_stable_or_current_fact",
                "human_review_required",
            ],
        )

    # 5. Policy / definitional probe -> policy explanation request (low risk).
    if _POLICY_RE.search(q):
        return _build(
            inp,
            next_id(),
            category=REQ_MISSING_POLICY_EXPLANATION,
            subject=subject,
            obj=obj,
            relation="policy_explanation",
            needed_evidence="policy_documentation_and_regression_tests",
            risk=RISK_LOW,
            activation=REQUIRES_REGRESSION_GATE,
            extra_constraints=["explain_safety_policy_not_world_fact"],
        )

    # 6-8. Connection / path relation gaps.
    if subject and obj:
        otype = _object_type(obj)
        if otype == "product":
            category = REQ_MISSING_PRODUCT_RELATION
            relation = "develops/produces/operates"
            evidence = "ORG -> develops/produces/operates -> PRODUCT"
        elif otype == "org":
            category = REQ_MISSING_ORGANIZATION_RELATION
            relation = "founder_of/leader_of/owned_by/subsidiary_of/operates/publishes"
            evidence = "explicit_org_relation (founder_of/leader_of/owned_by/subsidiary_of/operates/publishes)"
        else:
            category = REQ_MISSING_ENTITY_RELATION
            relation = "connected_to"
            evidence = "explicit_stable_relation_or_path"
        return _build(
            inp,
            next_id(),
            category=category,
            subject=subject,
            obj=obj,
            relation=relation,
            needed_evidence=evidence,
            risk=RISK_MEDIUM,
            activation=REQUIRES_LOCAL_SOURCE_DOC,
            extra_constraints=["require_explicit_stable_relation_before_answer"],
        )

    # 9. Fallback: an entity-style gap with no parseable pair.
    return _build(
        inp,
        next_id(),
        category=REQ_MISSING_ENTITY_RELATION,
        subject=subject,
        obj=obj,
        relation="connected_to",
        needed_evidence="explicit_stable_relation_or_path",
        risk=RISK_MEDIUM,
        activation=REQUIRES_LOCAL_SOURCE_DOC,
        extra_constraints=["require_explicit_stable_relation_before_answer"],
    )


def _build(
    inp: MissingKnowledgeInput,
    req_id: str,
    *,
    category: str,
    subject: str,
    obj: str,
    relation: str,
    needed_evidence: str,
    risk: str,
    activation: str,
    extra_constraints: Optional[List[str]] = None,
) -> CandidateKnowledgeRequest:
    constraints = [_WEAK_LINK_CONSTRAINT, _AUDIT_CONSTRAINT]
    constraints.extend(extra_constraints or [])

    if subject and obj:
        topic = f"{subject} and {obj} relationship"
        path = f"{subject} -> ? -> {obj}"
    elif subject:
        topic = f"{subject} {relation}".strip()
        path = f"{subject} -> ? -> {obj or 'target'}"
    else:
        topic = f"missing knowledge: {inp.question}"
        path = ""

    return CandidateKnowledgeRequest(
        id=req_id,
        source_opportunity_id=inp.id,
        question=inp.question,
        current_decision=inp.current_decision,
        current_behavior=inp.current_behavior,
        missing_knowledge_type=category,
        needed_evidence_type=needed_evidence,
        candidate_subject=subject,
        candidate_relation=relation,
        candidate_object=obj,
        suggested_relation_path=path,
        suggested_source_topic=topic,
        suggested_source_doc_title=f"{_slug(topic)}_local_doc",
        risk_level=risk,
        safety_constraints=constraints,
        must_remain_audit_until_supported=True,
        activation_status=activation,
        category=category,
    )
