"""Source document request builder for Candidate Knowledge Request v1.

Groups related :class:`CandidateKnowledgeRequest` objects into proposal-only
:class:`SourceDocumentRequest` objects describing local documents that could be
authored and ingested LATER. Nothing is created or ingested here.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple

from .types import (
    REQ_MISSING_SOURCE_QUALIFIED_FACT,
    CandidateKnowledgeRequest,
    SourceDocumentRequest,
)

_NEXT_STEP = "create local doc -> self_ingestion -> quarantine -> promotion -> regression"

_STANDARD_CLAIMS_TO_AVOID = [
    "Do not infer ownership or stable relations from weak contextual links",
    "Do not answer current/live metrics (valuation, revenue, net worth, stock, "
    "rankings, subscriber counts) without explicit as_of and source",
]

_STANDARD_SAFETY_NOTES = [
    "Keep current audit behavior until explicit stable evidence is ingested and promoted",
    "Route source-qualified / volatile facts through quarantine human review",
    "Do not activate proposed analyzer rules or extraction patterns",
]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).rstrip(".?! ")


def _group_key(req: CandidateKnowledgeRequest) -> Tuple[str, ...]:
    subj = _norm(req.candidate_subject)
    obj = _norm(req.candidate_object)
    if subj and obj:
        return tuple(sorted((subj, obj)))
    if subj:
        return (subj,)
    if obj:
        return (obj,)
    # No parseable entities: group by category + normalized question.
    return (req.category, _norm(req.question))


def build_source_document_requests(
    requests: List[CandidateKnowledgeRequest],
) -> List[SourceDocumentRequest]:
    groups: Dict[Tuple[str, ...], List[CandidateKnowledgeRequest]] = {}
    order: List[Tuple[str, ...]] = []
    for req in requests:
        key = _group_key(req)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(req)

    out: List[SourceDocumentRequest] = []
    for idx, key in enumerate(order, start=1):
        members = groups[key]
        rep = members[0]

        entities: List[str] = []
        for m in members:
            for ent in (m.candidate_subject, m.candidate_object):
                ent = (ent or "").strip()
                if ent and ent not in entities:
                    entities.append(ent)

        relations: List[str] = []
        for m in members:
            if m.candidate_subject and m.candidate_object:
                rel = f"{m.candidate_subject} {m.candidate_relation} {m.candidate_object}"
            else:
                rel = f"{m.candidate_relation}: {m.question}"
            if rel not in relations:
                relations.append(rel)

        claims_to_avoid = list(_STANDARD_CLAIMS_TO_AVOID)
        if any(m.category == REQ_MISSING_SOURCE_QUALIFIED_FACT for m in members):
            claims_to_avoid.append(
                "Do not promote source-qualified estimates into stable facts"
            )

        out.append(
            SourceDocumentRequest(
                id=f"sdr-{idx:04d}",
                topic=rep.suggested_source_topic,
                suggested_title=rep.suggested_source_doc_title,
                related_request_ids=[m.id for m in members],
                entities_to_cover=entities,
                relations_to_verify=relations,
                claims_to_avoid=claims_to_avoid,
                required_safety_notes=list(_STANDARD_SAFETY_NOTES),
                next_pipeline_step=_NEXT_STEP,
            )
        )
    return out
