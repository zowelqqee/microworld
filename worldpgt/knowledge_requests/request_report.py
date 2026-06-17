"""Summary + report builder for Candidate Knowledge Request v1."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .missing_knowledge_loader import LoadedInputs
from .types import (
    CandidateKnowledgeRequest,
    KnowledgeRequestReport,
    SkippedInput,
    SourceDocumentRequest,
)


def build_summary(
    requests: List[CandidateKnowledgeRequest],
    source_docs: List[SourceDocumentRequest],
    skipped: List[SkippedInput],
    missing_inputs_total: int,
    deduplicated_count: int,
) -> Dict[str, Any]:
    by_category = Counter(r.category for r in requests)
    by_risk = Counter(r.risk_level for r in requests)
    return {
        "requests_total": len(requests),
        "source_document_requests_total": len(source_docs),
        "requests_by_category": dict(sorted(by_category.items())),
        "requests_by_risk": dict(sorted(by_risk.items())),
        "missing_inputs_total": missing_inputs_total,
        "skipped_dangerous_inputs_total": len(skipped),
        "deduplicated_count": deduplicated_count,
        "safe_for_general_runtime": False,
        "request_only": True,
        "auto_ingest": False,
        "auto_promote": False,
        "trusted_memory_modified": False,
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "network_calls": False,
    }


def build_report(
    summary: Dict[str, Any],
    loaded: LoadedInputs,
    source_docs: List[SourceDocumentRequest],
    skipped: List[SkippedInput],
) -> KnowledgeRequestReport:
    top_docs = [
        {
            "id": d.id,
            "topic": d.topic,
            "suggested_title": d.suggested_title,
            "related_request_count": len(d.related_request_ids),
            "entities_to_cover": d.entities_to_cover,
            "relations_to_verify": d.relations_to_verify,
            "claims_to_avoid": d.claims_to_avoid,
            "next_pipeline_step": d.next_pipeline_step,
        }
        for d in sorted(
            source_docs, key=lambda d: len(d.related_request_ids), reverse=True
        )[:10]
    ]

    safety_notes = [
        {
            "opportunity_id": s.opportunity_id,
            "question": s.question,
            "opportunity_category": s.opportunity_category,
            "reason": s.reason,
            "note": s.note,
        }
        for s in skipped
    ]

    next_actions = [
        "Author the proposed local source documents OFFLINE; do not fetch from "
        "the network or auto-create wiki pages.",
        "Run each new local doc through self_ingestion -> quarantine -> promotion "
        "-> regression before any answer is enabled.",
        "Keep source-qualified / volatile facts source-qualified (as_of + source); "
        "never promote them into stable facts.",
        "Keep all missing-knowledge questions auditing until explicit stable "
        "evidence exists; do not weaken safety to increase coverage.",
    ]

    return KnowledgeRequestReport(
        summary=summary,
        source_artifacts_read=list(loaded.source_artifacts_read),
        warnings=list(loaded.warnings),
        errors=list(loaded.errors),
        top_source_document_requests=top_docs,
        safety_notes=safety_notes,
        safety_confirmations={
            "request_only": True,
            "safe_for_general_runtime": False,
            "auto_ingest": False,
            "auto_promote": False,
            "auto_apply": False,
            "trusted_memory_modified": False,
            "accepted_overlay_modified": False,
            "promoted_overlay_modified": False,
            "network_calls": False,
            "rules_activated": False,
            "patterns_activated": False,
        },
        next_recommended_actions=next_actions,
    )
