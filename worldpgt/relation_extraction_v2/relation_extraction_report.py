"""Report builder for Relation Extraction v2."""

from __future__ import annotations

from worldpgt.relation_extraction_v2.types import (
    ExtractedRelationCandidate,
    RelationExtractionQuarantineItem,
    RelationExtractionReport,
    RelationExtractionSummary,
)


def build_report(
    summary: RelationExtractionSummary,
    safe: list[ExtractedRelationCandidate],
    quarantine: list[RelationExtractionQuarantineItem],
    conflict_ids: list[str],
    errors: list[str],
) -> RelationExtractionReport:
    report = RelationExtractionReport(summary=summary.to_dict())

    report.top_accepted_candidates = [c.to_dict() for c in safe[:20]]
    report.top_quarantined_candidates = [q.to_dict() for q in quarantine[:20]]

    conflict_set = set(conflict_ids)
    report.conflict_examples = [
        q.to_dict() for q in quarantine
        if q.id in conflict_set
    ][:10]

    report.volatile_current_examples = [
        q.to_dict() for q in quarantine
        if "current_or_live_claim" in q.reason
    ][:10]

    # One example per relation type from safe candidates.
    by_type: dict[str, dict] = {}
    for c in safe:
        if c.relation not in by_type:
            by_type[c.relation] = c.to_dict()
    report.relation_type_examples = by_type

    report.errors = errors
    if errors:
        report.warnings.append(f"{len(errors)} doc(s) failed during extraction")
    if summary.quarantined_candidates_total > summary.accepted_relation_candidates_total * 2:
        report.warnings.append(
            "High quarantine rate. Consider tightening patterns or improving entity index."
        )

    report.recommended_next_actions = [
        "Review accepted candidates in relation_extraction_v2_candidates.json before promoting to overlay.",
        "Resolve conflicts listed in conflict_examples before ingestion.",
        "Run wiki_snapshot_ingestion pipeline again after integrating relation_claim candidates.",
        "Do NOT auto-ingest or auto-promote without manual review.",
        "Trusted memory, accepted overlay, and promoted overlay remain unchanged.",
    ]

    return report
