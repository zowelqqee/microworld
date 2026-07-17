"""Validate Crossref DOI proposals through the unchanged open-web gates."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from worldpgt.knowledge_pump.open_web_pump import _open_web_source_gate, build_proposal_overlay


_EXTRACTION = "crossref_doi_structured_metadata_v1"


def validate_crossref_doi_proposals(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Run source, v1, and v2 precision gates without promoting memory.

    The input is the bounded Crossref DOI proposal artifact.  The returned
    accepted list is still a proposal overlay; callers must not treat it as an
    accepted-memory or serving-overlay mutation.
    """

    candidates = [dict(row) for row in rows]
    result = build_proposal_overlay([], source_specific_candidates=candidates)
    accepted = [
        row for row in result["proposal_overlay"]
        if row.get("open_web_extraction") == _EXTRACTION
    ]
    by_doi: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        doi = str(row.get("canonical_doi") or "").casefold().strip()
        predicate = str(row.get("predicate") or "")
        if doi and predicate:
            by_doi[doi].add(predicate)
    input_dois = {
        str(row.get("canonical_doi") or "").casefold().strip()
        for row in candidates
        if str(row.get("canonical_doi") or "").strip()
    }
    source_gate_accepted = _open_web_source_gate(candidates)["accepted"]
    reasons = Counter(
        str(entry.get("reason") or "unknown")
        for entry in [*result["rejected"], *result["quarantine"]]
        if (entry.get("item") or {}).get("open_web_extraction") == _EXTRACTION
    )
    two_or_more = sorted(doi for doi, predicates in by_doi.items() if len(predicates) >= 2)
    return {
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "input_relation_count": len(candidates),
        "input_entity_count": len(input_dois),
        "passed_source_gate": len(source_gate_accepted),
        "passed_precision_gate": len(accepted),
        "rejected_or_quarantined": len(candidates) - len(accepted),
        "passed_by_predicate": dict(sorted(Counter(str(row.get("predicate") or "") for row in accepted).items())),
        "rejected_or_quarantined_by_reason": dict(sorted(reasons.items())),
        "entities_with_any_relation_after_gate": len(by_doi),
        "entities_with_two_or_more_predicate_groups_after_gate": len(two_or_more),
        "entities_with_second_relation_group_after_gate": len(two_or_more),
        "unlocked_canonical_dois": two_or_more,
        "accepted_proposal_overlay": accepted,
        "rejected": result["rejected"],
        "quarantine": result["quarantine"],
    }
