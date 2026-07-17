"""Validate bounded OpenAlex topic/citation proposals through unchanged gates."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from worldpgt.knowledge_pump.open_web_pump import _open_web_source_gate, build_proposal_overlay


_EXTRACTION = "openalex_api_topic_reference_structured_v1"


def validate_openalex_topic_reference_proposals(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Keep only source- and precision-accepted OpenAlex proposals.

    This function deliberately returns a proposal artifact.  It never writes
    accepted memory or a serving overlay, and uses the same source gate and
    v1/v2 precision firewalls as the Crossref lane.
    """

    candidates = [dict(row) for row in rows]
    result = build_proposal_overlay([], source_specific_candidates=candidates)
    accepted = [
        row for row in result["proposal_overlay"]
        if row.get("open_web_extraction") == _EXTRACTION
    ]
    by_entity: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        entity = str(row.get("canonical_openalex_id") or "").strip()
        predicate = str(row.get("predicate") or "").strip()
        if entity and predicate:
            by_entity[entity].add(predicate)
    input_entities = {
        str(row.get("canonical_openalex_id") or "").strip()
        for row in candidates
        if str(row.get("canonical_openalex_id") or "").strip()
    }
    source_gate_accepted = _open_web_source_gate(candidates)["accepted"]
    rejected_or_quarantined = [
        *result["rejected"], *result["quarantine"],
    ]
    reasons = Counter(
        str(entry.get("reason") or "unknown")
        for entry in rejected_or_quarantined
        if (entry.get("item") or {}).get("open_web_extraction") == _EXTRACTION
    )
    two_or_more = sorted(entity for entity, predicates in by_entity.items() if len(predicates) >= 2)
    compositions = Counter(
        "+".join(sorted(predicates))
        for entity, predicates in by_entity.items()
        if entity in two_or_more
    )
    titles_by_entity = {
        str(row.get("canonical_openalex_id") or "").strip(): str(row.get("subject") or "").strip()
        for row in accepted
    }
    multi_entities = [
        {
            "canonical_openalex_id": entity,
            "title": titles_by_entity.get(entity, ""),
            "predicate_groups": sorted(by_entity[entity]),
        }
        for entity in two_or_more
    ]
    return {
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "input_relation_count": len(candidates),
        "input_entity_count": len(input_entities),
        "passed_source_gate": len(source_gate_accepted),
        "passed_precision_gate": len(accepted),
        "rejected_or_quarantined": len(candidates) - len(accepted),
        "passed_by_predicate": dict(sorted(Counter(str(row.get("predicate") or "") for row in accepted).items())),
        "rejected_or_quarantined_by_reason": dict(sorted(reasons.items())),
        "entities_with_any_relation_after_gate": len(by_entity),
        "entities_with_two_or_more_predicate_groups_after_gate": len(two_or_more),
        "entities_with_second_relation_group_after_gate": len(two_or_more),
        "predicate_group_compositions": dict(sorted(compositions.items())),
        "unlocked_canonical_openalex_ids": two_or_more,
        "accepted_multi_predicate_entities": multi_entities,
        "accepted_proposal_overlay": accepted,
        "rejected": result["rejected"],
        "quarantine": result["quarantine"],
    }
