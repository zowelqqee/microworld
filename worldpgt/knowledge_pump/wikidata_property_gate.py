"""Proposal-only source and precision gate for structured Wikidata claims."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

from worldpgt.knowledge_pump.open_web_pump import _open_web_source_gate, build_proposal_overlay


_EXTRACTION = "wikidata_api_structured_property_v1"


def validate_wikidata_property_proposals(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = [dict(row) for row in rows]
    result = build_proposal_overlay([], source_specific_candidates=candidates)
    accepted = [row for row in result["proposal_overlay"] if row.get("open_web_extraction") == _EXTRACTION]
    by_qid: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        if row.get("canonical_qid") and row.get("predicate"):
            by_qid[str(row["canonical_qid"])].add(str(row["predicate"]))
    source_accepted = _open_web_source_gate(candidates)["accepted"]
    rejected_or_quarantined = [*result["rejected"], *result["quarantine"]]
    reasons = Counter(str(item.get("reason") or "unknown") for item in rejected_or_quarantined)
    multi = sorted(qid for qid, predicates in by_qid.items() if len(predicates) >= 2)
    compositions = Counter("+".join(sorted(by_qid[qid])) for qid in multi)
    return {
        "proposal_only": True, "accepted_memory_modified": False, "serving_overlay_modified": False,
        "input_relation_count": len(candidates), "input_entity_count": len({str(r.get("canonical_qid") or "") for r in candidates if r.get("canonical_qid")}),
        "passed_source_gate": len(source_accepted), "passed_precision_gate": len(accepted),
        "rejected_or_quarantined": len(candidates) - len(accepted),
        "passed_by_predicate": dict(sorted(Counter(str(row.get("predicate") or "") for row in accepted).items())),
        "rejected_or_quarantined_by_reason": dict(sorted(reasons.items())),
        "entities_with_any_relation_after_gate": len(by_qid),
        "entities_with_two_or_more_predicate_groups_after_gate": len(multi),
        "entities_with_second_relation_group_after_gate": len(multi),
        "predicate_group_compositions": dict(sorted(compositions.items())),
        "unlocked_canonical_qids": multi, "accepted_proposal_overlay": accepted,
        "rejected": result["rejected"], "quarantine": result["quarantine"],
    }
