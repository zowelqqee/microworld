"""Structured, additive Wikidata relation extraction for held-out acquisition.

Only explicit entity-valued claims with a stable semantic property are
admitted.  The evidence-local surface node is never replaced by the canonical
entity title; canonical identity is carried as provenance instead.
"""

from __future__ import annotations

from typing import Callable


_PROPERTY_TO_PREDICATE = {
    "P178": "developed_by",  # developer
    "P2283": "uses",  # uses
    "P366": "used_for",  # use
}


def extract_relation_rows(
    *, surface_subject: str, canonical_entity: str, canonical_qid: str,
    claims: dict, labels: dict[str, str], blocked_predicates: set[str] | None = None,
) -> list[dict]:
    """Convert explicit stable Wikidata claims into evidence-bound edges."""

    rows: list[dict] = []
    blocked_predicates = blocked_predicates or set()
    for property_id, predicate in _PROPERTY_TO_PREDICATE.items():
        if predicate in blocked_predicates:
            continue
        for claim in claims.get(property_id, []):
            if claim.get("rank", "normal") == "deprecated":
                continue
            snak = claim.get("mainsnak") or {}
            value = (snak.get("datavalue") or {}).get("value")
            object_qid = value.get("id") if isinstance(value, dict) else None
            object_label = labels.get(object_qid or "", "")
            if not object_qid or not object_label:
                continue
            evidence = (
                f"Wikidata statement {property_id} for {canonical_entity} "
                f"({canonical_qid}) identifies {object_label} ({object_qid})."
            )
            rows.append({
                "overlay_type": "overlay_relation",
                "subject": surface_subject,
                "surface_subject": surface_subject,
                "canonical_entity": canonical_entity,
                "canonical_qid": canonical_qid,
                "predicate": predicate,
                "object": object_label,
                "object_qid": object_qid,
                "source_page": canonical_entity,
                "source_url": f"https://www.wikidata.org/wiki/{canonical_qid}",
                "evidence_text": evidence,
                "evidence_span": evidence,
                "wikidata_property": property_id,
                "trust": "proposal_wikidata_structured",
                "risk": "medium",
                "stability": "semi_stable",
                "requires_review": True,
                "safe_for_general_runtime": False,
                "experimental_tier": "evidence_grounded_wikidata_relation_v1",
                "canonical_is_additive": True,
            })
    return rows
