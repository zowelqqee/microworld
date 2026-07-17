"""Structured, additive Wikidata relation extraction for held-out acquisition.

Only explicit entity-valued claims with a stable semantic property are
admitted.  The evidence-local surface node is never replaced by the canonical
entity title; canonical identity is carried as provenance instead.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Callable, Iterable

from worldpgt.knowledge_pump.wikidata_density_recon import _EXISTING_SCHEMA_PROPERTY_MAP, content_property_ids


_PROPERTY_TO_PREDICATE = {
    "P178": "developed_by",      # developer
    "P2283": "uses",             # uses
    "P366": "used_for",          # use
    "P112": "founded_by",        # founded by
    "P127": "owned_by",          # owned by
    "P159": "headquartered_in",  # headquarters location
    "P355": "parent_company_of", # subsidiary
    "P1056": "produces",         # product or material produced
    "P176": "product_of",        # manufacturer
    "P400": "runs_on",           # platform
    "P50": "created_by",         # author
    "P123": "published_by",      # publisher
}

# The density recon owns the wider correspondence used to distinguish
# previously-known schema from new predicate types.  Keep the historical
# relation layer map intact and add only direct, entity-valued correspondences.
_PROPERTY_TO_PREDICATE = {**_EXISTING_SCHEMA_PROPERTY_MAP, **_PROPERTY_TO_PREDICATE}


def _value_qid(claim: dict[str, Any]) -> str | None:
    value = ((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")
    return str(value.get("id")) if isinstance(value, dict) and str(value.get("id") or "").startswith("Q") else None


def _new_predicate(property_id: str, property_label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", property_label.casefold()).strip("_")
    return f"wikidata_{property_id.casefold()}_{slug or 'property'}"


def build_content_property_proposals(
    subjects: Iterable[dict[str, Any]], *, entities: dict[str, dict[str, Any]],
    labels: dict[str, str], property_labels: dict[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Extract content claims or preserve why a property was not extractable.

    A property absent from the current schema becomes a new predicate only once
    it recurs on at least three resolved subjects.  Everything else stays in
    explicit quarantine so the proposal boundary remains inspectable.
    """
    rows = [dict(row) for row in subjects]
    frequencies: Counter[str] = Counter(
        pid for row in rows for pid in content_property_ids(entities.get(str(row.get("canonical_qid") or ""), {}))
    )
    candidates: list[dict[str, Any]] = []
    quarantine: list[dict[str, Any]] = []
    for row in rows:
        qid = str(row.get("canonical_qid") or "")
        entity = entities.get(qid, {})
        subject = str(row.get("surface_subject") or row.get("subject") or "")
        canonical = str(entity.get("labels", {}).get("en", {}).get("value") or row.get("canonical_entity") or subject)
        claims = entity.get("claims") or {}
        for pid in sorted(content_property_ids(entity)):
            predicate = _PROPERTY_TO_PREDICATE.get(pid)
            if predicate is None and frequencies[pid] < 3:
                quarantine.append({"item": {"subject": subject, "canonical_qid": qid, "wikidata_property": pid, "wikidata_property_label": property_labels.get(pid, pid)}, "reason": "wikidata_property_no_schema_match"})
                continue
            predicate = predicate or _new_predicate(pid, property_labels.get(pid, pid))
            for claim in claims.get(pid, []):
                if claim.get("rank", "normal") == "deprecated":
                    continue
                object_qid = _value_qid(claim)
                if not object_qid:
                    quarantine.append({"item": {"subject": subject, "canonical_qid": qid, "wikidata_property": pid, "predicate": predicate}, "reason": "wikidata_property_non_entity_value"})
                    continue
                object_label = labels.get(object_qid, "")
                if not object_label:
                    quarantine.append({"item": {"subject": subject, "canonical_qid": qid, "wikidata_property": pid, "predicate": predicate, "object_qid": object_qid}, "reason": "wikidata_property_missing_object_label"})
                    continue
                evidence = f"Wikidata statement {pid} ({property_labels.get(pid, pid)}) for {canonical} ({qid}) identifies {object_label} ({object_qid})."
                candidates.append({
                    "overlay_type": "overlay_relation", "subject": subject, "surface_subject": subject,
                    "canonical_entity": canonical, "canonical_qid": qid, "predicate": predicate,
                    "object": object_label, "object_qid": object_qid, "source_page": canonical,
                    "source_url": f"https://www.wikidata.org/wiki/{qid}", "evidence_text": evidence,
                    "evidence_span": evidence, "wikidata_property": pid,
                    "wikidata_property_label": property_labels.get(pid, pid), "source_kind": "wikidata_api",
                    "open_web_extraction": "wikidata_api_structured_property_v1", "trust": "proposal_wikidata_structured",
                    "risk": "medium", "stability": "semi_stable", "requires_review": True,
                    "safe_for_general_runtime": False, "experimental_tier": "evidence_grounded_wikidata_relation_v1",
                    "canonical_is_additive": True,
                })
    return candidates, quarantine, dict(sorted(frequencies.items()))


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
