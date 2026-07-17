"""Pure measurement helpers for a proposal-free Wikidata density audit."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any, Iterable


# These properties describe schema, identity, provenance, presentation, or
# equivalent bookkeeping rather than an answerable relation about the item.
# External-ID, URL, and media claim datatypes are excluded separately below.
_STRUCTURAL_OR_META_PROPERTIES = frozenset({
    "P18",    # image
    "P31",    # instance of
    "P154",   # logo image
    "P279",   # subclass of
    "P373",   # Commons category
    "P460",   # said to be the same as
    "P646",   # Freebase ID
    "P910",   # topic's main category
    "P1269",  # facet of
    "P935",   # Commons gallery
    "P1343",  # described by source
    "P1424",  # topic's main template
    "P1476",  # title
    "P1889",  # different from
    "P275",   # copyright license
    "P5008",  # on focus list of Wikimedia project
    "P6216",  # copyright status
})

_NON_CONTENT_DATATYPES = frozenset({"commonsMedia", "external-id", "geo-shape", "tabular-data", "url"})

# A conservative semantic correspondence only.  A property absent from this
# table is a *potential* new type, not an extraction-ready relation.
_EXISTING_SCHEMA_PROPERTY_MAP = {
    "P50": "created_by",
    "P112": "founded_by",
    "P123": "published_by",
    "P127": "owned_by",
    "P159": "headquartered_in",
    "P176": "product_of",
    "P178": "developed_by",
    "P2283": "uses",
    "P2860": "references_work",
    "P17": "located_in",
    "P131": "located_in",
    "P276": "located_in",
    "P306": "runs_on",
    "P361": "part_of",
    "P355": "parent_company_of",
    "P366": "used_for",
    "P400": "runs_on",
    "P921": "has_topic",
    "P101": "has_topic",
    "P1056": "produces",
}


def _usable_claims(claims: object) -> list[dict[str, Any]]:
    return [
        claim for claim in (claims or [])
        if isinstance(claim, dict)
        and claim.get("rank", "normal") != "deprecated"
        and (claim.get("mainsnak") or {}).get("snaktype") == "value"
    ]


def content_property_ids(entity: dict[str, Any]) -> set[str]:
    """Return distinct non-meta, non-identifier property groups for one item."""

    result: set[str] = set()
    for property_id, claims in (entity.get("claims") or {}).items():
        if property_id in _STRUCTURAL_OR_META_PROPERTIES:
            continue
        usable = _usable_claims(claims)
        if not usable:
            continue
        datatype = str((usable[0].get("mainsnak") or {}).get("datatype") or "")
        if datatype in _NON_CONTENT_DATATYPES:
            continue
        result.add(str(property_id))
    return result


def summarize_property_density(
    resolved_subjects: Iterable[dict[str, Any]],
    entities: dict[str, dict[str, Any]],
    property_labels: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Measure content-property density without creating graph relations."""

    rows: list[dict[str, Any]] = []
    property_subject_counts: Counter[str] = Counter()
    for subject in resolved_subjects:
        qid = str(subject.get("canonical_qid") or "")
        properties = content_property_ids(entities.get(qid, {})) if qid else set()
        new_properties = properties - set(_EXISTING_SCHEMA_PROPERTY_MAP)
        for property_id in properties:
            property_subject_counts[property_id] += 1
        rows.append({
            "subject": str(subject.get("subject") or subject.get("surface_subject") or ""),
            "cohorts": sorted(subject.get("cohorts") or ()),
            "canonical_qid": qid or None,
            "resolution_status": str(subject.get("canonical_resolution_status") or ""),
            "content_property_group_count": len(properties),
            "new_schema_property_group_count": len(new_properties),
            "content_property_ids": sorted(properties),
            "new_schema_property_ids": sorted(new_properties),
        })

    # The requested density denominator is subjects with an exact QID.  The
    # unresolved surface cohort is reported separately rather than inflated
    # into a misleading "zero properties" bucket.
    resolved_rows = [row for row in rows if row["canonical_qid"]]
    counts = [row["content_property_group_count"] for row in resolved_rows]
    new_counts = [row["new_schema_property_group_count"] for row in resolved_rows]
    distribution = {
        "0": sum(count == 0 for count in counts),
        "1": sum(count == 1 for count in counts),
        "2": sum(count == 2 for count in counts),
        "3-4": sum(3 <= count <= 4 for count in counts),
        "5-9": sum(5 <= count <= 9 for count in counts),
        "10+": sum(count >= 10 for count in counts),
    }
    top = [
        {
            "property_id": property_id,
            "label": property_labels.get(property_id, property_id),
            "subject_count": count,
            "maps_to_existing_predicate": _EXISTING_SCHEMA_PROPERTY_MAP.get(property_id),
        }
        for property_id, count in sorted(property_subject_counts.items(), key=lambda item: (-item[1], item[0]))[:15]
    ]
    summary = {
        "resolved_subject_count": len(resolved_rows),
        "unresolved_subject_count": len(rows) - len(resolved_rows),
        "content_property_group_distribution_exclusive": distribution,
        "content_property_group_threshold_counts": {
            "0": distribution["0"],
            "1": distribution["1"],
            "2": distribution["2"],
            "3": sum(count >= 3 for count in counts),
            "5+": sum(count >= 5 for count in counts),
            "10+": sum(count >= 10 for count in counts),
        },
        "mean_content_property_groups": (sum(counts) / len(counts)) if counts else 0.0,
        "median_content_property_groups": median(counts) if counts else 0.0,
        "subjects_with_two_or_more_new_schema_property_groups": sum(count >= 2 for count in new_counts),
        "mean_new_schema_property_groups": (sum(new_counts) / len(new_counts)) if new_counts else 0.0,
        "top_15_content_bearing_properties": top,
        "existing_schema_property_map": dict(sorted(_EXISTING_SCHEMA_PROPERTY_MAP.items())),
        "excluded_structural_or_meta_properties": sorted(_STRUCTURAL_OR_META_PROPERTIES),
        "excluded_claim_datatypes": sorted(_NON_CONTENT_DATATYPES),
    }
    return summary, rows
