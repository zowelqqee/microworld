#!/usr/bin/env python3
"""Build the reversible, offline serving graph used by the iPhone demo.

This is deployment packaging only: it composes already approved serving
campaigns and the small precision-gated OpenAlex lane into one local JSON
overlay.  It never contacts a network service and never changes accepted or
promoted memory in the repository.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


_CAMPAIGN_FILES = (
    "campaign_extension_p12_v1/open_web_campaign_evidence_grounded_graph_overlay.json",
    "campaign_long_v2/open_web_campaign_evidence_grounded_graph_overlay.json",
    "campaign_overnight_feedback_v1/open_web_campaign_evidence_grounded_graph_overlay.json",
    "campaign_crossref_doi_v1/open_web_campaign_evidence_grounded_graph_overlay.json",
    "campaign_wikidata_seed_v1/open_web_campaign_evidence_grounded_graph_overlay.json",
)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"expected a JSON list: {path}")
    return [dict(row) for row in payload if isinstance(row, dict)]


def _read_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _relation_key(row: dict[str, Any]) -> tuple[str, str, str]:
    return tuple(_norm(row.get(field)) for field in ("subject", "predicate", "object"))


def _merge_relations(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically coalesce exact triples while retaining provenance."""
    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = _relation_key(row)
        if not all(key):
            continue
        current = merged.get(key)
        if current is None:
            merged[key] = dict(row)
            continue
        sources = {
            str(value).strip()
            for item in (current, row)
            for value in (item.get("source_url"), *(item.get("supporting_sources") or ()))
            if str(value or "").strip()
        }
        evidence: list[str] = []
        for item in (current, row):
            for value in (item.get("evidence_text"), *(item.get("supporting_evidence") or ())):
                compact = " ".join(str(value or "").split())
                if compact and compact not in evidence:
                    evidence.append(compact)
        current["support_count"] = max(1, int(current.get("support_count") or 1)) + max(
            1, int(row.get("support_count") or 1)
        )
        current["supporting_sources"] = sorted(sources)
        current["supporting_source_count"] = len(sources)
        current["supporting_evidence"] = evidence[:6]
    return [merged[key] for key in sorted(merged)]


def build(repo_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    experiments = repo_root / "worldpgt" / "experiments"
    base_path = experiments / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
    base = _read_rows(base_path)
    campaign_root = experiments / "open_web_pump_v1"
    campaign_rows: list[dict[str, Any]] = []
    campaign_counts: dict[str, int] = {}
    for relative in _CAMPAIGN_FILES:
        path = campaign_root / relative
        rows = _read_rows(path)
        campaign_counts[relative] = len(rows)
        campaign_rows.extend(row for row in rows if row.get("overlay_type") == "overlay_relation")

    openalex_path = repo_root / "artifacts" / "open_book_qa" / "openalex_seed_v1" / "precision_gate" / "accepted_proposal_overlay.json"
    openalex_rows = _read_rows(openalex_path)
    for row in openalex_rows:
        row.update({
            "experimental_tier": "evidence_grounded_structured_relation_v1",
            "trust": "user_authorized_openalex_serving",
            "serving_status": "user_authorized_experimental",
            "promotion_source": "openalex_topic_reference_precision_gate_v1",
            "experimental_query_only": True,
            "safe_for_general_runtime": False,
        })

    relations = _merge_relations([*campaign_rows, *openalex_rows])
    base_keys = {
        _relation_key(row)
        for row in base
        if row.get("overlay_type") == "overlay_relation"
    }
    added = [row for row in relations if _relation_key(row) not in base_keys]
    overlay = [*base, *added]
    source_counts = Counter(
        "openalex" if row.get("promotion_source") == "openalex_topic_reference_precision_gate_v1"
        else "crossref" if row.get("open_web_extraction") == "crossref_doi_structured_metadata_v1"
        else "wikidata" if row.get("open_web_extraction") == "wikidata_api_structured_property_v1"
        else "original_campaign"
        for row in added
    )
    cohort_counts = {
        "original_331": _read_object(
            repo_root / "artifacts" / "open_book_qa" / "wikidata_density_recon" / "summary.json"
        )["cohort_subject_counts"]["original_331"],
        "crossref_multi_evidence": _read_object(
            repo_root / "artifacts" / "open_book_qa" / "crossref_doi_seed_v1" / "precision_gate" / "summary.json"
        )["entities_with_second_relation_group_after_gate"],
        "wikidata_multi_evidence": _read_object(
            repo_root / "artifacts" / "open_book_qa" / "wikidata_seed_v1" / "precision_gate" / "summary.json"
        )["entities_with_second_relation_group_after_gate"],
        "openalex_multi_evidence": _read_object(
            repo_root / "artifacts" / "open_book_qa" / "openalex_seed_v1" / "precision_gate" / "summary.json"
        )["entities_with_second_relation_group_after_gate"],
    }
    summary = {
        "version": "ios_demo_v2_extended_serving_overlay",
        "offline_only": True,
        "accepted_memory_modified": False,
        "promoted_wiki_overlay_modified": False,
        "base_overlay_path": str(base_path.relative_to(repo_root)),
        "base_item_count": len(base),
        "campaign_input_item_counts": campaign_counts,
        "precision_gated_openalex_relation_count": len(openalex_rows),
        "added_relation_count": len(added),
        "added_relation_counts_by_lane": dict(sorted(source_counts.items())),
        "validated_multi_evidence_subject_cohorts": cohort_counts,
        "total_overlay_item_count": len(overlay),
        "total_relation_count": sum(row.get("overlay_type") == "overlay_relation" for row in overlay),
    }
    return overlay, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    args = parser.parse_args()
    overlay, summary = build(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(overlay, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
