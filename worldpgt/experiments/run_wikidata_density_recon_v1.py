"""Measure Wikidata claim density for the current graph; never extract relations."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from worldpgt.benchmarks.open_book_qa.dataset import _norm, load_experimental_relations
from worldpgt.benchmarks.open_book_qa.heldout_v1 import _main_relation_ids
from worldpgt.knowledge_pump.heldout_density_frontier import attach_wikidata_exact_resolution, build_density_frontier
from worldpgt.knowledge_pump.wikidata_density_recon import summarize_property_density
from worldpgt.relation_extraction_v2.types import ALLOWED_RELATIONS


_API = "https://www.wikidata.org/w/api.php"
_CROSSREF_EXTRACTION = "crossref_doi_structured_metadata_v1"


class _Client:
    def __init__(self, *, user_agent: str, delay_seconds: float) -> None:
        self.user_agent = user_agent
        self.delay_seconds = delay_seconds
        self.calls = 0

    def get(self, params: dict[str, str]) -> dict[str, Any]:
        if self.calls:
            sleep(self.delay_seconds)
        request = Request(_API + "?" + urlencode(params), headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=45) as response:  # nosec B310: official HTTPS API
            self.calls += 1
            return json.loads(response.read().decode("utf-8"))

    def search(self, label: str) -> list[dict[str, Any]]:
        payload = self.get({
            "action": "wbsearchentities", "search": label, "language": "en",
            "format": "json", "limit": "10", "type": "item",
        })
        return [item for item in payload.get("search", []) if isinstance(item, dict)]

    def entities(self, qids: list[str], *, properties: str) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for start in range(0, len(qids), 50):
            payload = self.get({
                "action": "wbgetentities", "ids": "|".join(qids[start:start + 50]),
                "props": properties, "languages": "en", "format": "json",
            })
            rows.update({
                str(qid): entity for qid, entity in (payload.get("entities") or {}).items()
                if isinstance(entity, dict)
            })
        return rows


def _original_331() -> list[dict[str, Any]]:
    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    # This provenance filter recreates the pre-Crossref 331-subject pool
    # without relying on an artifact directory name.
    relations = [
        row for row in load_experimental_relations()
        if row.get("open_web_extraction") != _CROSSREF_EXTRACTION
    ]
    frontier, summary = build_density_frontier(relations, main_ids)
    if summary["target_subject_count"] != 331:
        raise RuntimeError(f"expected the frozen original 331-subject pool, got {summary['target_subject_count']}")
    cached = json.loads(Path("artifacts/open_book_qa/targeted_wikidata_density_v2/resolution_manifest.json").read_text(encoding="utf-8"))
    by_subject = {_norm(row.get("surface_subject")): row for row in cached}
    output = []
    for row in frontier:
        previous = by_subject.get(_norm(row["surface_subject"]))
        if previous is None:
            raise RuntimeError(f"missing cached exact-resolution record for {row['surface_subject']!r}")
        output.append({
            "subject": row["surface_subject"], "surface_subject": row["surface_subject"],
            "canonical_qid": previous.get("canonical_qid"),
            "canonical_resolution_status": previous.get("canonical_resolution_status"),
            "cohorts": ["original_331"], "qid_resolution_source": "existing_exact_resolution_manifest",
        })
    return output


def _promoted_crossref_46() -> list[dict[str, Any]]:
    gate = json.loads(Path("artifacts/open_book_qa/crossref_doi_seed_v1/precision_gate/summary.json").read_text(encoding="utf-8"))
    unlocked = set(gate["unlocked_canonical_dois"])
    rows = json.loads(Path("artifacts/open_book_qa/crossref_doi_seed_v1/precision_gate/accepted_proposal_overlay.json").read_text(encoding="utf-8"))
    by_doi: dict[str, dict[str, Any]] = {}
    for row in rows:
        doi = str(row.get("canonical_doi") or "").casefold()
        if doi in unlocked:
            by_doi.setdefault(doi, row)
    if len(by_doi) != 46:
        raise RuntimeError(f"expected 46 promoted Crossref entities, got {len(by_doi)}")
    return [
        {
            "subject": row["subject"], "surface_subject": row["subject"],
            "canonical_qid": None, "canonical_resolution_status": "unresolved_not_fetched",
            "cohorts": ["crossref_promoted_46"], "qid_resolution_source": "fresh_exact_resolution",
        }
        for _doi, row in sorted(by_doi.items())
    ]


def _accepted_openalex_2() -> list[dict[str, Any]]:
    rows = json.loads(Path("artifacts/open_book_qa/openalex_seed_v1/precision_gate/accepted_proposal_overlay.json").read_text(encoding="utf-8"))
    by_subject = {_norm(row["subject"]): row for row in rows}
    if len(by_subject) != 2:
        raise RuntimeError(f"expected 2 accepted OpenAlex subjects, got {len(by_subject)}")
    return [
        {
            "subject": row["subject"], "surface_subject": row["subject"],
            "canonical_qid": None, "canonical_resolution_status": "unresolved_not_fetched",
            "cohorts": ["openalex_accepted_2"], "qid_resolution_source": "fresh_exact_resolution",
        }
        for _subject, row in sorted(by_subject.items())
    ]


def _merge_subjects(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for row in rows:
        key = _norm(row["subject"])
        current = merged.setdefault(key, dict(row))
        current["cohorts"] = sorted(set(current.get("cohorts") or ()) | set(row.get("cohorts") or ()))
    return [merged[key] for key in sorted(merged)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Measurement-only Wikidata density reconnaissance")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/wikidata_density_recon")
    parser.add_argument("--delay-seconds", type=float, default=0.25)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        raise SystemExit("refusing to fetch without --allow-network")
    if args.delay_seconds < 0:
        raise SystemExit("delay must be non-negative")
    user_agent = os.environ.get("MICROWORLD_WIKI_USER_AGENT", "")
    if not user_agent:
        raise SystemExit("MICROWORLD_WIKI_USER_AGENT is required")

    subjects = _merge_subjects([*_original_331(), *_promoted_crossref_46(), *_accepted_openalex_2()])
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    search_cache_path = root / "exact_search_cache.json"
    search_cache = {}
    if search_cache_path.is_file():
        search_cache = json.loads(search_cache_path.read_text(encoding="utf-8"))
    if not isinstance(search_cache, dict):
        raise RuntimeError("exact search checkpoint must be a JSON object")
    client = _Client(user_agent=user_agent, delay_seconds=args.delay_seconds)
    fresh = [row for row in subjects if row["qid_resolution_source"] == "fresh_exact_resolution"]
    for row in fresh:
        key = _norm(row["subject"])
        if key in search_cache:
            continue
        search_cache[key] = client.search(row["subject"])
        # The desktop command runner can impose a short wall-clock limit.
        # This checkpoint is measurement-only and makes the bounded API read
        # resumable without re-querying completed exact-label lookups.
        search_cache_path.write_text(json.dumps(search_cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    search_results = {_norm(row["subject"]): search_cache[_norm(row["subject"])] for row in fresh}
    fresh_resolved = attach_wikidata_exact_resolution(fresh, search_results)
    fresh_by_subject = {_norm(row["subject"]): row for row in fresh_resolved}
    for row in subjects:
        update = fresh_by_subject.get(_norm(row["subject"]))
        if update:
            row.update({key: update.get(key) for key in ("canonical_qid", "canonical_resolution_status")})

    qids = sorted({str(row["canonical_qid"]) for row in subjects if str(row.get("canonical_qid") or "").startswith("Q")})
    entities = client.entities(qids, properties="claims") if qids else {}
    # Labels are needed only for the measured top-15, not for all property IDs.
    provisional, _rows = summarize_property_density(subjects, entities, {})
    top_ids = [item["property_id"] for item in provisional["top_15_content_bearing_properties"]]
    property_entities = client.entities(top_ids, properties="labels") if top_ids else {}
    labels = {
        qid: str(entity.get("labels", {}).get("en", {}).get("value") or qid)
        for qid, entity in property_entities.items()
    }
    density, per_subject = summarize_property_density(subjects, entities, labels)

    status_counts = Counter(str(row.get("canonical_resolution_status") or "") for row in subjects)
    cohort_counts = Counter(cohort for row in subjects for cohort in row["cohorts"])
    resolution_by_cohort = {
        cohort: {
            "subject_count": sum(cohort in row["cohorts"] for row in subjects),
            "exact_qid_resolved": sum(
                cohort in row["cohorts"] and bool(row.get("canonical_qid")) for row in subjects
            ),
            "status_counts": dict(sorted(Counter(
                str(row.get("canonical_resolution_status") or "")
                for row in subjects if cohort in row["cohorts"]
            ).items())),
        }
        for cohort in sorted(cohort_counts)
    }
    root.joinpath("resolution_manifest.json").write_text(json.dumps(subjects, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    root.joinpath("per_subject_property_counts.json").write_text(json.dumps(per_subject, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "version": "wikidata_density_recon_v1",
        "measurement_only": True,
        "extraction_performed": False,
        "precision_gate_run": False,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "source": "official Wikidata Action API: wbsearchentities + batched wbgetentities",
        "cohort_subject_counts": dict(sorted(cohort_counts.items())),
        "unique_subject_count": len(subjects),
        "existing_exact_resolution_coverage_original_331": {
            "exact_qid_resolved": sum(bool(row.get("canonical_qid")) for row in subjects if "original_331" in row["cohorts"]),
            "status_counts": dict(sorted(Counter(
                str(row.get("canonical_resolution_status") or "")
                for row in subjects if "original_331" in row["cohorts"]
            ).items())),
        },
        "all_cohort_resolution_status_counts": dict(sorted(status_counts.items())),
        "exact_qid_resolution_by_cohort": resolution_by_cohort,
        "exact_qid_definition": "one unambiguous exact English-label Wikidata item; an English Wikipedia anchor is not required for this measurement",
        "exact_qid_subject_count": len(qids),
        "entity_records_returned": len(entities),
        "fresh_exact_search_queries": len(search_cache),
        "network_calls_this_invocation": client.calls,
        "property_density": density,
        "current_predicate_schema": sorted(ALLOWED_RELATIONS | {"has_topic", "references_work"}),
        "comparison_context": {
            "arxiv_precision_accepted_multi_predicate_entities": "0/331",
            "crossref_precision_accepted_multi_predicate_entities": "46/100",
            "openalex_precision_accepted_multi_predicate_entities": "2/6-paired",
            "wikidata_measurement": "potential content-property groups on exact-QID-resolved current subjects; not extracted or precision-validated relations",
        },
        "interpretation_boundary": "Unmapped content properties count only potential new predicate groups. They are neither candidate relations nor evidence of future precision-gate acceptance.",
    }
    root.joinpath("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
