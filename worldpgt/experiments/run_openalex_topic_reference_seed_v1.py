"""Create a bounded OpenAlex topic+citation diversity seed from seven DOI seeds."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from worldpgt.benchmarks.open_book_qa.dataset import _norm, load_experimental_relations, relation_id
from worldpgt.benchmarks.open_book_qa.heldout_v1 import _main_relation_ids, heldout_pool_diagnostics
from worldpgt.experiments.run_arxiv_source_specific_lane_v1 import lane_candidates
from worldpgt.knowledge_pump.openalex_topic_reference_seed import (
    doi_from_url,
    extract_topic_reference_rows,
    fetch_diverse_openalex_records,
    select_topic_reference_entities,
)


_CROSSREF_PROMOTION_EXTRACTION = "crossref_doi_structured_metadata_v1"


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded OpenAlex topic/citation predicate-diversity seed")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/openalex_seed_v1")
    parser.add_argument("--max-entities", type=int, default=7)
    parser.add_argument("--request-delay-sec", type=float, default=0.2)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        raise SystemExit("refusing to fetch without --allow-network")
    if args.max_entities < 1 or args.request_delay_sec < 0:
        raise SystemExit("max-entities must be positive and delay non-negative")
    user_agent = os.environ.get("MICROWORLD_OPENALEX_USER_AGENT") or "MicroWorldOpenAlexDiversity/1.0 (local research)"

    quarantine_rows, _existing_predicates = lane_candidates("openalex")
    seed_dois = sorted({doi_from_url(row.get("source_url")) for row in quarantine_rows if doi_from_url(row.get("source_url"))})
    works, fetch = fetch_diverse_openalex_records(
        seed_dois,
        request_delay_sec=args.request_delay_sec,
        user_agent=user_agent,
    )
    candidates = extract_topic_reference_rows(works, fetch.pop("reference_records"))
    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    candidates = [row for row in candidates if relation_id(row) not in main_ids]
    rows, manifest = select_topic_reference_entities(candidates, max_entities=args.max_entities)
    # The 331 target subjects were defined before the later Crossref promotion.
    # Rebuild exactly that baseline from the composed overlay by excluding only
    # the field-level Crossref promotion provenance, not by filename or title.
    pre_crossref_relations = [
        row for row in load_experimental_relations()
        if row.get("open_web_extraction") != _CROSSREF_PROMOTION_EXTRACTION
    ]
    _clean_groups, main_by_subject, _pool_summary = heldout_pool_diagnostics(
        pre_crossref_relations, main_ids,
    )
    main_subjects = set(main_by_subject)
    openalex_subjects = {_norm(row.get("subject")) for row in rows if _norm(row.get("subject"))}

    crossref_summary_path = Path("artifacts/open_book_qa/crossref_doi_seed_v1/precision_gate/summary.json")
    crossref_dois = set()
    if crossref_summary_path.is_file():
        crossref_dois = {str(value).casefold() for value in json.loads(crossref_summary_path.read_text(encoding="utf-8")).get("unlocked_canonical_dois", [])}
    openalex_dois = {str(row.get("canonical_doi") or "").casefold() for row in rows if row.get("canonical_doi")}

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("proposal_relations.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    root.joinpath("frozen_entity_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    summary = {
        "version": "openalex_topic_reference_seed_v1",
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "source": "official OpenAlex Works API",
        "selection": "each quarantine-seed DOI with one named highest-scored topic and one named referenced work",
        "starting_unique_quarantine_relations": len(quarantine_rows),
        "starting_unique_seed_dois": len(seed_dois),
        "max_entities": args.max_entities,
        "fetch": fetch,
        "frozen_entity_count": len(manifest),
        "proposal_relation_count": len(rows),
        "predicate_distribution": dict(sorted(Counter(row["predicate"] for row in rows).items())),
        "main_edge_overlap_count": len({relation_id(row) for row in rows} & main_ids),
        "main_target_subject_count": len(main_subjects),
        "main_subject_overlap_count": len(openalex_subjects & main_subjects),
        "main_subject_overlap_titles": sorted({row["subject"] for row in rows if _norm(row.get("subject")) in main_subjects}),
        "crossref_promoted_entity_overlap_count": len(openalex_dois & crossref_dois),
        "crossref_promoted_entity_overlap_dois": sorted(openalex_dois & crossref_dois),
    }
    root.joinpath("summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
