"""Build a fresh, bounded Crossref DOI proposal cohort for relation density."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path

from worldpgt.benchmarks.open_book_qa.dataset import relation_id
from worldpgt.benchmarks.open_book_qa.heldout_v1 import _main_relation_ids
from worldpgt.knowledge_pump.crossref_doi_seed import (
    extract_doi_relation_rows,
    fetch_crossref_doi_records,
    select_multi_predicate_doi_rows,
)
from worldpgt.knowledge_pump.open_web_pump import BROAD_OPEN_WEB_TOPICS


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded Crossref DOI structured seed cohort")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/crossref_doi_seed_v1")
    parser.add_argument("--max-entities", type=int, default=100)
    parser.add_argument("--max-queries", type=int, default=40)
    parser.add_argument("--records-per-query", type=int, default=10)
    parser.add_argument("--request-delay-sec", type=float, default=0.5)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        raise SystemExit("refusing to fetch without --allow-network")
    if min(args.max_entities, args.max_queries, args.records_per_query) < 1:
        raise SystemExit("limits must be positive")
    user_agent = os.environ.get("MICROWORLD_CROSSREF_USER_AGENT") or os.environ.get("MICROWORLD_WIKI_USER_AGENT")
    if not user_agent:
        raise SystemExit("MICROWORLD_CROSSREF_USER_AGENT or MICROWORLD_WIKI_USER_AGENT is required")

    records, fetch_report = fetch_crossref_doi_records(
        ((topic.bucket, topic.query) for topic in BROAD_OPEN_WEB_TOPICS),
        max_queries=args.max_queries,
        records_per_query=args.records_per_query,
        request_delay_sec=args.request_delay_sec,
        user_agent=user_agent,
    )
    extracted = [
        row
        for item, bucket in records
        for row in extract_doi_relation_rows(item, topic_bucket=bucket)
    ]
    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    extracted = [row for row in extracted if relation_id(row) not in main_ids]
    rows, manifest = select_multi_predicate_doi_rows(extracted, max_entities=args.max_entities)

    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    root.joinpath("proposal_relations.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    root.joinpath("frozen_entity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "version": "crossref_doi_seed_v1",
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "source": "official Crossref Works API DOI metadata",
        "selection": "works with explicit author and publisher metadata",
        "max_entities": args.max_entities,
        "fetch": fetch_report,
        "frozen_entity_count": len(manifest),
        "proposal_relation_count": len(rows),
        "predicate_distribution": dict(sorted(Counter(row["predicate"] for row in rows).items())),
        "main_edge_overlap_count": len({relation_id(row) for row in rows} & main_ids),
    }
    root.joinpath("summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
