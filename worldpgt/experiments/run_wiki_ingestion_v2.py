"""CLI for Wikipedia/Wikidata Ingestion v2 (read-only / dry-run).

Loads curated page records, extracts candidate memory items, runs the
deterministic review gate, and writes candidate/summary/review/CSV artifacts.

Everything produced is a *candidate* only. ``safe_for_runtime_memory`` is
always False. No memory file, accepted artifact, planner threshold, validator,
or generation behavior is touched, and no network access is performed.

Usage:
    python3 -m worldpgt.experiments.run_wiki_ingestion_v2 \\
      --input worldpgt/experiments/wiki_pages_curated_v2.json \\
      --output-json worldpgt/experiments/wiki_ingestion_v2_candidates.json \\
      --output-csv worldpgt/experiments/wiki_ingestion_v2_candidates.csv \\
      --summary-json worldpgt/experiments/wiki_ingestion_v2_summary.json \\
      --review-json worldpgt/experiments/wiki_ingestion_v2_review.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from worldpgt.knowledge.wiki_ingestion_v2 import WikiIngestionV2, as_dict

_CSV_FIELDS = [
    "item_type",
    "entity_id",
    "subject",
    "label",
    "predicate",
    "relation",
    "object",
    "surface",
    "target",
    "entity_type",
    "source_page",
    "source_name",
    "as_of",
    "claim_type",
    "stability",
    "strength",
    "risk",
    "status",
    "requires_recheck",
    "evidence_text",
]


def _write_csv(candidates: list, path: str) -> None:
    rows = []
    for item in candidates:
        d = as_dict(item)
        row = {field: d.get(field, "") for field in _CSV_FIELDS}
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Wikipedia/Wikidata Ingestion v2 (read-only candidate generation)"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=False, default=None)
    parser.add_argument("--summary-json", required=False, default=None)
    parser.add_argument("--review-json", required=False, default=None)
    args = parser.parse_args(argv)

    pipeline = WikiIngestionV2()
    pages = pipeline.load(args.input)
    print(f"[wiki_v2] Loaded {len(pages)} curated pages from {args.input}")

    result = pipeline.run(pages)
    s = result.summary

    print(f"[wiki_v2] Candidates total:        {s['candidates_total']}")
    print(f"[wiki_v2] entity_cards:            {s['entity_cards_count']}")
    print(f"[wiki_v2] definition_claims:       {s['definition_claims_count']}")
    print(f"[wiki_v2] relation_claims:         {s['relation_claims_count']}")
    print(f"[wiki_v2] context_links:           {s['context_links_count']}")
    print(f"[wiki_v2] source_qualified_facts:  {s['source_qualified_facts_count']}")
    print(f"[wiki_v2] volatile_claims:         {s['volatile_claims_count']}")
    print(f"[wiki_v2] duplicates_removed:      {result.duplicates_removed}")
    print(f"[wiki_v2] review errors:           {s['review_errors_count']}")
    print(f"[wiki_v2] review warnings:         {s['review_warnings_count']}")
    print(f"[wiki_v2] safe_for_runtime_memory: {s['safe_for_runtime_memory']}")

    # Candidate items.
    Path(args.output_json).write_text(
        json.dumps([as_dict(c) for c in result.candidates], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[wiki_v2] Candidates JSON written to {args.output_json}")

    if args.output_csv:
        _write_csv(result.candidates, args.output_csv)
        print(f"[wiki_v2] Candidates CSV written to {args.output_csv}")

    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[wiki_v2] Summary JSON written to {args.summary_json}")

    if args.review_json:
        review_payload = {
            "errors_count": s["review_errors_count"],
            "warnings_count": s["review_warnings_count"],
            "duplicates_removed": result.duplicates_removed,
            "issues": [as_dict(i) for i in result.review_issues],
        }
        Path(args.review_json).write_text(
            json.dumps(review_payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[wiki_v2] Review JSON written to {args.review_json}")

    print("\n[safety] sense_memory.py: NOT modified")
    print("[safety] accepted memory artifact: NOT modified")
    print("[safety] planner thresholds: NOT modified")
    print("[safety] validators: NOT weakened")
    print("[safety] generic fallback: NOT added")
    print("[safety] neural/GPT/training/embedding: NOT used")
    print("[safety] network access: NONE (local fixture only)")
    print("[safety] nanogpt/: NOT touched")
    print("[safety] safe_for_runtime_memory: FALSE — candidates only")


if __name__ == "__main__":
    main()
