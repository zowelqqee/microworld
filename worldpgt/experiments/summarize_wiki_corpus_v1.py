"""Print a compact corpus-scale summary for the curated wiki corpus.

Read-only. Loads the curated pages, ingestion candidates, and overlay summary
artifacts and prints scale / safety / split statistics. No network access, no
modification of any memory artifact.

Usage::

    python3 -m worldpgt.experiments.summarize_wiki_corpus_v1 \\
      --pages worldpgt/experiments/wiki_pages_curated_v2.json \\
      --candidates worldpgt/experiments/wiki_ingestion_v2_candidates.json \\
      --ingestion-summary worldpgt/experiments/wiki_ingestion_v2_summary.json \\
      --overlay-summary worldpgt/experiments/accepted_wiki_memory_overlay_v1_summary.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load(path: str) -> object:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize the curated wiki corpus")
    parser.add_argument("--pages", default="worldpgt/experiments/wiki_pages_curated_v2.json")
    parser.add_argument("--candidates", default="worldpgt/experiments/wiki_ingestion_v2_candidates.json")
    parser.add_argument("--ingestion-summary", default="worldpgt/experiments/wiki_ingestion_v2_summary.json")
    parser.add_argument("--overlay-summary", default="worldpgt/experiments/accepted_wiki_memory_overlay_v1_summary.json")
    args = parser.parse_args(argv)

    pages = _load(args.pages)
    candidates = _load(args.candidates)
    ing = _load(args.ingestion_summary)
    ov = _load(args.overlay_summary)

    by_candidate_type = Counter(c.get("item_type", "?") for c in candidates)
    source_facts = [c for c in candidates if c.get("item_type") == "source_qualified_fact"]
    weak_links = [c for c in candidates if c.get("item_type") == "context_link"]
    volatile = [c for c in candidates if c.get("stability") == "volatile"]

    print("=== Curated Wiki Corpus Summary ===")
    print(f"pages_total:            {len(pages)}")
    print(f"candidates_total:       {len(candidates)}")
    print(f"overlay_items_total:    {ov.get('overlay_items_total')}")
    print(f"skipped_candidates:     {ov.get('skipped_candidates_total')}")
    print()
    print(f"by_candidate_type:      {dict(sorted(by_candidate_type.items()))}")
    print(f"by_overlay_type:        {ov.get('by_overlay_type')}")
    print()
    print(f"source_facts_count:     {len(source_facts)}")
    print(f"weak_context_links:     {len(weak_links)}")
    print(f"volatile_count:         {len(volatile)}")
    print(f"stability_split:        {ing.get('by_stability')}")
    print(f"risk_split:             {ing.get('by_risk')}")
    print()
    print(f"review_errors:          {ing.get('review_errors_count')}")
    print(f"review_warnings:        {ing.get('review_warnings_count')}")
    print(f"safe_for_runtime_memory:    {ing.get('safe_for_runtime_memory')}")
    print(f"safe_for_general_runtime:   {ov.get('safe_for_general_runtime')}")
    print(f"safe_for_entity_qa_overlay: {ov.get('safe_for_entity_qa_overlay')}")
    print()
    print("Source-qualified volatile facts:")
    for f in source_facts:
        print(f"  - {f.get('subject')} | {f.get('predicate')} = {f.get('object')} "
              f"| {f.get('source_name')} as_of {f.get('as_of')} "
              f"| recheck={f.get('requires_recheck')} risk={f.get('risk')}")


if __name__ == "__main__":
    main()
