"""Build Wiki Candidate Memory Overlay v1 from ingestion v2 candidates.

Input:  wiki_ingestion_v2_candidates.json
Output: accepted_wiki_memory_overlay_v1.json
        accepted_wiki_memory_overlay_v1_summary.json
        accepted_wiki_memory_overlay_v1_skipped.json

SAFETY CONTRACT:
- accepted_knowledge_memory_v1.json is NOT modified.
- sense_memory.py is NOT modified.
- Planner thresholds are NOT modified.
- Validators are NOT weakened.
- No generic fallback is added.
- No neural/GPT/training/embedding code.
- No network access.
- nanogpt/ is untouched.
- safe_for_general_runtime is always False.

Usage::

    python3 -m worldpgt.experiments.build_wiki_memory_overlay_v1 \\
      --candidates-json worldpgt/experiments/wiki_ingestion_v2_candidates.json \\
      --output-json worldpgt/experiments/accepted_wiki_memory_overlay_v1.json \\
      --summary-json worldpgt/experiments/accepted_wiki_memory_overlay_v1_summary.json \\
      --skipped-json worldpgt/experiments/accepted_wiki_memory_overlay_v1_skipped.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from worldpgt.knowledge.wiki_candidate_overlay_builder import (
    WikiCandidateOverlayBuilder,
    as_dict,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build wiki memory overlay v1 from ingestion v2 candidates"
    )
    parser.add_argument("--candidates-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--summary-json", required=False, default=None)
    parser.add_argument("--skipped-json", required=False, default=None)
    args = parser.parse_args(argv)

    builder = WikiCandidateOverlayBuilder()
    candidates = builder.load_candidates(args.candidates_json)
    print(f"[overlay_v1] Loaded {len(candidates)} candidates from {args.candidates_json}")

    result = builder.build(candidates)
    s = result.summary

    print(f"[overlay_v1] Overlay items total:    {s['overlay_items_total']}")
    print(f"[overlay_v1] Skipped candidates:     {s['skipped_candidates_total']}")
    print(f"[overlay_v1] by_overlay_type:        {s['by_overlay_type']}")
    print(f"[overlay_v1] safe_for_general_runtime:   {s['safe_for_general_runtime']}")
    print(f"[overlay_v1] safe_for_entity_qa_overlay: {s['safe_for_entity_qa_overlay']}")

    Path(args.output_json).write_text(
        json.dumps([as_dict(i) for i in result.items], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[overlay_v1] Overlay JSON written to {args.output_json}")

    if args.summary_json:
        Path(args.summary_json).write_text(
            json.dumps(s, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[overlay_v1] Summary JSON written to {args.summary_json}")

    if args.skipped_json:
        Path(args.skipped_json).write_text(
            json.dumps([as_dict(sk) for sk in result.skipped], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[overlay_v1] Skipped JSON written to {args.skipped_json}")

    print("\n[safety] accepted_knowledge_memory_v1.json: NOT modified")
    print("[safety] sense_memory.py: NOT modified")
    print("[safety] planner thresholds: NOT modified")
    print("[safety] validators: NOT weakened")
    print("[safety] generic fallback: NOT added")
    print("[safety] neural/GPT/training/embedding: NOT used")
    print("[safety] network access: NONE")
    print("[safety] nanogpt/: NOT touched")
    print(f"[safety] safe_for_general_runtime: {s['safe_for_general_runtime']}")
    print(f"[safety] safe_for_entity_qa_overlay: {s['safe_for_entity_qa_overlay']}")


if __name__ == "__main__":
    main()
