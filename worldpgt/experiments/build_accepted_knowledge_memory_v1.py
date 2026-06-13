"""Build the accepted knowledge memory artifact v1.

Loads accepted_auto facts from the auto-review JSON and accepted_auto patterns
from the wiki pattern candidates JSON, then writes a versioned, read-only
artifact JSON and a compact manifest JSON.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- No live engine state is changed.
- Output artifacts are read-only, standalone documents.

Usage:
    python3 -m worldpgt.experiments.build_accepted_knowledge_memory_v1 \\
      --auto-review worldpgt/experiments/knowledge_ingestion_v1_auto_review.json \\
      --patterns   worldpgt/experiments/wiki_pattern_candidates_v1.json \\
      --output-json    worldpgt/experiments/accepted_knowledge_memory_v1.json \\
      --manifest-json  worldpgt/experiments/accepted_knowledge_memory_v1_manifest.json
"""

from __future__ import annotations

import argparse
import json

from worldpgt.knowledge.safe_memory_applier import SafeMemoryApplier

_MEMORY_VERSION = "accepted_knowledge_memory_v1"


def run(
    auto_review_path: str,
    patterns_path: str,
    output_json_path: str,
    manifest_json_path: str,
) -> dict:
    """Build and write the accepted memory artifact. Returns the stats dict."""
    with open(auto_review_path, encoding="utf-8") as f:
        auto_review_data = json.load(f)
    with open(patterns_path, encoding="utf-8") as f:
        patterns_data = json.load(f)

    applier = SafeMemoryApplier()
    applier.build_from_sources(auto_review_data, patterns_data)

    stats = applier.stats
    items_as_dicts = [it.to_dict() for it in applier.items]

    artifact = {
        "memory_version": _MEMORY_VERSION,
        "mode": "artifact_only",
        "auto_apply_to_live_memory": False,
        "source_artifacts": {
            "auto_review": auto_review_path,
            "patterns": patterns_path,
        },
        "items": items_as_dicts,
        "stats": {
            "fact_items": stats["fact_items"],
            "pattern_items": stats["pattern_items"],
            "total_items": stats["total_items"],
            "by_term": stats["by_term"],
            "by_sense": stats["by_sense"],
            "by_item_type": stats["by_item_type"],
        },
        "safety": {
            "sense_memory_modified": False,
            "live_behavior_modified": False,
            "thresholds_changed": False,
            "validators_weakened": False,
            "generic_fallback_added": False,
        },
    }

    manifest = {
        "memory_version": _MEMORY_VERSION,
        "artifact_path": output_json_path,
        "total_items": stats["total_items"],
        "fact_items": stats["fact_items"],
        "pattern_items": stats["pattern_items"],
        "excluded_needs_review": stats["excluded_needs_review"],
        "excluded_rejected_auto": stats["excluded_rejected_auto"],
        "excluded_high_risk": stats["excluded_high_risk"],
        "excluded_broad_or_generic": stats["excluded_broad_or_generic"],
        "deduplicated_count": stats["deduplicated_count"],
        "by_item_type": stats["by_item_type"],
        "by_term": stats["by_term"],
        "safety": artifact["safety"],
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)
        f.write("\n")

    with open(manifest_json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")

    return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        description="Build accepted knowledge memory artifact v1."
    )
    parser.add_argument("--auto-review", required=True, dest="auto_review")
    parser.add_argument("--patterns", required=True)
    parser.add_argument("--output-json", required=True, dest="output_json")
    parser.add_argument("--manifest-json", required=True, dest="manifest_json")
    args = parser.parse_args(argv)

    stats = run(
        auto_review_path=args.auto_review,
        patterns_path=args.patterns,
        output_json_path=args.output_json,
        manifest_json_path=args.manifest_json,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
