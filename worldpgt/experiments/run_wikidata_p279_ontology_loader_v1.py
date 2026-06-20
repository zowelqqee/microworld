"""Build a read-only Wikidata P279 ontology layer for a local overlay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.knowledge.wikidata_ontology_loader import (
    WikidataApiClient,
    build_wikidata_p279_ontology_layer,
    empty_is_a_object_labels,
    write_ontology_layer_artifacts,
)

_EXPERIMENTS = Path(__file__).resolve().parent
_DEFAULT_BASE = _EXPERIMENTS / "knowledge_pump_v1" / "is_a_promotion_v1" / "pump_is_a_promoted_overlay.json"
_DEFAULT_OUT = _EXPERIMENTS / "knowledge_pump_v1" / "wikidata_p279_ontology_v1"


def run(
    *,
    base_overlay_path: Path = _DEFAULT_BASE,
    out_dir: Path = _DEFAULT_OUT,
    max_depth: int = 3,
    max_seed_labels: int = 80,
    max_edges_per_node: int = 4,
    sleep_seconds: float = 0.05,
) -> dict:
    overlay_items = json.loads(base_overlay_path.read_text(encoding="utf-8"))
    client = WikidataApiClient(sleep_seconds=sleep_seconds)
    layer, report = build_wikidata_p279_ontology_layer(
        overlay_items,
        client,
        max_depth=max_depth,
        max_seed_labels=max_seed_labels,
        max_edges_per_node=max_edges_per_node,
    )
    return write_ontology_layer_artifacts(
        base_overlay_path=base_overlay_path,
        layer_items=layer,
        report=report,
        out_dir=out_dir,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a read-only Wikidata P279 ontology layer.")
    parser.add_argument("--base-overlay", default=str(_DEFAULT_BASE))
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUT))
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-seed-labels", type=int, default=80)
    parser.add_argument("--max-edges-per-node", type=int, default=4)
    parser.add_argument("--sleep-seconds", type=float, default=0.05)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Only print empty is_a object labels; do not call Wikidata.",
    )
    args = parser.parse_args(argv)

    base_path = Path(args.base_overlay)
    items = json.loads(base_path.read_text(encoding="utf-8"))
    empty_labels = empty_is_a_object_labels(items)
    if args.audit_only:
        print(f"empty_is_a_object_labels: {len(empty_labels)}")
        for label in empty_labels:
            print(f"  {label}")
        return 0

    report = run(
        base_overlay_path=base_path,
        out_dir=Path(args.out_dir),
        max_depth=args.max_depth,
        max_seed_labels=args.max_seed_labels,
        max_edges_per_node=args.max_edges_per_node,
        sleep_seconds=args.sleep_seconds,
    )
    print("Wikidata P279 Ontology Loader v1")
    for key in (
        "seed_empty_label_count",
        "searched_label_count",
        "resolved_seed_count",
        "raw_edge_count",
        "accepted_edge_count",
        "rejected_edge_count",
        "base_overlay_items",
        "merged_overlay_items",
        "layer_path",
        "merged_overlay_path",
    ):
        print(f"  {key}: {report.get(key)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
