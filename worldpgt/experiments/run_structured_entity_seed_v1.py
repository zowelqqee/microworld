"""Build a broad, structured multi-predicate seed cohort from Wikidata.

Existing evidence-local subjects remain in the target-enrichment cohort. This
separate cohort adds only new canonical entities whose official structured
claims contain at least two relation groups. Output is proposal-only.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
import os
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from worldpgt.benchmarks.open_book_qa.dataset import relation_id
from worldpgt.benchmarks.open_book_qa.heldout_v1 import _main_relation_ids
from worldpgt.knowledge_pump.wikidata_relation_layer import (
    _PROPERTY_TO_PREDICATE,
    extract_relation_rows,
)


_API = "https://www.wikidata.org/w/api.php"
_SPARQL = "https://query.wikidata.org/sparql"
_LANES = {
    "software": ("P178", "P366"),
    "organization": ("P112", "P159"),
    "corporate": ("P127", "P159"),
}
_DIVERSITY_FAMILIES = {
    "developed_by": "producer_or_owner",
    "founded_by": "producer_or_owner",
    "owned_by": "producer_or_owner",
    "product_of": "producer_or_owner",
    "headquartered_in": "location",
}


def _get_json(url: str, params: dict[str, str], *, user_agent: str) -> dict[str, Any]:
    request = Request(url + "?" + urlencode(params), headers={"User-Agent": user_agent})
    with urlopen(request, timeout=60) as response:  # nosec B310: official HTTPS APIs
        return json.loads(response.read().decode("utf-8"))


def _selector_query(limit_per_lane: int) -> str:
    branches = []
    for lane, (left, right) in _LANES.items():
        branches.append(
            "{ "
            f"?item wdt:{left} ?leftValue ; wdt:{right} ?rightValue . "
            f'BIND("{lane}" AS ?lane) '
            "}"
        )
    return "SELECT DISTINCT ?item ?lane WHERE { " + " UNION ".join(branches) + f" }} LIMIT {limit_per_lane * len(_LANES)}"


def _entity_batches(qids: list[str], *, user_agent: str, properties: str) -> dict[str, dict[str, Any]]:
    entities: dict[str, dict[str, Any]] = {}
    for start in range(0, len(qids), 50):
        payload = _get_json(_API, {
            "action": "wbgetentities", "ids": "|".join(qids[start:start + 50]),
            "props": properties, "languages": "en", "format": "json",
        }, user_agent=user_agent)
        entities.update({
            str(qid): item
            for qid, item in (payload.get("entities") or {}).items()
            if isinstance(item, dict)
        })
        sleep(0.2)
    return entities


def _object_qids(entities: dict[str, dict[str, Any]]) -> set[str]:
    return {
        str(value["id"])
        for entity in entities.values()
        for property_id in _PROPERTY_TO_PREDICATE
        for claim in (entity.get("claims") or {}).get(property_id, [])
        for value in [((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")]
        if claim.get("rank", "normal") != "deprecated"
        and isinstance(value, dict)
        and isinstance(value.get("id"), str)
    }


def _rows_for_entities(entities: dict[str, dict[str, Any]], labels: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for qid, entity in entities.items():
        enwiki = (entity.get("sitelinks") or {}).get("enwiki") or {}
        subject = str(entity.get("labels", {}).get("en", {}).get("value") or "")
        if not subject or not enwiki.get("title"):
            continue
        rows.extend(extract_relation_rows(
            surface_subject=subject,
            canonical_entity=subject,
            canonical_qid=qid,
            claims=entity.get("claims") or {},
            labels=labels,
        ))
    return rows


def _select_bounded_entities(
    rows: list[dict[str, Any]], *, max_entities: int, max_per_family_object: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        grouped[str(row["canonical_qid"])][str(row["predicate"])].append(row)
    accepted: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    family_object_counts: Counter[tuple[str, str]] = Counter()
    for qid in sorted(grouped, key=lambda value: sha256(value.encode()).hexdigest()):
        predicate_groups = grouped[qid]
        flattened = [row for predicate in sorted(predicate_groups) for row in predicate_groups[predicate]]
        if not (2 <= len(predicate_groups) and 2 <= len(flattened) <= 6):
            continue
        diversity_keys = {
            (_DIVERSITY_FAMILIES[predicate], str(row["object_qid"]))
            for predicate, predicate_rows in predicate_groups.items()
            if predicate in _DIVERSITY_FAMILIES
            for row in predicate_rows
            if str(row.get("object_qid") or "")
        }
        if any(family_object_counts[key] >= max_per_family_object for key in diversity_keys):
            continue
        manifest.append({
            "canonical_qid": qid,
            "canonical_entity": flattened[0]["canonical_entity"],
            "predicate_groups": sorted(predicate_groups),
            "relation_count": len(flattened),
            "wikipedia_anchor_required": True,
        })
        accepted.extend(flattened)
        family_object_counts.update(diversity_keys)
        if len(manifest) >= max_entities:
            break
    return accepted, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured multi-predicate Wikidata seed cohort")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/structured_entity_seed_v1")
    parser.add_argument("--max-entities", type=int, default=60)
    parser.add_argument("--selector-limit-per-lane", type=int, default=180)
    parser.add_argument("--max-per-family-object", type=int, default=3)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        raise SystemExit("refusing to fetch without --allow-network")
    if min(args.max_entities, args.selector_limit_per_lane, args.max_per_family_object) < 1:
        raise SystemExit("entity limits must be positive")
    user_agent = os.environ.get("MICROWORLD_WIKI_USER_AGENT", "")
    if not user_agent:
        raise SystemExit("MICROWORLD_WIKI_USER_AGENT is required")

    selector = _get_json(
        _SPARQL,
        {"query": _selector_query(args.selector_limit_per_lane), "format": "json"},
        user_agent=user_agent,
    )
    lane_by_qid: dict[str, set[str]] = defaultdict(set)
    for binding in selector.get("results", {}).get("bindings", []):
        qid = str(binding.get("item", {}).get("value") or "").rsplit("/", 1)[-1]
        lane = str(binding.get("lane", {}).get("value") or "")
        if qid.startswith("Q") and lane in _LANES:
            lane_by_qid[qid].add(lane)
    entities = _entity_batches(
        sorted(lane_by_qid), user_agent=user_agent, properties="claims|labels|sitelinks"
    )
    label_entities = _entity_batches(
        sorted(_object_qids(entities)), user_agent=user_agent, properties="labels"
    ) if entities else {}
    labels = {
        qid: str(item.get("labels", {}).get("en", {}).get("value") or "")
        for qid, item in label_entities.items()
    }
    all_rows = _rows_for_entities(entities, labels)
    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    all_rows = [row for row in all_rows if relation_id(row) not in main_ids]
    rows, manifest = _select_bounded_entities(
        all_rows,
        max_entities=args.max_entities,
        max_per_family_object=args.max_per_family_object,
    )
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "proposal_relations.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "frozen_entity_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    summary = {
        "version": "structured_entity_seed_v1",
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "selector": "official Wikidata entities with developer+use, founder+headquarters, or owner+headquarters",
        "wikipedia_anchor_required": True,
        "max_per_family_object": args.max_per_family_object,
        "selector_qid_count": len(lane_by_qid),
        "frozen_entity_count": len(manifest),
        "proposal_relation_count": len(rows),
        "predicate_distribution": dict(sorted(Counter(str(row["predicate"]) for row in rows).items())),
        "source_lanes_seen": dict(sorted(Counter(lane for lanes in lane_by_qid.values() for lane in lanes).items())),
        "main_edge_overlap_count": len({relation_id(row) for row in rows} & main_ids),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
