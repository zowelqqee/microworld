"""Acquire new predicate groups for existing subjects from official Wikidata.

Unlike the previous alphabetical Wikipedia batch, this runner resolves every
existing one-group subject first.  It keeps unresolved surface fragments in the
resolution manifest, but uses only an unambiguous exact Wikidata item for a
structured proposal relation.  It never writes accepted or serving memory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from worldpgt.benchmarks.open_book_qa.dataset import _norm, load_experimental_relations
from worldpgt.benchmarks.open_book_qa.heldout_v1 import _main_relation_ids
from worldpgt.knowledge_pump.heldout_density_frontier import (
    attach_wikidata_exact_resolution,
    build_density_frontier,
    require_wikipedia_anchor,
)
from worldpgt.knowledge_pump.wikidata_relation_layer import (
    _PROPERTY_TO_PREDICATE,
    extract_relation_rows,
)


_API = "https://www.wikidata.org/w/api.php"


class WikidataClient:
    """Rate-limited official API client with batched entity and label reads."""

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
        return [row for row in payload.get("search", []) if isinstance(row, dict)]

    def entities(self, qids: list[str], *, properties: str) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for start in range(0, len(qids), 50):
            payload = self.get({
                "action": "wbgetentities", "ids": "|".join(qids[start:start + 50]),
                "props": properties, "languages": "en", "format": "json",
            })
            result.update({
                str(qid): entity
                for qid, entity in (payload.get("entities") or {}).items()
                if isinstance(entity, dict)
            })
        return result


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


def _deduplicate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (
            _norm(str(row.get("surface_subject") or row.get("subject") or "")),
            str(row.get("predicate") or ""),
            _norm(str(row.get("object") or "")),
            str(row.get("wikidata_property") or ""),
        )
        unique.setdefault(key, row)
    return [unique[key] for key in sorted(unique)]


def _proposal_density(frontier: list[dict[str, Any]], proposals: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, set[str]] = {
        _norm(str(row["surface_subject"])): set(row.get("existing_clean_predicates") or ())
        for row in frontier
    }
    for row in proposals:
        groups.setdefault(_norm(str(row["surface_subject"])), set()).add(str(row["predicate"]))
    counts = Counter(len(predicates) for predicates in groups.values())
    return {
        "predicate_groups_per_subject": dict(sorted((str(count), total) for count, total in counts.items())),
        "subjects_with_1_predicate_group": counts[1],
        "subjects_with_2_predicate_groups": counts[2],
        "subjects_with_3_or_more_predicate_groups": sum(
            total for count, total in counts.items() if count >= 3
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Target-first structured Wikidata density acquisition")
    parser.add_argument("--output-dir", default="artifacts/open_book_qa/targeted_wikidata_density_v2")
    parser.add_argument("--max-targets", type=int, default=0, help="0 means every eligible existing subject")
    parser.add_argument("--delay-seconds", type=float, default=0.15)
    parser.add_argument("--allow-network", action="store_true")
    args = parser.parse_args()
    if not args.allow_network:
        raise SystemExit("refusing to fetch without --allow-network")
    if args.max_targets < 0:
        raise SystemExit("--max-targets must be non-negative")
    user_agent = os.environ.get("MICROWORLD_WIKI_USER_AGENT", "")
    if not user_agent:
        raise SystemExit("MICROWORLD_WIKI_USER_AGENT is required")

    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    frontier, frontier_summary = build_density_frontier(load_experimental_relations(), main_ids)
    if args.max_targets:
        frontier = frontier[:args.max_targets]
    client = WikidataClient(user_agent=user_agent, delay_seconds=args.delay_seconds)
    search_results = {
        _norm(str(target["surface_subject"])): client.search(str(target["surface_subject"]))
        for target in frontier
    }
    resolved = attach_wikidata_exact_resolution(frontier, search_results)
    qids = sorted({str(row["canonical_qid"]) for row in resolved if row.get("canonical_qid")})
    entities = client.entities(qids, properties="claims|labels|sitelinks") if qids else {}
    resolved = require_wikipedia_anchor(resolved, entities)
    labels_by_qid = client.entities(sorted(_object_qids(entities)), properties="labels") if entities else {}
    labels = {
        qid: str(entity.get("labels", {}).get("en", {}).get("value") or "")
        for qid, entity in labels_by_qid.items()
    }
    proposals: list[dict[str, Any]] = []
    for target in resolved:
        if target.get("canonical_resolution_status") != "resolved_wikidata_exact_enwiki":
            continue
        qid = str(target.get("canonical_qid") or "")
        entity = entities.get(qid)
        if entity is None:
            continue
        canonical = str(entity.get("labels", {}).get("en", {}).get("value") or target["canonical_entity"])
        proposals.extend(extract_relation_rows(
            surface_subject=str(target["surface_subject"]),
            canonical_entity=canonical,
            canonical_qid=qid,
            claims=entity.get("claims") or {},
            labels=labels,
            blocked_predicates=(
                set(target.get("existing_clean_predicates") or ())
                | set(target.get("predicates_touched_by_main") or ())
            ),
        ))
    proposals = _deduplicate(proposals)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "resolution_manifest.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (root / "proposal_relations.json").write_text(
        json.dumps(proposals, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    status_counts = Counter(str(row["canonical_resolution_status"]) for row in resolved)
    new_subjects = {_norm(str(row["surface_subject"])) for row in proposals}
    summary = {
        "version": "targeted_wikidata_density_v2",
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "selection": "all existing one-predicate subjects; exact unambiguous Wikidata resolution with English Wikipedia anchor",
        "target_subject_count": len(frontier),
        "resolution_status_counts": dict(sorted(status_counts.items())),
        "exact_resolved_subjects": len(qids),
        "anchored_resolved_subjects": sum(
            row["canonical_resolution_status"] == "resolved_wikidata_exact_enwiki"
            for row in resolved
        ),
        "proposal_relation_count": len(proposals),
        "subjects_with_new_predicate_group": len(new_subjects),
        "proposal_predicate_distribution": dict(sorted(Counter(
            str(row["predicate"]) for row in proposals
        ).items())),
        "relation_density_before": frontier_summary["target_density_before_acquisition"],
        "relation_density_after_proposal": _proposal_density(frontier, proposals),
        "network_calls": client.calls,
    }
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
