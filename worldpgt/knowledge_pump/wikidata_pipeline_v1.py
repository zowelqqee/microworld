"""Bounded, proposal-only Wikidata schema-expansion pipeline.

This module deliberately has no import from a promotion runner.  It resolves
subjects, extracts an explicit property whitelist, removes facts already in
the serving graph, and runs the established precision gate.
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

from worldpgt.benchmarks.open_book_qa.dataset import _norm, load_experimental_relations, relation_id
from worldpgt.knowledge_pump.heldout_density_frontier import attach_wikidata_exact_resolution
from worldpgt.knowledge_pump.wikidata_property_gate import validate_wikidata_property_proposals
from worldpgt.knowledge_pump.wikidata_relation_layer import _PROPERTY_TO_PREDICATE, extract_relation_rows

_API = "https://www.wikidata.org/w/api.php"
_DEFAULT_PROPERTIES = ("P1433", "P277", "P1535", "P1542", "P1072")
# Kept explicit instead of dynamically accepting every content claim: this is
# the bounded historical first-round set plus the five reviewed new mappings.
# Scalar P571 is intentionally fetched for audit completeness; the existing
# entity-edge extractor will quarantine rather than manufacture an endpoint.
_FIRST_ROUND_PLUS_TOP5 = (
    "P17", "P50", "P101", "P112", "P123", "P127", "P131", "P138", "P159", "P176", "P178",
    "P2283", "P275", "P276", "P282", "P2860", "P306", "P355", "P361", "P366", "P400", "P407",
    "P461", "P495", "P527", "P571", "P921", *_DEFAULT_PROPERTIES,
)


class WikidataClient:
    """Official Action API client with an invocation-wide request delay."""
    def __init__(self, *, user_agent: str, delay_seconds: float) -> None:
        self.user_agent, self.delay_seconds, self.calls = user_agent, delay_seconds, 0

    def get(self, params: dict[str, str]) -> dict[str, Any]:
        if self.calls:
            sleep(self.delay_seconds)
        request = Request(_API + "?" + urlencode(params), headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=45) as response:  # nosec B310: official HTTPS API
            self.calls += 1
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}

    def search(self, label: str) -> list[dict[str, Any]]:
        return [item for item in self.get({"action": "wbsearchentities", "search": label, "language": "en", "format": "json", "limit": "10", "type": "item"}).get("search", []) if isinstance(item, dict)]

    def entities(self, qids: list[str], *, properties: str) -> dict[str, dict[str, Any]]:
        rows: dict[str, dict[str, Any]] = {}
        for start in range(0, len(qids), 50):
            payload = self.get({"action": "wbgetentities", "ids": "|".join(qids[start:start + 50]), "props": properties, "languages": "en", "format": "json"})
            rows.update({str(qid): entity for qid, entity in (payload.get("entities") or {}).items() if isinstance(entity, dict)})
        return rows


def _load_subjects(value: str) -> list[dict[str, Any]]:
    if value == "main-dataset":
        rows = load_experimental_relations()
        by_subject: dict[str, dict[str, Any]] = {}
        for row in rows:
            subject = str(row.get("subject") or "").strip()
            if subject:
                by_subject.setdefault(_norm(subject), {"subject": subject, "surface_subject": subject})
        return [by_subject[key] for key in sorted(by_subject)]
    if value == "unresolved-pool":
        path = Path("artifacts/open_book_qa/wikidata_density_recon/resolution_manifest.json")
    else:
        path = Path(value)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("--subjects-source JSON must be a list")
    rows = []
    for item in payload:
        if isinstance(item, str) and item.strip():
            rows.append({"subject": item.strip(), "surface_subject": item.strip()})
        elif isinstance(item, dict):
            subject = str(item.get("surface_subject") or item.get("subject") or item.get("title") or "").strip()
            if subject:
                rows.append({**item, "subject": subject, "surface_subject": subject})
    if value == "unresolved-pool":
        rows = [row for row in rows if not row.get("canonical_qid")]
    return rows


def _property_ids(value: str) -> tuple[str, ...]:
    supported = tuple(dict.fromkeys((*_FIRST_ROUND_PLUS_TOP5, *_PROPERTY_TO_PREDICATE)))
    lookup = {pid: pid for pid in supported} | {predicate: pid for pid, predicate in _PROPERTY_TO_PREDICATE.items()}
    tokens = _DEFAULT_PROPERTIES if value == "default" else (_FIRST_ROUND_PLUS_TOP5 if value == "first-round-plus-top5" else tuple(token.strip() for token in value.split(",") if token.strip()))
    selected = tuple(lookup.get(token, lookup.get(token.casefold(), "")) for token in tokens)
    if not selected or any(not pid for pid in selected):
        raise ValueError("--property-whitelist accepts default, first-round-plus-top5, P1433,..., or published_in,...")
    return tuple(dict.fromkeys(selected))


def _object_qids(entities: dict[str, dict[str, Any]], properties: tuple[str, ...]) -> set[str]:
    return {str(value["id"]) for entity in entities.values() for pid in properties for claim in (entity.get("claims") or {}).get(pid, []) for value in [((claim.get("mainsnak") or {}).get("datavalue") or {}).get("value")] if claim.get("rank", "normal") != "deprecated" and isinstance(value, dict) and isinstance(value.get("id"), str)}


def remove_serving_overlap(rows: list[dict[str, Any]], serving_rows: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], int]:
    serving_ids = {relation_id(row) for row in (load_experimental_relations() if serving_rows is None else serving_rows)}
    kept = [row for row in rows if relation_id(row) not in serving_ids]
    return kept, len(rows) - len(kept)


def run_pipeline(subjects: list[dict[str, Any]], *, property_ids: tuple[str, ...], client: WikidataClient | None, serving_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Run the non-writing core; ``client=None`` is a deterministic dry run."""
    if client is None:
        return {"resolved": [{**row, "canonical_resolution_status": "not_fetched_dry_run"} for row in subjects], "raw_candidates": [], "candidates": [], "gate": validate_wikidata_property_proposals([]), "overlap": 0, "network_calls": 0}
    searches = {_norm(str(row["surface_subject"])): client.search(str(row["surface_subject"])) for row in subjects}
    candidate_qids = sorted({str(hit.get("id")) for hits in searches.values() for hit in hits if str(hit.get("id") or "").startswith("Q")})
    resolver_entities = client.entities(candidate_qids, properties="claims") if candidate_qids else {}
    # The alias resolver deliberately handles only the cases that exact
    # resolution cannot decide.  Preserve ordinary unique exact matches first.
    exact = attach_wikidata_exact_resolution(subjects, searches)
    unresolved = [row for row in exact if not row.get("canonical_qid")]
    aliases = attach_wikidata_exact_resolution(unresolved, searches, entities=resolver_entities, enable_alias_disambiguation=True)
    alias_by_subject = {_norm(str(row["surface_subject"])): row for row in aliases}
    resolved = [row if row.get("canonical_qid") else alias_by_subject[_norm(str(row["surface_subject"]))] for row in exact]
    qids = sorted({str(row["canonical_qid"]) for row in resolved if row.get("canonical_qid")})
    entities = client.entities(qids, properties="claims|labels|sitelinks") if qids else {}
    object_entities = client.entities(sorted(_object_qids(entities, property_ids)), properties="labels") if entities else {}
    labels = {qid: str(entity.get("labels", {}).get("en", {}).get("value") or "") for qid, entity in object_entities.items()}
    candidates = []
    for row in resolved:
        qid = str(row.get("canonical_qid") or "")
        entity = entities.get(qid)
        if not entity or not (entity.get("sitelinks") or {}).get("enwiki"):
            continue
        claims = {pid: (entity.get("claims") or {}).get(pid, []) for pid in property_ids}
        canonical = str(entity.get("labels", {}).get("en", {}).get("value") or row.get("canonical_entity") or row["surface_subject"])
        candidates.extend(extract_relation_rows(surface_subject=str(row["surface_subject"]), canonical_entity=canonical, canonical_qid=qid, claims=claims, labels=labels))
    unique = {relation_id(row): row for row in candidates}
    raw_candidates = [unique[key] for key in sorted(unique)]
    filtered, overlap = remove_serving_overlap(raw_candidates, serving_rows)
    return {"resolved": resolved, "raw_candidates": raw_candidates, "candidates": filtered, "gate": validate_wikidata_property_proposals(filtered), "overlap": overlap, "network_calls": client.calls}


def main() -> int:
    parser = argparse.ArgumentParser(description="Proposal-only bounded Wikidata schema expansion")
    parser.add_argument("--subjects-source", default="main-dataset", help="main-dataset, unresolved-pool, or JSON list")
    parser.add_argument("--property-whitelist", default="default")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-subjects", type=int, default=0)
    parser.add_argument("--skip-subjects", type=int, default=0, help="deterministically skip a prefix for resumable bounded batches")
    parser.add_argument("--delay-seconds", type=float, default=.15)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.max_subjects < 0 or args.skip_subjects < 0 or args.delay_seconds < 0: raise SystemExit("limits must be non-negative")
    if not args.dry_run and not args.allow_network: raise SystemExit("refusing to fetch without --allow-network (or use --dry-run)")
    subjects = _load_subjects(args.subjects_source)
    subjects = subjects[args.skip_subjects:]
    if args.max_subjects: subjects = subjects[:args.max_subjects]
    user_agent = os.environ.get("MICROWORLD_WIKI_USER_AGENT", "")
    if not args.dry_run and not user_agent: raise SystemExit("MICROWORLD_WIKI_USER_AGENT is required")
    result = run_pipeline(subjects, property_ids=_property_ids(args.property_whitelist), client=None if args.dry_run else WikidataClient(user_agent=user_agent, delay_seconds=args.delay_seconds))
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    for name, value in (("resolution_manifest.json", result["resolved"]), ("raw_candidates.json", result["raw_candidates"]), ("proposal_overlay.json", result["gate"]["accepted_proposal_overlay"]), ("rejected.json", result["gate"]["rejected"]), ("quarantine.json", result["gate"]["quarantine"])):
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accepted = result["gate"]["accepted_proposal_overlay"]
    groups: dict[str, set[str]] = defaultdict(set)
    for row in accepted: groups[_norm(row["subject"])].add(str(row["predicate"]))
    summary = {"version": "wikidata_pipeline_v1", "proposal_only": True, "accepted_memory_modified": False, "serving_overlay_modified": False, "review_state": "ready_for_human_review", "dry_run": args.dry_run, "subjects_input": len(subjects), "property_whitelist": list(_property_ids(args.property_whitelist)), "raw_candidates": len(result["raw_candidates"]), "candidates_after_serving_overlap_filter": len(result["candidates"]), "gate_passed": len(accepted), "new_subjects": len(groups), "new_multi_predicate_groups": sum(len(v) >= 2 for v in groups.values()), "already_promoted_overlap_filtered": result["overlap"], "proposal_relation_overlap_with_serving": 0, "network_calls": result["network_calls"], "gate": {key: value for key, value in result["gate"].items() if key not in {"accepted_proposal_overlay", "rejected", "quarantine"}}}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
