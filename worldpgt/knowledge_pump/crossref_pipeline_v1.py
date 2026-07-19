"""Bounded, cacheable, proposal-only Crossref DOI pipeline."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
from time import sleep
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

from worldpgt.benchmarks.open_book_qa.dataset import _norm, load_experimental_relations, relation_id
from worldpgt.knowledge_pump.crossref_doi_gate import validate_crossref_doi_proposals
from worldpgt.knowledge_pump.crossref_doi_seed import extract_doi_relation_rows

_API = "https://api.crossref.org/works"


def _clean_doi(value: object) -> str:
    return " ".join(str(value or "").split()).casefold()


def load_dois(value: str) -> list[str]:
    """Accept comma-separated DOIs or an existing manifest/JSON list."""
    path = Path(value)
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list): raise ValueError("--dois JSON must be a list")
        raw = [
            (item.get("canonical_doi") or item.get("DOI") or "") if isinstance(item, dict) else item
            for item in payload if isinstance(item, (str, dict))
        ]
    else:
        raw = value.split(",")
    return list(dict.fromkeys(doi for doi in (_clean_doi(item) for item in raw) if doi))


class CrossrefClient:
    def __init__(self, *, user_agent: str, delay_seconds: float) -> None:
        self.user_agent, self.delay_seconds, self.calls = user_agent, delay_seconds, 0

    def work(self, doi: str) -> dict[str, Any]:
        if self.calls: sleep(self.delay_seconds)
        request = Request(f"{_API}/{quote(doi, safe='')}", headers={"User-Agent": self.user_agent, "Accept": "application/json"})
        with urlopen(request, timeout=45) as response:  # nosec B310: official HTTPS API
            self.calls += 1
            payload = json.loads(response.read().decode("utf-8"))
        return payload if isinstance(payload, dict) else {}


def remove_serving_overlap(rows: list[dict[str, Any]], serving_rows: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], int]:
    serving_ids = {relation_id(row) for row in (load_experimental_relations() if serving_rows is None else serving_rows)}
    kept = [row for row in rows if relation_id(row) not in serving_ids]
    return kept, len(rows) - len(kept)


def run_pipeline(dois: list[str], *, client: CrossrefClient | None, save_raw_dir: Path | None, serving_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Fetch each requested DOI exactly once; no query/discovery expansion occurs."""
    if client is None:
        return {"raw": {}, "raw_candidates": [], "candidates": [], "gate": validate_crossref_doi_proposals([]), "overlap": 0, "network_calls": 0, "errors": []}
    raw: dict[str, dict[str, Any]] = {}; errors = []; candidates = []
    if save_raw_dir: save_raw_dir.mkdir(parents=True, exist_ok=True)
    for doi in dois:
        try:
            response = client.work(doi)
            raw[doi] = response
            if save_raw_dir:
                # DOI-safe deterministic filename preserves the full envelope
                # (status/message/etc.), not just fields used by extraction.
                (save_raw_dir / (quote(doi, safe="") + ".json")).write_text(json.dumps(response, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            item = response.get("message") if isinstance(response.get("message"), dict) else {}
            candidates.extend(extract_doi_relation_rows(item, topic_bucket="explicit_doi_input"))
        except Exception as exc:
            errors.append({"doi": doi, "error": str(exc)[:240]})
    unique = {relation_id(row): row for row in candidates}
    raw_candidates = [unique[key] for key in sorted(unique)]
    filtered, overlap = remove_serving_overlap(raw_candidates, serving_rows)
    return {"raw": raw, "raw_candidates": raw_candidates, "candidates": filtered, "gate": validate_crossref_doi_proposals(filtered), "overlap": overlap, "network_calls": client.calls, "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser(description="Proposal-only bounded Crossref DOI expansion")
    parser.add_argument("--dois", required=True, help="comma-separated DOI list or JSON manifest/list")
    parser.add_argument("--output", required=True)
    parser.add_argument("--save-raw-responses", action="store_true")
    parser.add_argument("--delay-seconds", type=float, default=.5)
    parser.add_argument("--allow-network", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.delay_seconds < 0: raise SystemExit("--delay-seconds must be non-negative")
    if not args.dry_run and not args.allow_network: raise SystemExit("refusing to fetch without --allow-network (or use --dry-run)")
    user_agent = os.environ.get("MICROWORLD_CROSSREF_USER_AGENT") or os.environ.get("MICROWORLD_WIKI_USER_AGENT", "")
    if not args.dry_run and not user_agent: raise SystemExit("MICROWORLD_CROSSREF_USER_AGENT or MICROWORLD_WIKI_USER_AGENT is required")
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    dois = load_dois(args.dois)
    result = run_pipeline(dois, client=None if args.dry_run else CrossrefClient(user_agent=user_agent, delay_seconds=args.delay_seconds), save_raw_dir=(root / "raw_responses") if args.save_raw_responses and not args.dry_run else None)
    for name, value in (("raw_candidates.json", result["raw_candidates"]), ("proposal_overlay.json", result["gate"]["accepted_proposal_overlay"]), ("rejected.json", result["gate"]["rejected"]), ("quarantine.json", result["gate"]["quarantine"])):
        (root / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    accepted = result["gate"]["accepted_proposal_overlay"]
    groups: dict[str, set[str]] = defaultdict(set)
    for row in accepted: groups[_norm(row["subject"])].add(str(row["predicate"]))
    summary = {"version": "crossref_pipeline_v1", "proposal_only": True, "accepted_memory_modified": False, "serving_overlay_modified": False, "review_state": "ready_for_human_review", "dry_run": args.dry_run, "dois_input": len(dois), "raw_response_cache_requested": args.save_raw_responses, "raw_response_cache_count": len(result["raw"]) if args.save_raw_responses else 0, "raw_candidates": len(result["raw_candidates"]), "candidates_after_serving_overlap_filter": len(result["candidates"]), "gate_passed": len(accepted), "new_subjects": len(groups), "new_multi_predicate_groups": sum(len(v) >= 2 for v in groups.values()), "already_promoted_overlap_filtered": result["overlap"], "proposal_relation_overlap_with_serving": 0, "network_calls": result["network_calls"], "errors": result["errors"], "gate": {key: value for key, value in result["gate"].items() if key not in {"accepted_proposal_overlay", "rejected", "quarantine"}}}
    (root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False)); return 0

if __name__ == "__main__": raise SystemExit(main())
