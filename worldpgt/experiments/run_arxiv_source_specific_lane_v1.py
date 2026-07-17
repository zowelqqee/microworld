"""Run one bounded source-specific relation lane over existing quarantine.

This is proposal-only: it reads a lane's already-quarantined relations for the
held-out-v1 target pool, re-derives them from their stored raw
records, and sends them through the unchanged open-web precision gates.  It
never modifies accepted, promoted, or serving memory.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from worldpgt.benchmarks.open_book_qa.dataset import _norm, load_experimental_relations
from worldpgt.benchmarks.open_book_qa.heldout_v1 import _main_relation_ids, heldout_pool_diagnostics
from worldpgt.knowledge_pump.open_web_pump import build_proposal_overlay
from worldpgt.knowledge_pump.source_specific_relation_extractors import extract_explicit_relations


_PUMP_ROOT = Path("worldpgt/experiments/open_web_pump_v1")


def _canonical_url(value: object) -> str:
    parts = urlsplit(str(value or "").strip())
    return urlunsplit(("", parts.netloc.casefold(), parts.path.rstrip("/"), parts.query, ""))


def _candidate_key(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        _norm(str(item.get("subject") or "")),
        _norm(str(item.get("predicate") or "")),
        _norm(str(item.get("object") or "")),
        _canonical_url(item.get("source_url")),
    )


def lane_candidates(lane: str) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    """Return one exact source-lane slice of the 331-subject audit pool."""

    main_ids = _main_relation_ids("artifacts/open_book_qa/dataset.jsonl")
    _groups, by_subject, _summary = heldout_pool_diagnostics(load_experimental_relations(), main_ids)
    target_subjects = set(by_subject)
    original_predicates = {subject: set(groups) for subject, groups in by_subject.items()}
    candidates: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for path in _PUMP_ROOT.rglob("open_web_quarantine.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("quarantine") or []
        for row in rows:
            if not isinstance(row, dict) or row.get("reason") != "open_web_relation_requires_source_specific_extractor":
                continue
            item = row.get("item") or {}
            if (
                not isinstance(item, dict)
                or str(item.get("source_kind") or "") != lane
                or _norm(str(item.get("subject") or "")) not in target_subjects
            ):
                continue
            candidates.setdefault(_candidate_key(item), dict(item))
    return [candidates[key] for key in sorted(candidates)], original_predicates


def _records_for_urls(lane: str, urls: set[str]) -> list[dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for manifest_path in _PUMP_ROOT.rglob("source_manifest.json"):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for row in manifest:
            if not isinstance(row, dict) or str(row.get("source_kind") or "") != lane:
                continue
            key = _canonical_url(row.get("source_url"))
            if key not in urls or key in records:
                continue
            raw_path = Path(str(row.get("raw_snapshot_path") or ""))
            if not raw_path.is_file():
                continue
            raw = json.loads(raw_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                records[key] = raw
    return list(records.values())


def spot_check(lane: str, seed: int) -> list[dict[str, Any]]:
    candidates, original_predicates = lane_candidates(lane)
    records = _records_for_urls(lane, {_canonical_url(item.get("source_url")) for item in candidates})
    accepted, rejected = extract_explicit_relations(
        lane, source_records=records, allowlisted_candidates=candidates
    )
    by_key = {_candidate_key(item): item for item in accepted}
    rejected_by_key = {_candidate_key(dict(row.get("item") or {})): row["reason"] for row in rejected}
    selection = random.Random(seed).sample(candidates, min(10, len(candidates)))
    record_by_url = {_canonical_url(record.get("source_url")): record for record in records}
    rows = []
    for candidate in selection:
        parsed = by_key.get(_candidate_key(candidate))
        record = record_by_url.get(_canonical_url(candidate.get("source_url")), {})
        rows.append({
            "candidate": candidate,
            "source_title": record.get("title", ""),
            "source_text": record.get("text", ""),
            "parser_verdict": "extracted" if parsed else "rejected",
            "parser_reason": "" if parsed else rejected_by_key.get(_candidate_key(candidate), "unknown"),
            "extracted": parsed,
            "existing_predicates": sorted(original_predicates.get(_norm(str(candidate.get("subject") or "")), set())),
            "predicate_novelty_before_extraction": (
                "potential_new_predicate"
                if str(candidate.get("predicate") or "") not in original_predicates.get(_norm(str(candidate.get("subject") or "")), set())
                else "duplicates_existing_predicate"
            ),
        })
    return rows


def full_run(lane: str) -> dict[str, Any]:
    candidates, original_predicates = lane_candidates(lane)
    records = _records_for_urls(lane, {_canonical_url(item.get("source_url")) for item in candidates})
    extracted, parser_rejected = extract_explicit_relations(
        lane, source_records=records, allowlisted_candidates=candidates
    )
    proposal = build_proposal_overlay([], source_specific_candidates=extracted)
    accepted = [item for item in proposal["proposal_overlay"] if item.get("open_web_extraction") == f"{lane}_explicit_relation_v1"]
    accepted_new_predicate = [
        item for item in accepted
        if str(item.get("predicate") or "") not in original_predicates.get(
            _norm(str(item.get("subject") or "")), set()
        )
    ]
    unlocked = {
        _norm(str(item.get("subject") or ""))
        for item in accepted_new_predicate
    }
    reasons = Counter(
        str(row.get("reason") or "unknown")
        for row in [*parser_rejected, *proposal["rejected"], *proposal["quarantine"]]
    )
    return {
        "lane": lane,
        "proposal_only": True,
        "accepted_memory_modified": False,
        "serving_overlay_modified": False,
        "unique_quarantine_relations": len(candidates),
        "stored_source_records_found": len(records),
        "source_specific_extracted": len(extracted),
        "passed_precision_gate": len(accepted),
        "passed_precision_gate_with_new_predicate": len(accepted_new_predicate),
        "passed_precision_gate_same_existing_predicate": len(accepted) - len(accepted_new_predicate),
        "rejected_or_quarantined": len(parser_rejected) + len(proposal["rejected"]) + len(proposal["quarantine"]),
        "reason_codes": dict(sorted(reasons.items())),
        "potential_unique_subjects": len({_norm(str(item.get("subject") or "")) for item in candidates}),
        "actual_subjects_with_second_relation_group": len(unlocked),
        "actual_unlocked_subjects": sorted(unlocked),
        "input_fingerprint": hashlib.sha256(
            json.dumps([_candidate_key(item) for item in candidates], sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Bounded source-specific relation lane")
    parser.add_argument("--lane", choices=("arxiv", "crossref", "openalex"), default="arxiv")
    parser.add_argument("--spot-check", action="store_true")
    parser.add_argument("--full-run", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--output", default="artifacts/open_book_qa/extractors")
    args = parser.parse_args()
    if args.spot_check == args.full_run:
        raise SystemExit("select exactly one of --spot-check or --full-run")
    output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    if args.spot_check:
        print(json.dumps({"lane": args.lane, "seed": args.seed, "cases": spot_check(args.lane, args.seed)}, ensure_ascii=False, indent=2))
    else:
        report = full_run(args.lane)
        (output / f"{args.lane}_full_run_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
