"""Relation Extraction v2 probe runner.

Loads ready snapshot docs, builds the entity surface index, extracts explicit
relation candidates, validates them, and writes artifacts under
``worldpgt/experiments/relation_extraction_v2/``.

SAFETY CONTRACT:
- No overlay written: accepted / promoted / snapshot dry-run untouched.
- No trusted memory written.
- No auto-ingest, no auto-promote, no auto-apply.
- No network calls.
- No neural / GPT / training / embedding code.
- nanogpt/ untouched.
- safe_for_general_runtime: always False.

Usage::

    python3 worldpgt/experiments/run_relation_extraction_v2_probe.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_candidate_extractor import extract_all_candidates
from worldpgt.relation_extraction_v2.relation_candidate_validator import validate_candidates
from worldpgt.relation_extraction_v2.relation_extraction_report import build_report
from worldpgt.relation_extraction_v2.types import RelationExtractionSummary
from worldpgt.wiki_snapshot_ingestion.ready_snapshot_loader import load_ready_snapshot_docs

_EXPERIMENTS = Path(__file__).resolve().parent
_SNAPSHOTS_DIR = _EXPERIMENTS / "wiki_snapshots_v1"
_READINESS = _SNAPSHOTS_DIR / "snapshot_readiness_report.json"
_MANIFEST = _SNAPSHOTS_DIR / "snapshot_manifest.json"
_ACCEPTED_OVERLAY = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_SNAPSHOT_DRY_RUN = _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
_DEFAULT_OUTDIR = _EXPERIMENTS / "relation_extraction_v2"

_PROTECTED = {
    "trusted_memory": _EXPERIMENTS / "accepted_knowledge_memory_v1.json",
    "accepted_overlay": _ACCEPTED_OVERLAY,
    "promoted_overlay": _PROMOTED_OVERLAY,
    "snapshot_dry_run_overlay": _SNAPSHOT_DRY_RUN,
}

_OUTPUT_CSV_FIELDS = [
    "id", "subject", "relation", "object", "confidence", "stability", "risk",
    "source_title", "pattern_id", "directionality", "safe_for_overlay_delta",
    "evidence_sentence",
]


def _hash(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_all_overlay_items() -> list[dict]:
    items: list[dict] = []
    for path in (_ACCEPTED_OVERLAY, _PROMOTED_OVERLAY, _SNAPSHOT_DRY_RUN):
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                items.extend(data)
    return items


def run(outdir: Path = _DEFAULT_OUTDIR) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    before_hashes = {k: _hash(v) for k, v in _PROTECTED.items()}

    # ---- Load ready docs -------------------------------------------- #
    ready_docs, skipped = load_ready_snapshot_docs(_READINESS, _MANIFEST)
    summary = RelationExtractionSummary(
        ready_docs_total=len(ready_docs) + len(skipped),
    )

    # ---- Build entity surface index ---------------------------------- #
    index = EntitySurfaceIndex(
        accepted_overlay_path=_ACCEPTED_OVERLAY,
        promoted_overlay_path=_PROMOTED_OVERLAY,
        snapshot_overlay_path=_SNAPSHOT_DRY_RUN,
        snapshot_manifest_path=_MANIFEST,
    )

    # ---- Extract raw candidates -------------------------------------- #
    raw_candidates, total_sentences, errors = extract_all_candidates(ready_docs, index)

    # ---- Load existing overlay for dup/conflict check ---------------- #
    all_overlay_items = _load_all_overlay_items()

    # ---- Validate candidates ----------------------------------------- #
    safe, quarantine, dup_ids, conflict_ids = validate_candidates(
        raw_candidates, index, all_overlay_items
    )

    # ---- Build summary ----------------------------------------------- #
    by_rel: dict[str, int] = {}
    by_stab: dict[str, int] = {}
    by_risk: dict[str, int] = {}
    for c in safe:
        by_rel[c.relation] = by_rel.get(c.relation, 0) + 1
        by_stab[c.stability] = by_stab.get(c.stability, 0) + 1
        by_risk[c.risk] = by_risk.get(c.risk, 0) + 1

    q_by_reason: dict[str, int] = {}
    for q in quarantine:
        for r in q.reason.split("; "):
            r = r.strip()
            if r:
                q_by_reason[r] = q_by_reason.get(r, 0) + 1

    after_hashes = {k: _hash(v) for k, v in _PROTECTED.items()}

    summary.docs_processed = len(ready_docs)
    summary.docs_failed = len(errors)
    summary.sentences_scanned = total_sentences
    summary.raw_relation_candidates_total = len(raw_candidates)
    summary.accepted_relation_candidates_total = len(safe)
    summary.quarantined_candidates_total = len(quarantine)
    summary.candidates_by_relation_type = by_rel
    summary.candidates_by_stability = by_stab
    summary.candidates_by_risk = by_risk
    summary.quarantine_by_reason = q_by_reason
    summary.duplicates_existing_count = len(dup_ids)
    summary.conflicts_existing_count = len(conflict_ids)
    summary.safe_for_overlay_delta_count = sum(1 for c in safe if c.safe_for_overlay_delta)
    summary.trusted_memory_modified = before_hashes["trusted_memory"] != after_hashes["trusted_memory"]
    summary.accepted_overlay_modified = before_hashes["accepted_overlay"] != after_hashes["accepted_overlay"]
    summary.promoted_overlay_modified = before_hashes["promoted_overlay"] != after_hashes["promoted_overlay"]
    summary.snapshot_dry_run_overlay_modified = (
        before_hashes["snapshot_dry_run_overlay"] != after_hashes["snapshot_dry_run_overlay"]
    )

    # ---- Write artifacts --------------------------------------------- #
    (outdir / "relation_extraction_v2_candidates.json").write_text(
        json.dumps([c.to_dict() for c in safe], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with (outdir / "relation_extraction_v2_candidates.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_OUTPUT_CSV_FIELDS)
        w.writeheader()
        for c in safe:
            row = c.to_dict()
            w.writerow({k: row.get(k, "") for k in _OUTPUT_CSV_FIELDS})

    (outdir / "relation_extraction_v2_quarantine.json").write_text(
        json.dumps([q.to_dict() for q in quarantine], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary_dict = summary.to_dict()
    (outdir / "relation_extraction_v2_summary.json").write_text(
        json.dumps(summary_dict, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    report = build_report(summary, safe, quarantine, conflict_ids, errors)
    (outdir / "relation_extraction_v2_report.json").write_text(
        json.dumps(report.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    return summary_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Relation Extraction v2 probe")
    parser.add_argument("--outdir", default=str(_DEFAULT_OUTDIR))
    args = parser.parse_args(argv)

    s = run(Path(args.outdir))

    print(f"[relation_extraction_v2] ready_docs_total:               {s['ready_docs_total']}")
    print(f"[relation_extraction_v2] docs_processed:                 {s['docs_processed']}")
    print(f"[relation_extraction_v2] docs_failed:                    {s['docs_failed']}")
    print(f"[relation_extraction_v2] sentences_scanned:              {s['sentences_scanned']}")
    print(f"[relation_extraction_v2] raw_relation_candidates_total:  {s['raw_relation_candidates_total']}")
    print(f"[relation_extraction_v2] accepted_relation_candidates:   {s['accepted_relation_candidates_total']}")
    print(f"[relation_extraction_v2] quarantined_candidates:         {s['quarantined_candidates_total']}")
    print(f"[relation_extraction_v2] duplicates_existing_count:      {s['duplicates_existing_count']}")
    print(f"[relation_extraction_v2] conflicts_existing_count:       {s['conflicts_existing_count']}")
    print(f"[relation_extraction_v2] safe_for_overlay_delta_count:   {s['safe_for_overlay_delta_count']}")
    print(f"[relation_extraction_v2] candidates_by_relation_type:    {s['candidates_by_relation_type']}")
    print(f"[relation_extraction_v2] quarantine_by_reason:           {s['quarantine_by_reason']}")

    print("\n[safety] trusted_memory_modified:              " + str(s['trusted_memory_modified']))
    print("[safety] accepted_overlay_modified:            " + str(s['accepted_overlay_modified']))
    print("[safety] promoted_overlay_modified:            " + str(s['promoted_overlay_modified']))
    print("[safety] snapshot_dry_run_overlay_modified:    " + str(s['snapshot_dry_run_overlay_modified']))
    print("[safety] network_calls:                        " + str(s['network_calls']))
    print("[safety] auto_ingest:                          " + str(s['auto_ingest']))
    print("[safety] auto_promote:                         " + str(s['auto_promote']))
    print("[safety] safe_for_general_runtime:             " + str(s['safe_for_general_runtime']))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
