"""Wikipedia Snapshot Self-Ingestion v1 runner.

Offline proposal pipeline only:

- reads local snapshot readiness/manifest artifacts
- selects only ``ready_for_self_ingestion=true`` normalized docs
- reuses WikiIngestionV2 and WikiCandidateOverlayBuilder
- compares against accepted and promoted overlays
- writes quarantine, duplicate/conflict reports, delta proposal, dry-run overlay
- runs regressions against the dry-run overlay when possible

It never writes accepted memory, accepted overlay, promoted overlay, runtime QA
behavior, planner thresholds, validators, or ingestion semantics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

# Allow running directly as a script.
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from worldpgt.wiki_snapshot_ingestion.ready_snapshot_loader import (
    load_ready_snapshot_docs,
    write_selection,
)
from worldpgt.wiki_snapshot_ingestion.snapshot_batch_ingestor import (
    ingest_snapshot_batch,
    write_candidates,
)
from worldpgt.wiki_snapshot_ingestion.snapshot_delta_builder import (
    classify_snapshot_overlay_items,
    write_delta_artifacts,
)
from worldpgt.wiki_snapshot_ingestion.snapshot_ingestion_report import (
    build_report,
    build_summary,
    write_json,
)
from worldpgt.wiki_snapshot_ingestion.snapshot_quarantine import write_quarantine
from worldpgt.wiki_snapshot_ingestion.snapshot_regression_runner import run_snapshot_regressions

_EXPERIMENTS = Path(__file__).resolve().parent
_SNAPSHOTS = _EXPERIMENTS / "wiki_snapshots_v1"
_DEFAULT_OUTDIR = _EXPERIMENTS / "wiki_snapshot_ingestion_v1"
_READINESS = _SNAPSHOTS / "snapshot_readiness_report.json"
_MANIFEST = _SNAPSHOTS / "snapshot_manifest.json"
_ACCEPTED = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED = _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
_TRUSTED = _EXPERIMENTS / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY = _ROOT / "worldpgt" / "continuation" / "sense_memory.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _protected_hashes() -> dict[str, str]:
    return {
        "trusted_memory": _sha256(_TRUSTED),
        "accepted_overlay": _sha256(_ACCEPTED),
        "promoted_overlay": _sha256(_PROMOTED),
        "sense_memory": _sha256(_SENSE_MEMORY),
    }


def run_snapshot_self_ingestion(
    readiness_path: str | Path = _READINESS,
    manifest_path: str | Path = _MANIFEST,
    accepted_overlay_path: str | Path = _ACCEPTED,
    promoted_overlay_path: str | Path = _PROMOTED,
    out_dir: str | Path = _DEFAULT_OUTDIR,
    run_regressions: bool = True,
) -> dict[str, Any]:
    before = _protected_hashes()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_artifacts = [
        str(readiness_path),
        str(manifest_path),
        str(accepted_overlay_path),
        str(promoted_overlay_path),
    ]

    readiness_rows = _read_json(Path(readiness_path))
    selected, skipped = load_ready_snapshot_docs(readiness_path, manifest_path)
    write_selection(
        selected,
        out / "ready_snapshot_selection.json",
        out / "ready_snapshot_selection.csv",
    )

    candidates, overlay_items, doc_failures, candidates_by_type, doc_status = ingest_snapshot_batch(selected)
    write_candidates(
        candidates,
        out / "snapshot_ingestion_candidates.json",
        out / "snapshot_ingestion_candidates.csv",
    )

    accepted_items = _read_json(Path(accepted_overlay_path))
    promoted_items = _read_json(Path(promoted_overlay_path))
    (
        delta,
        dup_accepted,
        dup_promoted,
        conflicts,
        quarantine,
        tainted_sources,
    ) = classify_snapshot_overlay_items(overlay_items, accepted_items, promoted_items)
    quarantine = list(doc_failures) + quarantine
    q_breakdown = write_quarantine(quarantine, out / "snapshot_ingestion_quarantine.json")
    write_delta_artifacts(out, delta, dup_accepted, dup_promoted, conflicts)

    base_overlay_name = "promoted"
    dry_run_overlay = list(promoted_items) + [d.overlay_item for d in delta]
    dry_run_path = out / "snapshot_dry_run_overlay.json"
    write_json(dry_run_path, dry_run_overlay)

    warnings = []
    errors: list[dict[str, Any]] = []
    if tainted_sources:
        warnings.append(f"tainted_sources_excluded_from_delta:{len(tainted_sources)}")
    if not selected:
        warnings.append("no_ready_snapshot_docs_selected")

    if run_regressions:
        regressions = run_snapshot_regressions(_EXPERIMENTS, dry_run_path, out)
    else:
        regressions = []
        write_json(out / "snapshot_regression_summary.json", [])

    summary = build_summary(
        snapshot_docs_total=len(readiness_rows),
        ready_docs_selected=len(selected),
        not_ready_docs_skipped=len(skipped),
        doc_status=doc_status,
        candidates_total=len(candidates),
        candidates_by_type=candidates_by_type,
        overlay_items_total=len(overlay_items),
        duplicates_accepted_count=len(dup_accepted),
        duplicates_promoted_count=len(dup_promoted),
        conflicts_count=len(conflicts),
        quarantined_count=len(quarantine),
        rejected_count=sum(1 for q in quarantine if q.suggested_action == "reject"),
        safe_delta_items_count=len(delta),
        dry_run_overlay_items_count=len(dry_run_overlay),
        regressions=regressions,
    )

    after = _protected_hashes()
    if after != before:
        errors.append({"reason": "protected_file_hash_changed", "before": before, "after": after})
        summary.trusted_memory_modified = before["trusted_memory"] != after["trusted_memory"]
        summary.accepted_overlay_modified = before["accepted_overlay"] != after["accepted_overlay"]
        summary.promoted_overlay_modified = before["promoted_overlay"] != after["promoted_overlay"]

    summary_payload = summary.to_dict()
    summary_payload["dry_run_overlay_base"] = base_overlay_name
    summary_payload["quarantine_breakdown"] = q_breakdown
    write_json(out / "snapshot_ingestion_summary.json", summary_payload)

    report = build_report(
        summary=summary,
        source_artifacts_read=source_artifacts,
        selected_docs=selected,
        skipped_docs=skipped,
        quarantine=quarantine,
        conflicts=conflicts,
        delta=delta,
        regressions=regressions,
        warnings=warnings,
        errors=errors,
    )
    report_payload = report.to_dict()
    report_payload["summary"] = summary_payload
    report_payload["dry_run_overlay_base"] = base_overlay_name
    report_payload["tainted_sources_excluded_from_delta"] = tainted_sources
    write_json(out / "snapshot_ingestion_report.json", report_payload)

    return {
        "summary": summary_payload,
        "report": report_payload,
        "selected": selected,
        "skipped": skipped,
        "candidates": candidates,
        "overlay_items": overlay_items,
        "delta": delta,
        "quarantine": quarantine,
        "conflicts": conflicts,
        "duplicates_accepted": dup_accepted,
        "duplicates_promoted": dup_promoted,
        "regressions": regressions,
        "out_dir": out,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Wikipedia Snapshot Self-Ingestion v1")
    parser.add_argument("--readiness", default=str(_READINESS))
    parser.add_argument("--manifest", default=str(_MANIFEST))
    parser.add_argument("--accepted-overlay", default=str(_ACCEPTED))
    parser.add_argument("--promoted-overlay", default=str(_PROMOTED))
    parser.add_argument("--out-dir", default=str(_DEFAULT_OUTDIR))
    parser.add_argument("--skip-regressions", action="store_true")
    args = parser.parse_args(argv)

    result = run_snapshot_self_ingestion(
        readiness_path=args.readiness,
        manifest_path=args.manifest,
        accepted_overlay_path=args.accepted_overlay,
        promoted_overlay_path=args.promoted_overlay,
        out_dir=args.out_dir,
        run_regressions=not args.skip_regressions,
    )
    s = result["summary"]
    print("Wikipedia Snapshot Self-Ingestion v1 (proposal only)")
    print(f"  ready_docs_selected          : {s['ready_docs_selected']}")
    print(f"  not_ready_docs_skipped       : {s['not_ready_docs_skipped']}")
    print(f"  ingestion_docs_attempted     : {s['ingestion_docs_attempted']}")
    print(f"  ingestion_docs_succeeded     : {s['ingestion_docs_succeeded']}")
    print(f"  ingestion_docs_failed        : {s['ingestion_docs_failed']}")
    print(f"  candidates_total             : {s['candidates_total']}")
    print(f"  candidates_by_type           : {s['candidates_by_type']}")
    print(f"  overlay_items_total          : {s['overlay_items_total']}")
    print(f"  duplicates accepted/promoted : {s['duplicates_accepted_count']} / {s['duplicates_promoted_count']}")
    print(f"  conflicts_count              : {s['conflicts_count']}")
    print(f"  quarantined_count            : {s['quarantined_count']}")
    print(f"  rejected_count               : {s['rejected_count']}")
    print(f"  safe_delta_items_count       : {s['safe_delta_items_count']}")
    print(f"  dry_run_overlay_base         : {s['dry_run_overlay_base']}")
    print(f"  dry_run_overlay_items_count  : {s['dry_run_overlay_items_count']}")
    print(f"  regressions passed/failed    : {s['regressions_passed_count']} / {s['regressions_failed_count']}")
    print(f"  all_critical_passed          : {s['all_critical_passed']}")
    print(f"  artifacts written to         : {result['out_dir']}")
    print(f"  auto_ingest                  : {s['auto_ingest']}")
    print(f"  auto_promote                 : {s['auto_promote']}")
    print(f"  trusted_memory_modified      : {s['trusted_memory_modified']}")
    print(f"  accepted_overlay_modified    : {s['accepted_overlay_modified']}")
    print(f"  promoted_overlay_modified    : {s['promoted_overlay_modified']}")
    print(f"  runtime_behavior_modified    : {s['runtime_behavior_modified']}")
    print(f"  network_calls                : {s['network_calls']}")
    print(f"  safe_for_general_runtime     : {s['safe_for_general_runtime']}")
    return 0 if s["all_critical_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

