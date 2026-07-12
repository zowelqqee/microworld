"""Manifest writers for local Wikipedia snapshots."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from worldpgt.wiki_snapshots.snapshot_readiness import evaluate_snapshot_readiness
from worldpgt.wiki_snapshots.types import ManifestRow, PageSnapshot

MANIFEST_FIELDS = [
    "title",
    "normalized_title",
    "pageid",
    "revision_id",
    "source_url",
    "raw_snapshot_path",
    "normalized_doc_path",
    "retrieved_at",
    "raw_text_sha256",
    "fetch_status",
    "ready_for_self_ingestion",
    "requires_quarantine",
    "safe_for_general_runtime",
]


def build_manifest_row(
    snapshot: PageSnapshot,
    raw_snapshot_path: str | Path,
    normalized_doc_path: str | Path,
) -> ManifestRow:
    readiness = evaluate_snapshot_readiness(snapshot, normalized_doc_path)
    return ManifestRow(
        title=snapshot.title,
        normalized_title=snapshot.normalized_title,
        pageid=snapshot.pageid,
        revision_id=snapshot.revision_id,
        source_url=snapshot.source_url,
        raw_snapshot_path=str(raw_snapshot_path),
        normalized_doc_path=str(normalized_doc_path),
        retrieved_at=snapshot.retrieved_at,
        raw_text_sha256=snapshot.raw_text_sha256,
        fetch_status=snapshot.fetch_status,
        ready_for_self_ingestion=readiness.ready_for_self_ingestion,
        requires_quarantine=True,
        safe_for_general_runtime=False,
    )


def write_manifest(rows: list[ManifestRow], json_path: str | Path, csv_path: str | Path) -> None:
    json_out = Path(json_path)
    csv_out = Path(csv_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [row.to_dict() for row in rows]
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in payload:
            writer.writerow(row)

