"""Load only readiness-approved local Wikipedia snapshot docs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc

SELECTION_FIELDS = [
    "title",
    "normalized_title",
    "source_url",
    "retrieved_at",
    "revision_id",
    "raw_text_sha256",
    "normalized_doc_path",
]


def _by_title(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in rows:
        for key in ("title", "normalized_title"):
            title = row.get(key) or ""
            if title:
                out[str(title)] = row
    return out


def load_ready_snapshot_docs(
    readiness_path: str | Path,
    manifest_path: str | Path,
) -> tuple[list[ReadySnapshotDoc], list[dict[str, Any]]]:
    readiness_file = Path(readiness_path)
    manifest_file = Path(manifest_path)
    readiness = json.loads(readiness_file.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    manifest_by_title = _by_title(manifest)

    selected: list[ReadySnapshotDoc] = []
    skipped: list[dict[str, Any]] = []

    for item in readiness:
        title = str(item.get("title", ""))
        row = manifest_by_title.get(title)
        if row is None:
            skipped.append({"title": title, "reasons": ["manifest_row_missing"]})
            continue
        reasons = list(item.get("reasons") or [])
        if item.get("ready_for_self_ingestion") is not True:
            skipped.append({"title": title, "reasons": reasons or ["not_ready"]})
            continue
        path = Path(str(row.get("normalized_doc_path", "")))
        missing = []
        if not path.is_file():
            missing.append("normalized_doc_missing")
        for key in ("source_url", "retrieved_at", "raw_text_sha256"):
            if not row.get(key):
                missing.append(f"{key}_missing")
        if row.get("fetch_status") != "success":
            missing.append(f"fetch_status:{row.get('fetch_status')}")
        if missing:
            skipped.append({"title": title, "reasons": missing})
            continue
        selected.append(
            ReadySnapshotDoc(
                title=str(row.get("title") or title),
                normalized_title=str(row.get("normalized_title") or title),
                source_url=str(row.get("source_url")),
                retrieved_at=str(row.get("retrieved_at")),
                revision_id=row.get("revision_id"),
                raw_text_sha256=str(row.get("raw_text_sha256")),
                normalized_doc_path=str(path),
                manifest_row=dict(row),
            )
        )
    return selected, skipped


def write_selection(
    docs: list[ReadySnapshotDoc],
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_out = Path(json_path)
    csv_out = Path(csv_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [doc.to_dict() for doc in docs]
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SELECTION_FIELDS)
        writer.writeheader()
        for doc in docs:
            row = doc.to_dict()
            writer.writerow({field: row.get(field, "") for field in SELECTION_FIELDS})
