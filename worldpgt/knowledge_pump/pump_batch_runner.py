"""Bounded network batch runner for Knowledge Pump v1."""

from __future__ import annotations

from pathlib import Path

from worldpgt.knowledge_pump.pump_checkpoint import utc_now
from worldpgt.knowledge_pump.types import ExpandedAllowlistEntry, PumpBatchRecord
from worldpgt.wiki_snapshots.mediawiki_client import MediaWikiClient
from worldpgt.wiki_snapshots.snapshot_manifest import build_manifest_row
from worldpgt.wiki_snapshots.snapshot_normalizer import safe_title_filename, write_normalized_doc
from worldpgt.wiki_snapshots.snapshot_readiness import evaluate_snapshot_readiness
import json


def run_fetch_batch(
    batch_index: int,
    entries: list[ExpandedAllowlistEntry],
    out_dir: str | Path,
    allow_network: bool,
    user_agent: str,
    delay_sec: float = 0.5,
    include_links: bool = True,
) -> tuple[PumpBatchRecord, list[dict]]:
    started = utc_now()
    titles = [e.normalized_title for e in entries]
    raw_dir = Path(out_dir) / "raw_snapshots"
    doc_dir = Path(out_dir) / "normalized_docs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    doc_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    fetched: list[str] = []
    success: list[str] = []
    failed: list[str] = []
    ready = 0
    not_ready = 0
    network_calls = 0

    if not allow_network:
        return (
            PumpBatchRecord(batch_index, titles, [], [], [], 0, 0, 0, started, utc_now(), "planned_no_network"),
            rows,
        )

    client = MediaWikiClient(
        titles, allow_network=True, user_agent=user_agent, delay_sec=delay_sec,
        include_links=include_links,
    )
    for title in titles:
        snapshot = client.fetch_page(title)
        fetched.append(title)
        raw_path = raw_dir / f"{safe_title_filename(snapshot.normalized_title or snapshot.title)}.json"
        raw_path.write_text(json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        doc_path = write_normalized_doc(snapshot, doc_dir)
        row = build_manifest_row(snapshot, raw_path, doc_path).to_dict()
        rows.append(row)
        status = evaluate_snapshot_readiness(snapshot, doc_path)
        if snapshot.fetch_status == "success":
            success.append(title)
        else:
            failed.append(title)
        if status.ready_for_self_ingestion:
            ready += 1
        else:
            not_ready += 1
    network_calls = client.network_calls
    return (
        PumpBatchRecord(batch_index, titles, fetched, success, failed, ready, not_ready, network_calls, started, utc_now(), "completed"),
        rows,
    )
