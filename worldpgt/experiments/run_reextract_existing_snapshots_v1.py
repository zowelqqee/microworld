"""Offline re-extraction over existing Knowledge Pump snapshots.

This runner does not fetch, promote, or mutate accepted memory. It reads the
local ``batch_snapshots`` raw/normalized docs, re-runs extraction in chunks, and
writes a separate artifact family for monitoring long runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from worldpgt.knowledge_pump.extraction_yield_v2 import (
    extract_yield_v2,
    write_extraction_yield_v2_artifacts,
)
from worldpgt.knowledge_pump.precision_cleanup_v2_1 import apply_precision_cleanup_v2_1
from worldpgt.knowledge_pump.precision_firewall import apply_precision_firewall
from worldpgt.knowledge_pump.precision_firewall_v2 import apply_precision_firewall_v2
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_candidate_extractor import extract_all_candidates
from worldpgt.relation_extraction_v2.relation_candidate_validator import (
    _build_existing_relations,
    _is_duplicate,
    validate_candidates,
)
from worldpgt.wiki_snapshot_ingestion.snapshot_batch_ingestor import ingest_snapshot_batch
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc
from worldpgt.wiki_snapshots.snapshot_normalizer import safe_title_filename
from worldpgt.wiki_snapshots.snapshot_readiness import evaluate_snapshot_readiness
from worldpgt.wiki_snapshots.types import PageSnapshot


_ROOT = Path(__file__).resolve().parents[2]
_EXP = _ROOT / "worldpgt" / "experiments"
_PUMP = _EXP / "knowledge_pump_v1"
_BATCH_SNAPSHOTS = _PUMP / "batch_snapshots"
_ACCEPTED = _EXP / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED = _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
_SNAPSHOT_DRY_RUN = _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
_SNAPSHOTS = _EXP / "wiki_snapshots_v1"
_DEFAULT_OUT = _PUMP / "reextract_existing_v1"


def _read_json(path: Path, default: Any) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key, value in out.items():
                if isinstance(value, (list, dict)):
                    out[key] = json.dumps(value, ensure_ascii=False)
            writer.writerow(out)


def _page_snapshot_from_raw(row: dict[str, Any]) -> PageSnapshot:
    title = str(row.get("normalized_title") or row.get("title") or "")
    return PageSnapshot(
        title=str(row.get("title") or title),
        normalized_title=title,
        pageid=row.get("pageid"),
        revision_id=row.get("revision_id"),
        timestamp=str(row.get("timestamp") or ""),
        source_url=str(row.get("source_url") or ""),
        api_url=str(row.get("api_url") or ""),
        retrieved_at=str(row.get("retrieved_at") or ""),
        raw_text=str(row.get("raw_text") or ""),
        raw_text_sha256=str(row.get("raw_text_sha256") or ""),
        license_note=str(row.get("license_note") or ""),
        fetch_status=str(row.get("fetch_status") or "error"),
        error=str(row.get("error") or ""),
        links=list(row.get("links") or []),
    )


def load_ready_batch_docs(batch_dir: Path) -> tuple[list[ReadySnapshotDoc], list[dict[str, Any]]]:
    raw_dir = batch_dir / "raw_snapshots"
    doc_dir = batch_dir / "normalized_docs"
    docs: list[ReadySnapshotDoc] = []
    skipped: list[dict[str, Any]] = []
    for raw_path in sorted(raw_dir.glob("*.json")):
        row = _read_json(raw_path, {})
        snapshot = _page_snapshot_from_raw(row)
        title = snapshot.normalized_title or snapshot.title
        doc_path = doc_dir / f"{safe_title_filename(title)}.md"
        readiness = evaluate_snapshot_readiness(snapshot, doc_path)
        if not readiness.ready_for_self_ingestion:
            skipped.append({"title": title, "reasons": readiness.reasons})
            continue
        docs.append(
            ReadySnapshotDoc(
                title=snapshot.title,
                normalized_title=title,
                source_url=snapshot.source_url,
                retrieved_at=snapshot.retrieved_at,
                revision_id=snapshot.revision_id,
                raw_text_sha256=snapshot.raw_text_sha256,
                normalized_doc_path=str(doc_path),
                manifest_row=dict(row),
            )
        )
    return docs, skipped


def _chunks(items: list[ReadySnapshotDoc], chunk_size: int) -> list[list[ReadySnapshotDoc]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _count_by_type(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("overlay_type") or "unknown") for item in items))


def _count_by_predicate(items: list[dict[str, Any]]) -> dict[str, int]:
    return dict(Counter(str(item.get("predicate") or "definition") for item in items))


def _load_overlay_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for path in (_ACCEPTED, _PROMOTED, _SNAPSHOT_DRY_RUN):
        data = _read_json(path, [])
        if isinstance(data, list):
            items.extend(data)
    return items


def _apply_precision(items: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    p1 = apply_precision_firewall(items)
    p2 = apply_precision_firewall_v2(p1["accepted"])
    p3 = apply_precision_cleanup_v2_1(p2["accepted"])
    accepted = p3["accepted"]
    return accepted, {
        "precision_v1_before": p1.get("answerable_before_count", len(items)),
        "precision_v1_after": len(p1["accepted"]),
        "precision_v1_rejected": len(p1.get("rejected", [])),
        "precision_v1_quarantined": len(p1.get("quarantine", [])),
        "precision_v2_before": p2.get("answerable_before_count", len(p1["accepted"])),
        "precision_v2_after": len(p2["accepted"]),
        "precision_v2_rejected": len(p2.get("rejected", [])),
        "precision_v2_quarantined": len(p2.get("quarantine", [])),
        "precision_cleanup_before": p3.get("answerable_before_count", len(p2["accepted"])),
        "precision_cleanup_after": len(accepted),
        "precision_cleanup_rejected": len(p3.get("rejected", [])),
        "precision_cleanup_quarantined": len(p3.get("quarantine", [])),
    }


class _DictCandidate:
    """Adapts an overlay-shaped dict to the (subject, relation, object)
    attribute interface ``_is_duplicate`` expects, so its already-reviewed
    inverse-predicate-alias table (founded/founded_by, owns/owned_by, ...)
    can be reused here instead of re-deriving it."""

    def __init__(self, item: dict[str, Any]) -> None:
        self.subject = str(item.get("subject") or "")
        self.relation = str(item.get("predicate") or "")
        self.object = str(item.get("object") or item.get("definition") or "")


def _drop_duplicates_of_existing_overlay(
    items: list[dict[str, Any]],
    base_overlay_items: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Filter out any overlay_relation candidate that only restates a triple
    (or its inverse-predicate form) already present in the base overlay --
    e.g. a fresh "SpaceX founded_by Elon Musk" extraction when "Elon Musk
    founded SpaceX" is already trusted memory. ``extract_yield_v2`` candidates
    never go through ``validate_candidates``'s own duplicate check (that path
    only sees ``relation_candidate_extractor`` output), so without this pass
    the same fact can re-enter as a second sentence under a different
    predicate spelling."""

    existing = _build_existing_relations(base_overlay_items)
    kept: list[dict[str, Any]] = []
    dropped = 0
    for item in items:
        if item.get("overlay_type") == "overlay_relation" and _is_duplicate(
            _DictCandidate(item), existing
        ):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def _dedupe_answerable(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        otype = str(item.get("overlay_type") or "")
        if otype == "overlay_relation":
            key = (
                otype,
                str(item.get("subject") or "").casefold(),
                str(item.get("predicate") or "").casefold(),
                str(item.get("object") or "").casefold(),
            )
        elif otype == "overlay_definition":
            key = (
                otype,
                str(item.get("subject") or "").casefold(),
                "is_a",
                str(item.get("definition") or "").casefold(),
            )
        else:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def run(
    *,
    batch_dir: Path = _BATCH_SNAPSHOTS,
    out_dir: Path = _DEFAULT_OUT,
    limit: int | None = None,
    chunk_size: int = 100,
    max_sentences_per_doc: int = 50,
    enable_spacy: bool = False,
) -> dict[str, Any]:
    start = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    progress_path = out_dir / "progress.jsonl"
    if progress_path.exists():
        progress_path.unlink()

    docs, skipped = load_ready_batch_docs(batch_dir)
    docs_total = len(docs)
    if limit is not None:
        docs = docs[:limit]
    chunks = _chunks(docs, chunk_size)
    _write_json(out_dir / "selected_docs.json", [doc.to_dict() for doc in docs])
    _write_json(out_dir / "skipped_docs_sample.json", skipped[:200])

    index = EntitySurfaceIndex(
        _ACCEPTED,
        _PROMOTED,
        _SNAPSHOT_DRY_RUN,
        _SNAPSHOTS / "snapshot_manifest.json",
    )
    base_overlay_items = _load_overlay_items()

    all_answerable: list[dict[str, Any]] = []
    all_v2: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, Any]] = []
    totals = Counter()

    for chunk_index, chunk_docs in enumerate(chunks, start=1):
        chunk_start = time.time()
        chunk_dir = out_dir / "chunks" / f"chunk_{chunk_index:04d}"
        titles = [doc.normalized_title for doc in chunk_docs]
        event = {
            "event": "chunk_start",
            "chunk_index": chunk_index,
            "chunk_count": len(chunks),
            "docs_in_chunk": len(chunk_docs),
            "docs_done_before": (chunk_index - 1) * chunk_size,
            "titles_sample": titles[:5],
        }
        print(
            f"[reextract] chunk {chunk_index}/{len(chunks)} "
            f"docs={len(chunk_docs)} sample={titles[:3]}",
            flush=True,
        )
        _append_jsonl(progress_path, event)

        candidates, overlay_items, failures, by_type, doc_status = ingest_snapshot_batch(chunk_docs)
        raw_relations, sentences, errors = extract_all_candidates(
            chunk_docs, index, enable_spacy=enable_spacy
        )
        safe_relations, quarantine, duplicate_ids, conflict_ids = validate_candidates(
            raw_relations, index, base_overlay_items
        )
        relation_rows = [candidate.to_dict() for candidate in safe_relations]
        v2_items, v2_stats = extract_yield_v2(
            chunk_docs,
            index,
            max_sentences_per_doc=max_sentences_per_doc,
        )
        answerable_input = [
            item
            for item in [*overlay_items, *relation_rows, *v2_items]
            if item.get("overlay_type") in {"overlay_relation", "overlay_definition"}
        ]
        accepted, precision_summary = _apply_precision(answerable_input)
        accepted = _dedupe_answerable(accepted)

        all_answerable.extend(accepted)
        all_v2.extend(v2_items)
        elapsed = round(time.time() - chunk_start, 2)
        row = {
            "chunk_index": chunk_index,
            "docs": len(chunk_docs),
            "ingestion_candidates": len(candidates),
            "snapshot_overlay_items": len(overlay_items),
            "raw_relation_candidates": len(raw_relations),
            "safe_relation_candidates": len(safe_relations),
            "relation_quarantine": len(quarantine),
            "relation_duplicates": len(duplicate_ids),
            "relation_conflicts": len(conflict_ids),
            "sentences_scanned": sentences,
            "v2_items": len(v2_items),
            "accepted_answerable": len(accepted),
            "accepted_by_type": _count_by_type(accepted),
            "accepted_by_predicate": _count_by_predicate(accepted),
            "doc_status": doc_status,
            "ingestion_by_type": by_type,
            "failures": len(failures),
            "errors": errors[:10],
            "elapsed_sec": elapsed,
            **precision_summary,
        }
        chunk_rows.append(row)
        totals.update({
            "docs_processed": len(chunk_docs),
            "ingestion_candidates": len(candidates),
            "snapshot_overlay_items": len(overlay_items),
            "raw_relation_candidates": len(raw_relations),
            "safe_relation_candidates": len(safe_relations),
            "relation_quarantine": len(quarantine),
            "relation_duplicates": len(duplicate_ids),
            "relation_conflicts": len(conflict_ids),
            "sentences_scanned": sentences,
            "v2_items": len(v2_items),
            "accepted_answerable_before_global_dedupe": len(accepted),
            "failures": len(failures),
        })
        _write_json(chunk_dir / "summary.json", row)
        _write_json(chunk_dir / "accepted_answerable.json", accepted)
        _write_json(chunk_dir / "v2_items.json", v2_items)
        _write_json(chunk_dir / "safe_relation_candidates.json", relation_rows)

        global_accepted = _dedupe_answerable(all_answerable)
        summary = _summary(
            docs_total=docs_total,
            docs_selected=len(docs),
            chunks_total=len(chunks),
            chunks_done=chunk_index,
            totals=dict(totals),
            accepted=global_accepted,
            started_at=start,
            network_calls=False,
        )
        _write_json(out_dir / "summary.json", summary)
        _append_jsonl(progress_path, {"event": "chunk_done", **row})
        print(
            f"[reextract] chunk {chunk_index} done: "
            f"accepted={len(accepted)} global={len(global_accepted)} "
            f"elapsed={elapsed}s",
            flush=True,
        )

    accepted_global = _dedupe_answerable(all_answerable)
    accepted_global, existing_overlay_duplicates = _drop_duplicates_of_existing_overlay(
        accepted_global, base_overlay_items
    )
    _write_json(out_dir / "reextract_answerable_delta.json", accepted_global)
    _write_csv(
        out_dir / "chunk_summary.csv",
        chunk_rows,
        [
            "chunk_index",
            "docs",
            "ingestion_candidates",
            "snapshot_overlay_items",
            "raw_relation_candidates",
            "safe_relation_candidates",
            "relation_quarantine",
            "relation_duplicates",
            "relation_conflicts",
            "sentences_scanned",
            "v2_items",
            "accepted_answerable",
            "elapsed_sec",
        ],
    )
    write_extraction_yield_v2_artifacts(
        out_dir / "extraction_yield_v2",
        all_v2,
        {"network_calls": False},
    )
    final_summary = _summary(
        docs_total=docs_total,
        docs_selected=len(docs),
        chunks_total=len(chunks),
        chunks_done=len(chunks),
        totals=dict(totals),
        accepted=accepted_global,
        started_at=start,
        network_calls=False,
    )
    final_summary["status"] = "complete"
    final_summary["existing_overlay_duplicates_dropped"] = existing_overlay_duplicates
    _write_json(out_dir / "summary.json", final_summary)
    _append_jsonl(progress_path, {"event": "complete", **final_summary})
    return final_summary


def _summary(
    *,
    docs_total: int,
    docs_selected: int,
    chunks_total: int,
    chunks_done: int,
    totals: dict[str, Any],
    accepted: list[dict[str, Any]],
    started_at: float,
    network_calls: bool,
) -> dict[str, Any]:
    by_type = _count_by_type(accepted)
    by_predicate = _count_by_predicate(accepted)
    elapsed = round(time.time() - started_at, 2)
    return {
        "status": "running" if chunks_done < chunks_total else "complete",
        "network_calls": network_calls,
        "docs_total_ready": docs_total,
        "docs_selected": docs_selected,
        "chunks_total": chunks_total,
        "chunks_done": chunks_done,
        "docs_processed": int(totals.get("docs_processed", 0)),
        "answerable_after_precision": len(accepted),
        "relation_count": by_type.get("overlay_relation", 0),
        "definition_count": by_type.get("overlay_definition", 0),
        "accepted_by_predicate": by_predicate,
        "totals": totals,
        "elapsed_sec": elapsed,
        "avg_docs_per_sec": round(totals.get("docs_processed", 0) / elapsed, 4) if elapsed else 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=_BATCH_SNAPSHOTS)
    parser.add_argument("--out-dir", type=Path, default=_DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--max-sentences-per-doc", type=int, default=50)
    parser.add_argument("--enable-spacy", action="store_true", default=False)
    args = parser.parse_args(argv)

    summary = run(
        batch_dir=args.batch_dir,
        out_dir=args.out_dir,
        limit=args.limit,
        chunk_size=args.chunk_size,
        max_sentences_per_doc=args.max_sentences_per_doc,
        enable_spacy=args.enable_spacy,
    )
    for key in (
        "status",
        "network_calls",
        "docs_total_ready",
        "docs_processed",
        "chunks_done",
        "chunks_total",
        "answerable_after_precision",
        "relation_count",
        "definition_count",
        "elapsed_sec",
    ):
        print(f"{key}: {summary.get(key)}")
    print(f"out_dir: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
