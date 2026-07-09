"""Batch ingestion over readiness-approved local snapshot docs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from worldpgt.knowledge.wiki_candidate_overlay_builder import (
    WikiCandidateOverlayBuilder,
    as_dict as overlay_as_dict,
)
from worldpgt.knowledge.wiki_ingestion_v2 import WikiIngestionV2, as_dict as candidate_as_dict
from worldpgt.wiki_snapshot_ingestion.snapshot_page_adapter import adapt_snapshot_doc
from worldpgt.wiki_snapshot_ingestion.types import (
    ReadySnapshotDoc,
    SnapshotIngestionCandidate,
    SnapshotQuarantineItem,
)


def _with_provenance(payload: dict[str, Any], doc: ReadySnapshotDoc) -> dict[str, Any]:
    out = dict(payload)
    out["snapshot_source_title"] = doc.normalized_title
    out["snapshot_source_url"] = doc.source_url
    out["snapshot_retrieved_at"] = doc.retrieved_at
    out["snapshot_revision_id"] = doc.revision_id
    out["snapshot_raw_text_sha256"] = doc.raw_text_sha256
    out["safe_for_general_runtime"] = False
    return out


def ingest_snapshot_batch(
    docs: list[ReadySnapshotDoc],
) -> tuple[list[SnapshotIngestionCandidate], list[dict[str, Any]], list[SnapshotQuarantineItem], dict[str, int], dict[str, int]]:
    known_titles = [doc.normalized_title for doc in docs]
    ingestion = WikiIngestionV2()
    builder = WikiCandidateOverlayBuilder()
    candidates: list[SnapshotIngestionCandidate] = []
    overlay_items: list[dict[str, Any]] = []
    failures: list[SnapshotQuarantineItem] = []
    doc_status = {"attempted": len(docs), "succeeded": 0, "failed": 0}
    by_type: Counter[str] = Counter()

    for doc_idx, doc in enumerate(docs):
        try:
            page = adapt_snapshot_doc(doc, known_titles=known_titles)
            result = ingestion.run([page])
            cand_dicts = []
            for cand_idx, cand in enumerate(result.candidates):
                cdict = _with_provenance(candidate_as_dict(cand), doc)
                item_type = str(cdict.get("item_type", "unknown"))
                by_type[item_type] += 1
                candidate_id = f"snap-{doc_idx:03d}-{cand_idx:03d}"
                candidates.append(
                    SnapshotIngestionCandidate(
                        candidate_id=candidate_id,
                        source_doc_title=doc.normalized_title,
                        source_doc_hash=doc.raw_text_sha256,
                        item_type=item_type,
                        candidate=cdict,
                    )
                )
                cand_dicts.append(cdict)
            built = builder.build(cand_dicts)
            for built_idx, item in enumerate(built.items):
                odict = _with_provenance(overlay_as_dict(item), doc)
                odict["snapshot_candidate_id"] = f"snap-{doc_idx:03d}-overlay-{built_idx:03d}"
                overlay_items.append(odict)
            doc_status["succeeded"] += 1
        except Exception as exc:
            doc_status["failed"] += 1
            failures.append(
                SnapshotQuarantineItem(
                    candidate_id=f"snap-{doc_idx:03d}-doc-error",
                    source_doc_title=doc.normalized_title,
                    reason="snapshot_doc_ingestion_error",
                    risk="medium",
                    suggested_action="human_review",
                    text=str(exc),
                    overlay_item={},
                )
            )

    return candidates, overlay_items, failures, dict(sorted(by_type.items())), doc_status


def write_candidates(
    candidates: list[SnapshotIngestionCandidate],
    json_path: str | Path,
    csv_path: str | Path,
) -> None:
    json_out = Path(json_path)
    csv_out = Path(csv_path)
    json_out.parent.mkdir(parents=True, exist_ok=True)
    payload = [c.to_dict() for c in candidates]
    json_out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    fields = ["candidate_id", "source_doc_title", "source_doc_hash", "item_type"]
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for c in candidates:
            writer.writerow({field: getattr(c, field) for field in fields})

