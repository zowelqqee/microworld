"""Local document reader for Wikipedia Self-Ingestion v1.

Reads local text / JSON documents referenced by validated manifest sources and
produces normalized document objects. Strictly local file I/O — no network.
"""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.self_ingestion.types import NormalizedDocument, SourceEntry


def _title_from(raw_text: str, fallback: str) -> str:
    for line in raw_text.splitlines():
        s = line.strip()
        if s.upper().startswith("TITLE:"):
            return s.split(":", 1)[1].strip()
    return fallback


def _chunks(raw_text: str) -> list[str]:
    # Split on blank lines into coarse chunks; keep non-empty.
    blocks, cur = [], []
    for line in raw_text.splitlines():
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur).strip())
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur).strip())
    return [b for b in blocks if b]


def read_document(source: SourceEntry) -> NormalizedDocument:
    doc = NormalizedDocument(
        source_id=source.source_id,
        title=source.source_id,
        raw_text="",
        source_type=source.source_type,
        trust_tier=source.trust_tier,
        source_path=source.path,
    )
    try:
        raw_text = Path(source.path).read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - manifest already checked existence
        doc.read_errors.append(f"read_failed:{exc}")
        return doc

    doc.raw_text = raw_text

    if source.source_type == "local_wikipedia_like_json":
        try:
            data = json.loads(raw_text)
            doc.title = str(data.get("title", source.source_id))
            doc.chunks = [raw_text]
        except json.JSONDecodeError as exc:
            doc.read_errors.append(f"json_decode_error:{exc}")
            doc.title = source.source_id
            doc.chunks = _chunks(raw_text)
    else:
        doc.title = _title_from(raw_text, source.source_id)
        doc.chunks = _chunks(raw_text)

    return doc


def read_documents(sources: list[SourceEntry]) -> list[NormalizedDocument]:
    return [read_document(s) for s in sources]
