"""Adapt normalized snapshot markdown docs into WikiIngestionV2 page records."""

from __future__ import annotations

import re
from pathlib import Path

from worldpgt.knowledge.wiki_ingestion_v2_types import WikiLink, WikiPageRecord, WikiSource
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc

_HEADER_RE = re.compile(r"^([^:]+):\s*(.*)$")
_SECTION_RE = re.compile(r"^=+\s*.+?\s*=+$|^#+\s+.+$")


def split_snapshot_doc(text: str) -> tuple[dict[str, str], str]:
    lines = text.splitlines()
    header: dict[str, str] = {}
    body_start = 0
    for idx, line in enumerate(lines):
        if idx == 0 and line.startswith("# "):
            header["title"] = line[2:].strip()
            continue
        if not line.strip() and idx > 0:
            maybe_next = lines[idx + 1] if idx + 1 < len(lines) else ""
            if not _HEADER_RE.match(maybe_next):
                body_start = idx + 1
                break
            continue
        match = _HEADER_RE.match(line)
        if match:
            header[match.group(1).strip().lower()] = match.group(2).strip()
            continue
    body = "\n".join(lines[body_start:]).strip()
    return header, body


def first_paragraph(body: str) -> str:
    chunks: list[str] = []
    for line in body.splitlines():
        stripped = line.strip()
        if not stripped:
            if chunks:
                break
            continue
        if _SECTION_RE.match(stripped):
            if chunks:
                break
            continue
        chunks.append(stripped)
    return " ".join(chunks).strip()


def extract_mentions(text: str, titles: list[str], self_title: str, max_links: int = 20) -> list[WikiLink]:
    low = f" {text.lower()} "
    self_norm = self_title.lower()
    links: list[WikiLink] = []
    seen: set[str] = set()
    for title in sorted(titles, key=len, reverse=True):
        norm = title.lower()
        if norm == self_norm or norm in seen or len(norm) < 3:
            continue
        if re.search(r"(?<!\w)" + re.escape(norm) + r"(?!\w)", low):
            links.append(WikiLink(surface=title, target=title))
            seen.add(norm)
        if len(links) >= max_links:
            break
    return links


def adapt_snapshot_doc(doc: ReadySnapshotDoc, known_titles: list[str] | None = None) -> WikiPageRecord:
    text = Path(doc.normalized_doc_path).read_text(encoding="utf-8")
    _header, body = split_snapshot_doc(text)
    lead = first_paragraph(body)
    if not lead:
        raise ValueError(f"snapshot doc has no body text: {doc.title}")
    links = extract_mentions((lead + "\n" + body[:2000]), known_titles or [], doc.normalized_title)
    source = WikiSource(
        source_id=f"snapshot:{doc.raw_text_sha256[:16]}",
        source_type="local_wikipedia_snapshot",
        retrieved_at=doc.retrieved_at,
        source_url=doc.source_url,
    )
    return WikiPageRecord(
        page_id=f"snapshot:{doc.raw_text_sha256[:16]}",
        title=doc.normalized_title or doc.title,
        lead_paragraph=lead,
        source=source,
        wikidata_id=None,
        entity_type_hint=None,
        infobox={},
        links=links,
        categories=["local_wikipedia_snapshot"],
    )

