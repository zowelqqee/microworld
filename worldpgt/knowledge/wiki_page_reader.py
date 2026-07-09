"""Reads curated Wikipedia-style page records from a local JSON fixture.

Read-only. Performs no network access. Input must be a local file path.
"""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.knowledge.wiki_ingestion_v2_types import (
    WikiLink,
    WikiPageRecord,
    WikiSource,
)


class WikiPageReader:
    """Loads and validates curated page-like JSON records."""

    def load(self, path: str | Path) -> list[WikiPageRecord]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("curated wiki fixture must be a JSON list of pages")
        pages: list[WikiPageRecord] = []
        for entry in raw:
            pages.append(self._parse_page(entry))
        return pages

    def _parse_page(self, entry: dict) -> WikiPageRecord:
        src_raw = entry.get("source") or {}
        source = WikiSource(
            source_id=src_raw["source_id"],
            source_type=src_raw["source_type"],
            retrieved_at=src_raw["retrieved_at"],
            source_url=src_raw.get("source_url"),
        )
        links = [
            WikiLink(surface=l["surface"], target=l["target"])
            for l in entry.get("links", [])
        ]
        return WikiPageRecord(
            page_id=entry["page_id"],
            title=entry["title"],
            lead_paragraph=entry["lead_paragraph"],
            source=source,
            wikidata_id=entry.get("wikidata_id"),
            entity_type_hint=entry.get("entity_type_hint"),
            infobox=dict(entry.get("infobox", {})),
            links=links,
            categories=list(entry.get("categories", [])),
        )
