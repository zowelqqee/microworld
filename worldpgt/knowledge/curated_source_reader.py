"""Loads curated wiki-style snippets from a JSON source file."""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.knowledge.types import CuratedSnippet


class CuratedSourceReader:
    """Reads and validates a curated snippet JSON file."""

    def load(self, path: str | Path) -> list[CuratedSnippet]:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        snippets: list[CuratedSnippet] = []
        for entry in raw:
            snippets.append(
                CuratedSnippet(
                    source_id=entry["source_id"],
                    source_type=entry["source_type"],
                    term=entry["term"],
                    sense=entry["sense"],
                    title=entry["title"],
                    text=entry["text"],
                    source_url=entry.get("source_url"),
                    trust_seed=float(entry["trust_seed"]),
                )
            )
        return snippets
