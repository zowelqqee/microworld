"""Persistence for discovered graph patterns.

Patterns are stored as a plain JSON artifact so the nightly discovery run and
interactive reasoning share one file. The store never merges or mutates
patterns — each save is a full, deterministic snapshot of one discovery run.
"""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.reasoning.types import GraphPattern

DEFAULT_PATTERNS_PATH = (
    Path(__file__).resolve().parent.parent / "artifacts" / "graph_patterns.json"
)


def save_patterns(
    patterns: list[GraphPattern],
    path: str | Path | None = None,
    metadata: dict | None = None,
) -> Path:
    """Write patterns (plus run metadata) to JSON; returns the written path."""
    target = Path(path) if path is not None else DEFAULT_PATTERNS_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": dict(metadata or {}),
        "patterns": [p.to_dict() for p in patterns],
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return target


def load_patterns(path: str | Path | None = None) -> list[GraphPattern]:
    """Load patterns from JSON; missing file → empty list (patterns are
    optional context, never a hard dependency)."""
    source = Path(path) if path is not None else DEFAULT_PATTERNS_PATH
    if not source.exists():
        return []
    payload = json.loads(source.read_text())
    items = payload.get("patterns", payload if isinstance(payload, list) else [])
    return [GraphPattern.from_dict(item) for item in items]


def load_metadata(path: str | Path | None = None) -> dict:
    source = Path(path) if path is not None else DEFAULT_PATTERNS_PATH
    if not source.exists():
        return {}
    payload = json.loads(source.read_text())
    if isinstance(payload, dict):
        return dict(payload.get("metadata") or {})
    return {}
