"""Types for Wikipedia Snapshot Collector v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class PageSnapshot:
    title: str
    normalized_title: str
    pageid: int | None
    revision_id: int | None
    timestamp: str
    source_url: str
    api_url: str
    retrieved_at: str
    raw_text: str
    raw_text_sha256: str
    license_note: str
    fetch_status: str
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ManifestRow:
    title: str
    normalized_title: str
    pageid: int | None
    revision_id: int | None
    source_url: str
    raw_snapshot_path: str
    normalized_doc_path: str
    retrieved_at: str
    raw_text_sha256: str
    fetch_status: str
    ready_for_self_ingestion: bool
    requires_quarantine: bool
    safe_for_general_runtime: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ReadinessResult:
    title: str
    ready_for_self_ingestion: bool
    requires_quarantine: bool
    safe_for_general_runtime: bool
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

