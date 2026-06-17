"""Types for Wikipedia Snapshot Self-Ingestion v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ReadySnapshotDoc:
    title: str
    normalized_title: str
    source_url: str
    retrieved_at: str
    revision_id: int | None
    raw_text_sha256: str
    normalized_doc_path: str
    manifest_row: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotIngestionCandidate:
    candidate_id: str
    source_doc_title: str
    source_doc_hash: str
    item_type: str
    candidate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotDuplicate:
    candidate_id: str
    source_doc_title: str
    duplicate_base: str
    overlay_item: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotConflict:
    candidate_id: str
    source_doc_title: str
    conflict_base: str
    reason: str
    overlay_item: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotQuarantineItem:
    candidate_id: str
    source_doc_title: str
    reason: str
    risk: str
    suggested_action: str
    text: str
    overlay_item: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotOverlayDeltaItem:
    candidate_id: str
    source_doc_title: str
    overlay_item: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotRegressionResult:
    name: str
    status: str
    summary: dict[str, Any] = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotIngestionSummary:
    snapshot_docs_total: int
    ready_docs_selected: int
    not_ready_docs_skipped: int
    ingestion_docs_attempted: int
    ingestion_docs_succeeded: int
    ingestion_docs_failed: int
    candidates_total: int
    candidates_by_type: dict[str, int]
    overlay_items_total: int
    duplicates_accepted_count: int
    duplicates_promoted_count: int
    conflicts_count: int
    quarantined_count: int
    rejected_count: int
    safe_delta_items_count: int
    dry_run_overlay_items_count: int
    regressions_run_count: int
    regressions_passed_count: int
    regressions_failed_count: int
    all_critical_passed: bool
    auto_ingest: bool = False
    auto_promote: bool = False
    trusted_memory_modified: bool = False
    accepted_overlay_modified: bool = False
    promoted_overlay_modified: bool = False
    runtime_behavior_modified: bool = False
    network_calls: bool = False
    safe_for_general_runtime: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SnapshotIngestionReport:
    summary: dict[str, Any]
    source_artifacts_read: list[str]
    selected_docs: list[dict[str, Any]]
    skipped_docs: list[dict[str, Any]]
    candidate_type_breakdown: dict[str, int]
    quarantine_breakdown: dict[str, int]
    conflict_breakdown: dict[str, int]
    safe_delta_examples: list[dict[str, Any]]
    risky_current_volatile_examples: list[dict[str, Any]]
    regression_results: list[dict[str, Any]]
    warnings: list[str]
    errors: list[dict[str, Any]]
    recommended_next_actions: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

