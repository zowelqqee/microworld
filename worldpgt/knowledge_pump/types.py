"""Dataclasses for Knowledge Pump 5000 Mode v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class FrontierTitle:
    title: str
    source: str
    reason: str
    weight: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExpandedAllowlistEntry:
    title: str
    normalized_title: str
    priority: int
    reason: str
    source: str
    risk_hint: str
    already_fetched: bool
    selected_for_batch: bool
    batch_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PumpCheckpoint:
    target_total: int
    batch_size: int
    current_batch_index: int
    selected_titles_total: int
    already_fetched_titles: list[str] = field(default_factory=list)
    successful_fetch_titles: list[str] = field(default_factory=list)
    failed_fetch_titles: list[str] = field(default_factory=list)
    ready_titles: list[str] = field(default_factory=list)
    not_ready_titles: list[str] = field(default_factory=list)
    last_completed_stage: str = "initialized"
    last_run_at: str = ""
    can_resume: bool = True
    safe_to_continue: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PumpBatchRecord:
    batch_index: int
    titles_planned: list[str]
    titles_fetched: list[str]
    fetch_success: list[str]
    fetch_failed: list[str]
    ready_count: int
    not_ready_count: int
    network_calls: int
    started_at: str
    finished_at: str
    status: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

