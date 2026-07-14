"""Checkpointed, unattended campaign runner for the open-web proposal pump.

It automates the bounded batch loop; it is not a daemon and never promotes
facts.  A process can be restarted with ``--resume`` after interruption, while
the checkpoint keeps source-rate-limit decisions and completed ranges visible.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable

from worldpgt.knowledge_pump.open_web_pump import (
    BROAD_OPEN_WEB_TOPICS,
    OpenWebTopic,
    build_paged_query_plan,
    consolidate_regated_campaign,
    run_open_web_pump,
)

_CHECKPOINT_NAME = "open_web_campaign_checkpoint.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _load_checkpoint(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def run_open_web_campaign(
    *,
    output_dir: str | Path,
    topics: Iterable[OpenWebTopic] = BROAD_OPEN_WEB_TOPICS,
    batch_size: int = 18,
    records_per_query: int = 10,
    pages_per_query: int = 1,
    page_start: int = 0,
    request_delay_sec: float = 0.5,
    allow_network: bool = False,
    resume: bool = True,
    max_segments: int | None = None,
    run_batch: Callable[..., dict[str, Any]] = run_open_web_pump,
    consolidate: Callable[[str | Path], dict[str, Any]] = consolidate_regated_campaign,
) -> dict[str, Any]:
    """Run every source-query range automatically, persisting after each batch."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if records_per_query < 1:
        raise ValueError("records_per_query must be at least 1")
    if records_per_query > 200:
        raise ValueError("records_per_query must be at most 200 for the OpenAlex source")
    if pages_per_query < 1:
        raise ValueError("pages_per_query must be at least 1")
    if page_start < 0:
        raise ValueError("page_start must be non-negative")
    if records_per_query * (page_start + pages_per_query) > 10_000:
        raise ValueError("records_per_query * (page_start + pages_per_query) must not exceed 10,000 for Crossref offset paging")
    if request_delay_sec < 0:
        raise ValueError("request_delay_sec must be non-negative")
    if max_segments is not None and max_segments < 1:
        raise ValueError("max_segments must be at least 1 when provided")

    root = Path(output_dir)
    checkpoint_path = root / _CHECKPOINT_NAME
    plan_total = len(build_paged_query_plan(topics, pages_per_query=pages_per_query, page_start=page_start))
    existing = _load_checkpoint(checkpoint_path) if resume else None
    if existing and (
        int(existing.get("plan_total", -1)) != plan_total
        or int(existing.get("pages_per_query", 1)) != pages_per_query
        or int(existing.get("page_start", 0)) != page_start
    ):
        raise ValueError("campaign checkpoint belongs to a different query plan")
    state: dict[str, Any] = existing or {
        "proposal_only": True,
        "accepted_memory_modified": False,
        "promoted_overlay_modified": False,
        "runtime_behavior_modified": False,
        "plan_total": plan_total,
        "batch_size": batch_size,
        "records_per_query": records_per_query,
        "pages_per_query": pages_per_query,
        "page_start": page_start,
        "request_delay_sec": request_delay_sec,
        "next_query": 0,
        "disabled_sources": [],
        "segments": [],
        "status": "initialized",
    }

    if not allow_network:
        state["status"] = "planned_no_network"
        _write_json_atomic(checkpoint_path, state)
        return state

    segments_run = 0
    while int(state["next_query"]) < plan_total:
        if max_segments is not None and segments_run >= max_segments:
            state["status"] = "paused_max_segments"
            _write_json_atomic(checkpoint_path, state)
            return state
        start_query = int(state["next_query"])
        query_span = min(batch_size, plan_total - start_query)
        segment_dir = root / f"segment_{start_query:03d}"
        result = run_batch(
            output_dir=segment_dir,
            topics=topics,
            start_query=start_query,
            max_queries=query_span,
            records_per_query=records_per_query,
            pages_per_query=pages_per_query,
            page_start=page_start,
            request_delay_sec=request_delay_sec,
            allow_network=True,
            skip_sources=tuple(state.get("disabled_sources") or ()),
        )
        collection = result.get("collection") if isinstance(result.get("collection"), dict) else {}
        disabled = set(str(source) for source in state.get("disabled_sources") or ())
        disabled.update(str(source) for source in collection.get("rate_limited_sources") or ())
        for error in collection.get("errors") or []:
            if isinstance(error, dict) and "429" in str(error.get("error") or ""):
                disabled.add(str(error.get("source") or ""))
        disabled.discard("")
        state["disabled_sources"] = sorted(disabled)
        state["segments"].append({
            "start_query": start_query,
            "query_span": query_span,
            "output_dir": str(segment_dir),
            "records_total": int(collection.get("records_total") or 0),
            "proposal_item_count": int((result.get("proposal") or {}).get("proposal_item_count") or 0),
            "exploratory_relation_item_count": int((result.get("proposal") or {}).get("exploratory_relation_item_count") or 0),
            "rate_limited_sources": list(collection.get("rate_limited_sources") or ()),
        })
        state["next_query"] = start_query + query_span
        state["status"] = "running"
        _write_json_atomic(checkpoint_path, state)
        segments_run += 1

    campaign = consolidate(root)
    state["campaign"] = campaign
    state["status"] = "completed"
    _write_json_atomic(checkpoint_path, state)
    return state
