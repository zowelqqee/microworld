"""Turn UI audits into a bounded, proposal-only open-web query frontier.

The feedback loop never treats an audit as evidence.  It only ranks what to
look for next.  Policy-blocked requests are excluded, while low-confidence
relations already found in abstracts are reported separately for review rather
than being recirculated as acquisition targets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from worldpgt.knowledge_pump.audit_gap_analyzer import analyze_gaps
from worldpgt.knowledge_pump.open_web_pump import OpenWebTopic

_EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
_DEFAULT_REVIEW_GLOB = "open_web_pump_v1/campaign_*/open_web_campaign_evidence_grounded_review.json"
_SOURCES = ("openalex", "crossref", "arxiv")


def _norm(value: object) -> str:
    return " ".join(str(value or "").casefold().split())


def _read_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _review_summary(paths: Iterable[Path]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    subject_count = 0
    for path in paths:
        for row in _read_list(path):
            subject_count += 1
            quality = row.get("evidence_quality") if isinstance(row.get("evidence_quality"), dict) else {}
            for issue in quality.get("issues") or []:
                if isinstance(issue, str) and issue:
                    reason_counts[issue] += 1
    return {
        "review_relation_count": subject_count,
        "review_issue_counts": dict(sorted(reason_counts.items())),
    }


def build_open_web_feedback_frontier(
    *,
    output_path: str | Path,
    audit_log_path: str | Path | None = None,
    review_paths: Iterable[str | Path] | None = None,
    period_days: int = 30,
    max_queries: int = 24,
) -> dict[str, Any]:
    """Write an inspectable acquisition frontier from genuine UI knowledge gaps."""
    if max_queries < 1:
        raise ValueError("max_queries must be at least 1")
    if period_days < 0:
        raise ValueError("period_days must be non-negative")
    audit_path = Path(audit_log_path) if audit_log_path is not None else None
    report = analyze_gaps(audit_path, period_days=period_days, require_acquisition_eligibility=False)
    queries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for gap in report.acquisition_candidates:
        entity = str(gap.entity or "").strip()
        key = _norm(entity)
        if not key or key == "[unknown entity]" or key in seen:
            continue
        seen.add(key)
        queries.append({
            "query": entity,
            "bucket": "ui_audit_gap",
            "sources": list(_SOURCES),
            "gap_type": gap.gap_type,
            "audit_count": gap.count,
            "top_reasons": list(gap.top_reasons),
        })
        if len(queries) >= max_queries:
            break
    resolved_review_paths = tuple(
        Path(path) for path in review_paths
    ) if review_paths is not None else tuple(sorted(_EXPERIMENTS.glob(_DEFAULT_REVIEW_GLOB)))
    payload = {
        "proposal_only": True,
        "accepted_memory_modified": False,
        "promoted_overlay_modified": False,
        "safe_for_general_runtime": False,
        "period_days": period_days,
        "audit_event_count": report.total_audit_events,
        "query_count": len(queries),
        "queries": queries,
        "policy_blocked": [entry.to_dict() for entry in report.policy_blocked],
        "review": _review_summary(resolved_review_paths),
        "review_paths": [str(path) for path in resolved_review_paths if path.is_file()],
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def feedback_topics(payload: dict[str, Any]) -> tuple[OpenWebTopic, ...]:
    """Adapt a feedback artifact into ordinary bounded campaign topics."""
    topics: list[OpenWebTopic] = []
    for row in payload.get("queries") or []:
        if not isinstance(row, dict):
            continue
        query = str(row.get("query") or "").strip()
        sources = tuple(str(source) for source in row.get("sources") or () if str(source) in _SOURCES)
        if query and sources:
            topics.append(OpenWebTopic(query, str(row.get("bucket") or "ui_audit_gap"), sources))
    return tuple(topics)
