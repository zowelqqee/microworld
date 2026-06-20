"""Analyze accumulated audit events and classify them into knowledge gaps.

Reads the JSONL audit log, filters to a time window, and returns a
GapReport that separates acquisition targets from policy-blocked cases.
Policy-blocked entities are not sent to the frontier.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from worldpgt.knowledge.staleness_detector import detect_stale_candidates
from worldpgt.knowledge_pump.audit_types import EntityGapEntry, GapReport

_DEFAULT_LOG = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "knowledge_pump_v1"
    / "audit_log.jsonl"
)
_EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"
_PUMP_DIR = _EXPERIMENTS / "knowledge_pump_v1"

_DEFAULT_OVERLAY_INDEX_PATHS = (
    _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json",
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
    _PUMP_DIR / "pump_dry_run_overlay.json",
)

_DEFAULT_INGESTION_INDEX_PATHS = (
    _EXPERIMENTS / "wiki_snapshots_v1" / "snapshot_manifest.json",
    _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_ingestion_candidates.json",
    _PUMP_DIR / "pump_fresh_ingestion_candidates.json",
    _PUMP_DIR / "pump_fresh_relation_candidates.json",
    _PUMP_DIR / "pump_fresh_relation_candidates_v2.json",
    _PUMP_DIR / "pump_fresh_relation_candidates_merged.json",
    _PUMP_DIR / "pump_safe_delta.json",
)

# support_kind values that always mean a policy block, never an acquisition gap.
_POLICY_SUPPORT_KINDS = frozenset({"audit_blocked_context"})

# Keywords inside multihop audit_reason that indicate policy, not missing data.
_POLICY_REASON_KEYWORDS = frozenset({
    "volatile_hop",
    "high_risk_hop",
    "transitive_founder_leak",
    "transitive_owner_leak",
    "direct_relation_required",
    "current_sensitive",
    "relation_inversion",
    "unsupported_universal",
    "private",
})

# Substrings in the reason text that reveal a policy block regardless of support_kind.
# Matches "current/live data", "live data", "volatile/live", etc.
_POLICY_REASON_SUBSTRINGS = (
    "current/live data",
    "live data",
    "volatile/live",
    "asks for current",
    "must not fabricate",
    "policy",
    "blocked",
    "universal generalization",
    "reversed or unsupported",
    "private/sensitive",
)

# Substrings that indicate the entity was present but the specific fact is missing.
_MISSING_FACT_SUBSTRINGS = (
    "no stable relation",
    "no explicit stable path",
    "no stable fact",
    "relation is missing",
)

_TOP_REASONS_LIMIT = 3
_SYNTHETIC_MARKERS = frozenset({
    "fictional",
    "synthetic",
    "test",
    "fixture",
    "mock",
    "benchmark_unsupported",
})


def _norm(value: Any) -> str:
    return " ".join(str(value or "").strip().casefold().split())


def _strip_article(value: str) -> str:
    normed = _norm(value)
    return normed[4:] if normed.startswith("the ") else normed


def _entity_keys(entity: str) -> set[str]:
    normed = _norm(entity)
    if not normed or normed == "[unknown entity]":
        return set()
    keys = {normed}
    without_article = _strip_article(entity)
    if without_article:
        keys.add(without_article)
    return keys


def _load_json_list(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict)]


def _add_value(index: set[str], value: Any) -> None:
    text = _norm(value)
    if text:
        index.add(text)
        index.add(_strip_article(text))


def _build_overlay_source_index(paths: tuple[Path, ...] = _DEFAULT_OVERLAY_INDEX_PATHS) -> set[str]:
    """Return entities that have source-page-backed overlay evidence."""

    index: set[str] = set()
    for path in paths:
        for row in _load_json_list(path):
            source_page = row.get("source_page") or row.get("snapshot_source_title")
            if not source_page:
                continue
            _add_value(index, row.get("label"))
            _add_value(index, row.get("subject"))
            _add_value(index, source_page)
    return {value for value in index if value}


def _build_ingestion_seen_index(paths: tuple[Path, ...] = _DEFAULT_INGESTION_INDEX_PATHS) -> set[str]:
    """Return entities/pages seen by the snapshot or pump ingestion pipeline."""

    index: set[str] = set()
    for path in paths:
        for row in _load_json_list(path):
            candidate = row.get("candidate")
            if isinstance(candidate, dict):
                for key in ("label", "subject", "source_page", "snapshot_source_title"):
                    _add_value(index, candidate.get(key))
            for key in (
                "title",
                "normalized_title",
                "source_doc_title",
                "source_title",
                "source_page",
                "snapshot_source_title",
                "subject",
            ):
                _add_value(index, row.get(key))
    return {value for value in index if value}


def _contains_synthetic_marker(value: Any) -> bool:
    text = _norm(value)
    return any(marker in text for marker in _SYNTHETIC_MARKERS)


def _entry_marked_synthetic(entry: dict) -> bool:
    values: list[Any] = [
        entry.get("source"),
        entry.get("source_system"),
        entry.get("category"),
        entry.get("benchmark_category"),
        entry.get("audit_source"),
    ]
    tags = entry.get("tags") or entry.get("source_tags") or entry.get("flags")
    if isinstance(tags, list):
        values.extend(tags)
    elif tags:
        values.append(tags)
    return any(_contains_synthetic_marker(value) for value in values)


def _eligible_for_acquisition(
    entity: str,
    entries: list[dict],
    *,
    overlay_source_entities: set[str],
    ingestion_seen_entities: set[str],
) -> bool:
    """Return whether an entity should be allowed into production acquisition.

    Production gap reports should point the pump at real, source-resolvable
    entities with missing facts. Synthetic/unsupported benchmark entities remain
    useful audit rows, but they should not become frontier targets.
    """

    keys = _entity_keys(entity)
    if not keys:
        return False
    if any(_entry_marked_synthetic(entry) for entry in entries):
        return False
    return bool(keys & overlay_source_entities or keys & ingestion_seen_entities)


def _gap_type(entry: dict) -> str:
    """Classify a raw log entry into one of three gap types."""
    support_kind = entry.get("support_kind", "")
    reason = entry.get("reason", "").lower()

    if entry.get("temporal_mismatch") is True:
        return "stale_snapshot"
    if any(
        marker in reason
        for marker in (
            "snapshot_requires_as_of",
            "snapshot_missing_as_of",
            "aggregate_requires_as_of",
            "missing as_of",
            "outdated snapshot",
            "stale snapshot",
        )
    ):
        return "stale_snapshot"

    if support_kind in _POLICY_SUPPORT_KINDS:
        return "policy_blocked"

    # Universal policy check on reason text — applies to all support_kinds.
    if any(sub in reason for sub in _POLICY_REASON_SUBSTRINGS):
        return "policy_blocked"

    if support_kind == "multihop_audit":
        if any(kw in reason for kw in _POLICY_REASON_KEYWORDS):
            return "policy_blocked"
        if "missing" in reason:
            return "missing_facts"
        return "unknown_entity"

    if support_kind == "missing_knowledge":
        # Entity recognised but specific fact absent.
        if any(sub in reason for sub in _MISSING_FACT_SUBSTRINGS):
            return "missing_facts"
        # "no stable definition" → entity not in the overlay at all.
        if "no stable definition" in reason:
            return "unknown_entity"
        return "missing_facts"

    if support_kind == "unsupported":
        return "unknown_entity"

    # Fallback: inspect reason text.
    if any(kw in reason for kw in ("missing", "no fact", "no supported")):
        return "missing_facts"
    return "unknown_entity"


def _parse_ts(ts: str) -> datetime:
    if not ts:
        return datetime.min.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def load_audit_log(log_path: Path | None = None) -> list[dict]:
    """Load all JSONL entries from the audit log. Returns [] if file absent."""
    path = log_path or _DEFAULT_LOG
    if not path.exists():
        return []
    entries: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return entries


def analyze_gaps(
    log_path: Path | None = None,
    *,
    period_days: int = 30,
    overlay_path: Path | str | None = None,
    current_date: str | date | datetime | None = None,
    require_acquisition_eligibility: bool = False,
) -> GapReport:
    """Read audit log, filter to period_days, and produce a GapReport.

    period_days=0 means use all entries regardless of age.
    Acquisition candidates (missing_facts + unknown_entity) are sorted by
    count descending; policy_blocked likewise. Policy-blocked entities are
    excluded from acquisition_candidates and must not go to the frontier.
    If overlay_path is supplied, snapshot facts from that overlay are scanned
    for freshness recheck candidates.
    """
    all_entries = load_audit_log(log_path)

    if period_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=period_days)
        entries = [e for e in all_entries if _parse_ts(e.get("timestamp", "")) >= cutoff]
    else:
        entries = all_entries

    counts: Counter[tuple[str, str]] = Counter()
    reasons: dict[tuple[str, str], list[str]] = defaultdict(list)
    grouped_entries: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for entry in entries:
        entity = (entry.get("entity") or "").strip() or "[unknown entity]"
        gtype = _gap_type(entry)
        key = (entity, gtype)
        counts[key] += 1
        grouped_entries[key].append(entry)
        reason = entry.get("reason", "")
        if reason and reason not in reasons[key]:
            reasons[key].append(reason)

    overlay_source_entities: set[str] = set()
    ingestion_seen_entities: set[str] = set()
    if require_acquisition_eligibility:
        overlay_source_entities = _build_overlay_source_index()
        ingestion_seen_entities = _build_ingestion_seen_index()

    acquisition: list[EntityGapEntry] = []
    blocked: list[EntityGapEntry] = []
    for (entity, gtype), count in sorted(counts.items(), key=lambda x: -x[1]):
        entry_obj = EntityGapEntry(
            entity=entity,
            gap_type=gtype,
            count=count,
            top_reasons=reasons[(entity, gtype)][:_TOP_REASONS_LIMIT],
        )
        if gtype == "policy_blocked":
            blocked.append(entry_obj)
        elif gtype == "stale_snapshot":
            continue
        elif require_acquisition_eligibility and not _eligible_for_acquisition(
            entity,
            grouped_entries[(entity, gtype)],
            overlay_source_entities=overlay_source_entities,
            ingestion_seen_entities=ingestion_seen_entities,
        ):
            continue
        else:
            acquisition.append(entry_obj)

    stale_candidates = []
    if overlay_path is not None and Path(overlay_path).exists():
        stale_candidates = detect_stale_candidates(overlay_path, current_date)

    return GapReport(
        generated_at=datetime.now(timezone.utc).isoformat(),
        period_days=period_days,
        total_audit_events=len(entries),
        acquisition_candidates=acquisition,
        policy_blocked=blocked,
        stale_candidates=stale_candidates,
    )
