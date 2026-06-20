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

from worldpgt.knowledge.staleness_detector import detect_stale_candidates
from worldpgt.knowledge_pump.audit_types import EntityGapEntry, GapReport

_DEFAULT_LOG = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "knowledge_pump_v1"
    / "audit_log.jsonl"
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

    for entry in entries:
        entity = (entry.get("entity") or "").strip() or "[unknown entity]"
        gtype = _gap_type(entry)
        key = (entity, gtype)
        counts[key] += 1
        reason = entry.get("reason", "")
        if reason and reason not in reasons[key]:
            reasons[key].append(reason)

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
