"""Write structured audit events to a persistent JSONL log.

Called at the two renderer sites where Decision: audit is emitted.
Each call appends one JSON line; no reads, no lock, no truncation.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from worldpgt.assistant_surface.types import AssistantAnswer
    from worldpgt.multihop_qa.assistant_adapter import AssistantMultihopResult

_DEFAULT_LOG = (
    Path(__file__).resolve().parent.parent
    / "experiments"
    / "knowledge_pump_v1"
    / "audit_log.jsonl"
)

# Simplified mirrors of question_router.py intent patterns — entity only, no routing.
_ENTITY_PATTERNS = [
    re.compile(r"^(?:who|what)\s+is\s+(.+?)[\?.]?$", re.IGNORECASE),
    re.compile(
        r"^what\s+does\s+(.+?)\s+"
        r"(?:develop|own|found|lead|make|create|operate|run|produce|launch|build)",
        re.IGNORECASE,
    ),
    re.compile(r"^how\s+(?:is|are|was|were)\s+(.+?)\s+connected", re.IGNORECASE),
    re.compile(
        r"^(?:who|what)\s+(?:founded|owns|leads|runs|created|operates|made|launched)\s+(.+?)[\?.]?$",
        re.IGNORECASE,
    ),
    re.compile(r"^what\s+does\s+(.+?)\s+develop\s+that", re.IGNORECASE),
]


def _extract_entity(question: str) -> str:
    """Return the main entity from a question string, or "" if not matched."""
    q = (question or "").strip()
    for pat in _ENTITY_PATTERNS:
        m = pat.match(q)
        if m:
            entity = m.group(1).strip().rstrip("?.")
            if 0 < len(entity) <= 80:
                return entity
    return ""


def _append(path: Path, entry: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _temporal_mismatch(reason: str, risk_flags: list[str] | None = None) -> bool:
    flags = set(risk_flags or [])
    if "temporal_mismatch" in flags:
        return True
    text = (reason or "").lower()
    return any(
        marker in text
        for marker in (
            "snapshot_requires_as_of",
            "snapshot_missing_as_of",
            "aggregate_requires_as_of",
            "missing as_of",
            "outdated snapshot",
            "stale snapshot",
        )
    )


def log_audit_event(
    answer: "AssistantAnswer",
    *,
    log_path: Path | None = None,
    entity: str | None = None,
    relation_hint: str | None = None,
    source: str = "assistant_surface",
) -> None:
    """Append one audit event from an AssistantAnswer to the JSONL log.

    No-ops silently for non-audit decisions so callers need no guard.
    """
    if answer.decision != "audit":
        return
    reason = answer.answer_text.strip() or "no supported answer exists"
    _append(log_path or _DEFAULT_LOG, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": answer.question,
        "entity": (entity or "").strip() or _extract_entity(answer.question),
        "relation_hint": (relation_hint or "").strip(),
        "support_kind": answer.support_kind,
        "reason": reason,
        "source": source,
        "temporal_mismatch": _temporal_mismatch(reason, answer.risk_flags),
    })


def log_multihop_audit_event(
    result: "AssistantMultihopResult",
    *,
    log_path: Path | None = None,
) -> None:
    """Append one audit event from an AssistantMultihopResult to the JSONL log.

    No-ops silently for non-audit decisions so callers need no guard.
    """
    if result.decision != "audit":
        return
    reason = result.audit_reason or "no supported multi-hop relation chain found"
    _append(log_path or _DEFAULT_LOG, {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": result.question,
        "entity": _extract_entity(result.question),
        "support_kind": "multihop_audit",
        "reason": reason,
        "source": "multihop",
        "temporal_mismatch": _temporal_mismatch(reason),
    })
