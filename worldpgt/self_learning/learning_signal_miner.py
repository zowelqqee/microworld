"""Signal miner for the Self-Learning Loop v1.

Turns loaded artifacts into a flat list of :class:`LearningSignal` objects.
This layer is purely descriptive: it counts and normalizes what already
happened. It makes no policy decisions and proposes nothing.
"""

from __future__ import annotations

from collections import Counter
from typing import List

from .artifact_loader import LoadedArtifacts, OPTIONAL_JSON, REQUIRED_JSON, OPTIONAL_CSV
from .types import LearningSignal

# Reverse maps so each signal can carry the real relative path of its source.
_ALL_REL = {**REQUIRED_JSON, **OPTIONAL_JSON, **OPTIONAL_CSV}


def _rel(name: str) -> str:
    return _ALL_REL.get(name, name)


def mine_signals(loaded: LoadedArtifacts) -> List[LearningSignal]:
    signals: List[LearningSignal] = []
    signals.extend(_mine_quarantine(loaded))
    signals.extend(_mine_quarantine_reason_counts(loaded))
    signals.extend(_mine_manifest_errors(loaded))
    signals.extend(_mine_qa_audits(loaded))
    signals.extend(_mine_promotion_status(loaded))
    signals.extend(_mine_regression_status(loaded))
    signals.extend(_mine_overlay_relations(loaded))
    return signals


def _mine_quarantine(loaded: LoadedArtifacts) -> List[LearningSignal]:
    items = loaded.json_data.get("quarantine") or []
    out: List[LearningSignal] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            LearningSignal(
                signal_type="quarantine_item",
                source_artifact=_rel("quarantine"),
                text=str(item.get("text", "")),
                source_row_id=item.get("candidate_id"),
                reason=item.get("reason"),
                risk=item.get("risk"),
                observed_decision=item.get("suggested_action"),
                extra={"source_id": item.get("source_id")},
            )
        )
    return out


def _mine_quarantine_reason_counts(loaded: LoadedArtifacts) -> List[LearningSignal]:
    """Frequency of each quarantine reason (computed from the quarantine list)."""
    items = loaded.json_data.get("quarantine") or []
    counter: Counter = Counter(
        str(i.get("reason")) for i in items if isinstance(i, dict) and i.get("reason")
    )
    out: List[LearningSignal] = []
    for reason, count in sorted(counter.items()):
        out.append(
            LearningSignal(
                signal_type="quarantine_reason_count",
                source_artifact=_rel("quarantine"),
                text=reason,
                reason=reason,
                count=count,
            )
        )
    return out


def _mine_manifest_errors(loaded: LoadedArtifacts) -> List[LearningSignal]:
    items = loaded.json_data.get("manifest_errors") or []
    out: List[LearningSignal] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        out.append(
            LearningSignal(
                signal_type="manifest_error",
                source_artifact=_rel("manifest_errors"),
                text=str(item.get("source_id", "")),
                source_row_id=item.get("source_id"),
                reason=item.get("reason"),
            )
        )
    return out


# QA outputs whose audit rows can indicate genuinely missing stable knowledge.
# The adversarial set is deliberately excluded here: its audit rows are guard
# behavior, and are mined from quarantine instead.
_BENIGN_QA_OUTPUTS = (
    "entity_qa_outputs",
    "entity_qa_expansion_outputs",
    "cross_page_qa_outputs",
)


def _mine_qa_audits(loaded: LoadedArtifacts) -> List[LearningSignal]:
    out: List[LearningSignal] = []
    for name in _BENIGN_QA_OUTPUTS:
        rows = loaded.csv_data.get(name)
        if not rows:
            continue
        for row in rows:
            if (row.get("decision") or "").strip().lower() != "audit":
                continue
            out.append(
                LearningSignal(
                    signal_type="qa_audit",
                    source_artifact=_rel(name),
                    text=str(row.get("question", "")),
                    source_row_id=row.get("row_id"),
                    reason=row.get("audit_reason") or row.get("detected_intent"),
                    observed_decision="audit",
                    extra={"detected_intent": row.get("detected_intent")},
                )
            )
    return out


def _mine_promotion_status(loaded: LoadedArtifacts) -> List[LearningSignal]:
    report = loaded.json_data.get("promotion_report") or {}
    if not isinstance(report, dict):
        return []
    return [
        LearningSignal(
            signal_type="promotion_status",
            source_artifact=_rel("promotion_report"),
            text="promotion_status",
            extra={
                "safe_to_promote": report.get("safe_to_promote"),
                "trusted_memory_modified": report.get("trusted_memory_modified"),
                "accepted_overlay_modified": report.get("accepted_overlay_modified"),
                "base_overlay_items": report.get("base_overlay_items"),
                "promoted_overlay_items": report.get("promoted_overlay_items"),
                "accepted_delta_items": report.get("accepted_delta_items"),
                "rejected_delta_items": report.get("rejected_delta_items"),
                "blocked_delta_items": report.get("blocked_delta_items"),
                "weak_links_remain_weak": report.get("weak_links_remain_weak"),
                "volatile_items_not_promoted": report.get("volatile_items_not_promoted"),
                "private_items_not_promoted": report.get("private_items_not_promoted"),
                "inverted_relations_not_promoted": report.get(
                    "inverted_relations_not_promoted"
                ),
            },
        )
    ]


def _mine_regression_status(loaded: LoadedArtifacts) -> List[LearningSignal]:
    """Extract per-suite regression PASS/FAIL status."""
    status = loaded.json_data.get("promotion_regression_summary")
    artifact = _rel("promotion_regression_summary")
    if not isinstance(status, dict) or not status:
        # Fall back to the regression block embedded in the ingestion report.
        report = loaded.json_data.get("self_ingestion_report") or {}
        status = report.get("qa_regression_status") if isinstance(report, dict) else None
        artifact = _rel("self_ingestion_report")
    out: List[LearningSignal] = []
    if isinstance(status, dict):
        for suite, info in sorted(status.items()):
            info = info if isinstance(info, dict) else {}
            out.append(
                LearningSignal(
                    signal_type="regression_status",
                    source_artifact=artifact,
                    text=suite,
                    observed_decision=info.get("status"),
                    extra=dict(info),
                )
            )
    return out


def _mine_overlay_relations(loaded: LoadedArtifacts) -> List[LearningSignal]:
    """Distinct relation predicates observed in the proposed overlay delta."""
    items = loaded.json_data.get("overlay_delta_proposal") or []
    seen = set()
    out: List[LearningSignal] = []
    for item in items:
        if not isinstance(item, dict) or item.get("overlay_type") != "overlay_relation":
            continue
        predicate = str(item.get("predicate", "")).strip()
        if not predicate or predicate in seen:
            continue
        seen.add(predicate)
        out.append(
            LearningSignal(
                signal_type="overlay_relation",
                source_artifact=_rel("overlay_delta_proposal"),
                text=predicate,
                reason="predicate",
                risk=item.get("risk"),
                extra={
                    "subject": item.get("subject"),
                    "object": item.get("object"),
                    "stability": item.get("stability"),
                },
            )
        )
    return out
