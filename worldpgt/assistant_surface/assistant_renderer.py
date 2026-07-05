"""User-facing renderer for the Assistant Surface v1.

Turns an :class:`AssistantAnswer` into a clean, honest, plain-text block. For
answers it shows the body plus a compact support/overlay/decision footer. For
audits it explains why no stable answer is given. The snapshot dry-run overlay
is always clearly marked as a proposal, not accepted memory.

Deterministic string formatting only. No ML, no network.
"""

from __future__ import annotations

from worldpgt.assistant_surface.types import (
    AssistantAnswer,
    OVERLAY_MODE_CUSTOM_PATH,
    OVERLAY_MODE_PUMP_DRY_RUN,
    OVERLAY_MODE_SNAPSHOT_DRY_RUN,
)
from worldpgt.knowledge_pump.audit_event_logger import log_audit_event

_SUPPORT_LABELS = {
    "stable_definition": "stable definition",
    "stable_relation": "stable relation",
    "semi_stable_relation": "semi-stable relation",
    "explicit_connection_path": "explicit connection path",
    "explicit_is_a_chain": "explicit is_a chain",
    "explicit_type_contradiction": "explicit type contradiction",
    "entity_type_mismatch": "entity type mismatch",
    "source_qualified_fact": "source-qualified fact (volatile)",
    "web_search_result": "live web search result (volatile)",
    "safe_policy_answer": "safe policy explanation",
    "audit_blocked_context": "blocked by safety policy",
    "missing_knowledge": "missing knowledge",
    "unsupported": "unsupported",
}

_SNAPSHOT_MARKER = "Overlay mode: snapshot-dry-run proposal, not accepted memory."
_PUMP_MARKER = "Overlay mode: pump-dry-run proposal, not accepted memory."
_CUSTOM_MARKER = "Overlay mode: custom overlay path, not accepted memory."


def render(answer: AssistantAnswer) -> str:
    lines: list[str] = []

    marker = {
        OVERLAY_MODE_SNAPSHOT_DRY_RUN: _SNAPSHOT_MARKER,
        OVERLAY_MODE_PUMP_DRY_RUN: _PUMP_MARKER,
        OVERLAY_MODE_CUSTOM_PATH: _CUSTOM_MARKER,
    }.get(answer.overlay_mode)
    if marker:
        lines.append(marker)
        lines.append("")

    if answer.decision == "answer":
        lines.append(answer.answer_text.strip())
        lines.append("")
        lines.append(f"Support: {_SUPPORT_LABELS.get(answer.support_kind, answer.support_kind)}.")
        lines.append(f"Overlay: {answer.overlay_mode}.")
        lines.append("Decision: answer.")
    elif answer.decision == "no":
        lines.append(answer.answer_text.strip())
        lines.append("")
        lines.append(f"Support: {_SUPPORT_LABELS.get(answer.support_kind, answer.support_kind)}.")
        lines.append(f"Overlay: {answer.overlay_mode}.")
        lines.append("Decision: no.")
    else:
        lines.append("I can't answer that as a stable fact from Microworld's current memory.")
        lines.append("")
        reason = answer.answer_text.strip() or "no supported answer exists"
        lines.append(f"Reason: {reason}.")
        lines.append("Decision: audit.")
        lines.append(f"Overlay: {answer.overlay_mode}.")
        log_audit_event(answer)

    return "\n".join(lines).strip()
