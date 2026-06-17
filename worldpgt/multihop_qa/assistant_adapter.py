"""Assistant/CLI adapter for optional Multi-hop QA v1.

This module is deliberately thin: it loads no network resources, writes no
memory, and delegates all chain reasoning to the existing multi-hop analyzer,
graph, planner, validator, and renderer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from worldpgt.multihop_qa.answer_renderer import render as render_multihop_plan
from worldpgt.multihop_qa.path_planner import MultihopPlanner
from worldpgt.multihop_qa.question_analyzer import analyze
from worldpgt.multihop_qa.relation_graph import RelationGraph
from worldpgt.multihop_qa.types import MultihopPlan

_HOW_CONNECTED_RE = re.compile(
    r"how\s+(?:is|are|was|were)\s+.+?\s+connected\s+to\s+.+?[\?.]?$",
    re.IGNORECASE,
)
_WHAT_LINKS_RE = re.compile(
    r"what\s+links\s+.+?\s+to\s+.+?[\?.]?$",
    re.IGNORECASE,
)
_IS_CONNECTED_THROUGH_RE = re.compile(
    r"is\s+.+?\s+connected\s+to\s+.+?\s+through\s+.+?[\?.]?$",
    re.IGNORECASE,
)
_DEVELOPS_ISA_RE = re.compile(
    r"what\s+does\s+.+?\s+develop\s+that\s+is\s+(?:a|an)\s+.+?[\?.]?$",
    re.IGNORECASE,
)


@dataclass
class AssistantMultihopResult:
    """Result object used by the CLI integration."""

    question: str
    decision: str
    answer_text: str
    audit_reason: str
    chain_type: str
    chain_label: str
    hop1: dict[str, Any] | None
    hop2: dict[str, Any] | None
    support_kind: str = "explicit_multi_hop_relation_chain"

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "decision": self.decision,
            "answer_text": self.answer_text,
            "audit_reason": self.audit_reason,
            "chain_type": self.chain_type,
            "chain_label": self.chain_label,
            "hop1": self.hop1,
            "hop2": self.hop2,
            "support_kind": self.support_kind,
        }


def is_multihop_compatible(question: str) -> bool:
    """Return True only for explicit CLI-supported multi-hop question forms."""

    q = (question or "").strip()
    return bool(
        _HOW_CONNECTED_RE.fullmatch(q)
        or _WHAT_LINKS_RE.fullmatch(q)
        or _IS_CONNECTED_THROUGH_RE.fullmatch(q)
        or _DEVELOPS_ISA_RE.fullmatch(q)
    )


def _result_from_plan(question: str, plan: MultihopPlan) -> AssistantMultihopResult:
    answer_text = render_multihop_plan(plan)
    return AssistantMultihopResult(
        question=question,
        decision=plan.decision,
        answer_text=answer_text,
        audit_reason=plan.audit_reason or "",
        chain_type=plan.question.chain_type,
        chain_label=plan.chain_label,
        hop1=plan.path.hop1.to_detail_dict() if plan.path else None,
        hop2=plan.path.hop2.to_detail_dict() if plan.path else None,
    )


def try_answer_multihop(
    question: str,
    overlay_items: list[dict[str, Any]],
) -> AssistantMultihopResult:
    """Try optional multi-hop QA over explicit supported relation chains.

    The graph builder itself excludes definitions, entity cards, weak context
    links, and source facts. The adapter also rejects unsupported CLI question
    forms before planning, so direct-only founder/owner questions cannot leak a
    transitive answer through the CLI integration.
    """

    if not is_multihop_compatible(question):
        return AssistantMultihopResult(
            question=question,
            decision="audit",
            answer_text="[audit] no recognized multi-hop question pattern matched",
            audit_reason="no recognized multi-hop question pattern matched",
            chain_type="unsupported",
            chain_label="unsupported",
            hop1=None,
            hop2=None,
        )

    graph = RelationGraph(overlay_items)
    planner = MultihopPlanner(graph)
    plan = planner.plan(analyze(question))
    return _result_from_plan(question, plan)


def render_cli_multihop_result(
    result: AssistantMultihopResult,
    *,
    overlay_mode: str,
    overlay_marker: str = "",
) -> str:
    """Render a multi-hop result in the same CLI footer style as assistant QA."""

    lines: list[str] = []
    if overlay_marker:
        lines.append(overlay_marker)
        lines.append("")

    if result.decision == "answer":
        lines.append(result.answer_text.strip())
        lines.append("")
        lines.append("Support: explicit multi-hop relation chain.")
        lines.append(f"Overlay: {overlay_mode}.")
        lines.append("Decision: answer.")
    else:
        reason = result.audit_reason or "no supported multi-hop relation chain found"
        lines.append("I can't answer that as a stable fact from Microworld's current memory.")
        lines.append("")
        lines.append(f"Reason: {reason}.")
        lines.append("Decision: audit.")
        lines.append(f"Overlay: {overlay_mode}.")

    return "\n".join(lines).strip()
