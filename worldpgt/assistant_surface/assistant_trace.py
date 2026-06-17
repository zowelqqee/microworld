"""Trace construction helpers for the Assistant Surface v1.

A trace records the deterministic decision path: routing, context selection, the
QA path chosen, and the support/audit reasoning. It is purely descriptive — it
never influences memory and carries no runtime side effects.
"""

from __future__ import annotations

from worldpgt.assistant_surface.types import (
    AssistantContextSummary,
    AssistantRoute,
    AssistantTrace,
)


def new_trace(route: AssistantRoute) -> AssistantTrace:
    trace = AssistantTrace(route_intent=route.intent, risk_flags=list(route.risk_flags))
    trace.add(f"router: intent={route.intent}")
    if route.subject:
        trace.add(f"router: subject={route.subject}")
    if route.secondary:
        trace.add(f"router: secondary={route.secondary}")
    if route.is_hard_safety:
        trace.add("router: hard-safety route engaged")
    return trace


def attach_context(trace: AssistantTrace, summary: AssistantContextSummary) -> None:
    trace.context_summary = summary.to_dict()
    trace.add(
        "context: pack built "
        f"(valid={summary.pack_valid}, entities={len(summary.matched_entities)}, "
        f"paths={len(summary.candidate_paths)}, blocked={len(summary.blocked_paths)}, "
        f"weak={len(summary.weak_links)}, source_facts={len(summary.source_facts)})"
    )
    if summary.safety_notes:
        trace.add("context: safety_notes=" + "; ".join(summary.safety_notes))


def finalize(trace: AssistantTrace, source_system: str, support_kind: str) -> None:
    trace.source_system = source_system
    trace.support_kind = support_kind
    trace.add(f"decision: source_system={source_system}, support_kind={support_kind}")
