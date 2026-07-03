"""Render explanation chains into honest, plain-text answers.

Mirrors the multihop answer renderer's footer style: the decision and support
kind are always stated explicitly, and a partial chain says exactly how far
the exploration got.
"""

from __future__ import annotations

from worldpgt.reasoning.types import ExplanationChain, ExplanationStep


def _step_line(index: int, step: ExplanationStep) -> str:
    label = {
        "fact": "fact",
        "rule": "rule",
        "pattern": "pattern (observation)",
        "note": "note",
    }[step.kind]
    suffix = ""
    if step.kind == "pattern" and step.confidence is not None:
        suffix = f" (confidence {step.confidence:.0%})"
    return f"  {index}. [{label}] {step.display()}{suffix}"


def render(chain: ExplanationChain) -> str:
    target = f"{chain.subject} | {chain.predicate} | {chain.object}"

    if chain.decision == "audit":
        reason = chain.audit_reason or "the fact is not in the graph"
        return (
            f"[audit] I can't explain \"{target}\": {reason}.\n"
            "Decision: audit."
        )

    lines: list[str] = []
    if chain.decision == "answer":
        lines.append(f"Why {target}:")
    else:
        lines.append(f"I can only partially explain {target}:")

    for i, step in enumerate(chain.steps, start=1):
        lines.append(_step_line(i, step))

    lines.append("")
    if chain.fact_status == "inferred":
        lines.append("Support: inference-rule proof chain over verified facts.")
    else:
        lines.append("Support: explanatory chain over verified facts.")
    if chain.decision == "partial" and chain.frontier:
        lines.append(f"Frontier reached: {', '.join(chain.frontier)}.")
    lines.append(f"Decision: {chain.decision}.")
    return "\n".join(lines)
