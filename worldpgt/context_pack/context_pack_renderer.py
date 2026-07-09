"""Compact human-readable renderer for Working Context Pack v1.

Produces a deterministic multi-line text summary. No generation, no model —
just structured fields formatted for a human reviewer.
"""

from __future__ import annotations

from .types import WorkingContextPack


def render(pack: WorkingContextPack) -> str:
    lines = [f"Question: {pack.question}"]
    lines.append(f"Overlay mode: {pack.overlay_mode}")

    ents = ", ".join(m.name for m in pack.matched_entities) or "none"
    lines.append(f"Matched entities: {ents}")

    if pack.candidate_paths:
        lines.append("Stable/semi-stable paths:")
        for p in pack.candidate_paths:
            lines.append(f"- {' -> '.join(p.nodes)} ({p.support})")
    else:
        lines.append("Stable/semi-stable paths: none")

    if pack.direct_relations:
        lines.append("Direct relations:")
        for r in pack.direct_relations[:8]:
            lines.append(f"- {r.subject} -{r.predicate}-> {r.object} ({r.stability})")

    if pack.weak_links:
        lines.append("Weak links:")
        for w in pack.weak_links[:8]:
            lines.append(f"- {w.source_page} ~ {w.target} ({w.trust})")
    else:
        lines.append("Weak links: none")

    if pack.source_facts:
        lines.append("Source-qualified facts:")
        for sf in pack.source_facts:
            lines.append(
                f"- {sf.subject} {sf.predicate} {sf.object} "
                f"(source={sf.source_name}, as_of={sf.as_of}, volatile/requires_recheck)"
            )
    else:
        lines.append("Source-qualified facts: none")

    if pack.blocked_paths:
        lines.append("Blocked:")
        for b in pack.blocked_paths:
            lines.append(f"- {b.reason}: {b.detail}")

    if pack.missing_knowledge_hints:
        lines.append("Missing-knowledge hints:")
        for h in pack.missing_knowledge_hints:
            lines.append(
                f"- {h.request_id} [{h.category}] needs {h.needed_evidence_type} "
                f"(audit until supported={h.must_remain_audit_until_supported})"
            )

    if pack.safety_notes:
        lines.append("Safety notes:")
        for n in pack.safety_notes:
            lines.append(f"- {n}")

    verdict = (
        "safe to answer from explicit stable support"
        if pack.safe_for_answering
        else "NOT safe to answer as a stable fact"
    )
    lines.append(f"Safety: {verdict} (safe_for_general_runtime=False)")
    return "\n".join(lines)
