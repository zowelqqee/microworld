"""Promotion gates for generated relation families and local types.

Generated artifacts stay ``generated`` until every gate passes; only then are
they marked ``promoted``. Anything that fails a hard gate is ``rejected`` with a
reason. Generated schema is never silently promoted into accepted memory — the
gate only annotates status on the separate generated artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from worldpgt.schema_induction.types import (
    LocalType,
    PromotionDecision,
    RelationFamily,
)


@dataclass(frozen=True)
class GateConfig:
    min_evidence: int = 2
    min_sources: int = 1
    min_confidence: float = 0.6
    allow_self_loop: bool = False
    min_type_members: int = 2


def evaluate_relation_family(
    family: RelationFamily,
    claim_subject_lookup: dict[str, str] | None = None,
    config: GateConfig | None = None,
) -> PromotionDecision:
    """Run gates against one relation family."""

    cfg = config or GateConfig()
    lookup = claim_subject_lookup or {}
    passed: list[str] = []
    failed: list[str] = []

    if family.evidence_count >= cfg.min_evidence:
        passed.append("evidence_count")
    else:
        failed.append("evidence_count")

    if family.source_doc_count >= cfg.min_sources:
        passed.append("source_doc_count")
    else:
        failed.append("source_doc_count")

    if family.confidence >= cfg.min_confidence:
        passed.append("confidence")
    else:
        failed.append("confidence")

    if family.canonical_label.strip():
        passed.append("non_empty_trigger")
    else:
        failed.append("non_empty_trigger")

    # role consistency: the family must carry at least one non-subject role.
    if any(r != "subject" for r in family.roles):
        passed.append("role_consistency")
    else:
        failed.append("role_consistency")

    # source trace: every example claim must be present (non-empty list).
    if family.example_claim_ids:
        passed.append("source_trace")
    else:
        failed.append("source_trace")

    # no empty subject in any example claim.
    empty_subject = any(
        not (lookup.get(cid, "x").strip()) for cid in family.example_claim_ids
    ) if lookup else False
    if not empty_subject:
        passed.append("no_empty_subject")
    else:
        failed.append("no_empty_subject")

    status = "promoted" if not failed else "generated"
    reason = None if not failed else "failed gates: " + ", ".join(failed)
    return PromotionDecision(
        target_id=family.family_id,
        target_kind="relation_family",
        status=status,
        passed=tuple(passed),
        failed=tuple(failed),
        reason=reason,
    )


def evaluate_local_type(
    local_type: LocalType, config: GateConfig | None = None
) -> PromotionDecision:
    """Run gates against one local type."""

    cfg = config or GateConfig()
    passed: list[str] = []
    failed: list[str] = []

    if len(local_type.members) >= cfg.min_type_members:
        passed.append("min_members")
    else:
        failed.append("min_members")

    if local_type.label.strip():
        passed.append("non_empty_label")
    else:
        failed.append("non_empty_label")

    if local_type.confidence >= cfg.min_confidence:
        passed.append("confidence")
    else:
        failed.append("confidence")

    status = "promoted" if not failed else "generated"
    reason = None if not failed else "failed gates: " + ", ".join(failed)
    return PromotionDecision(
        target_id=local_type.type_id,
        target_kind="local_type",
        status=status,
        passed=tuple(passed),
        failed=tuple(failed),
        reason=reason,
    )


def apply_decisions(
    families: list[RelationFamily],
    decisions: dict[str, PromotionDecision],
) -> list[RelationFamily]:
    """Return families with ``promotion_status`` updated from decisions."""

    out: list[RelationFamily] = []
    for fam in families:
        decision = decisions.get(fam.family_id)
        if decision is None:
            out.append(fam)
            continue
        out.append(
            replace(
                fam,
                promotion_status=decision.status,
                rejection_reason=decision.reason if decision.status == "rejected" else None,
            )
        )
    return out
