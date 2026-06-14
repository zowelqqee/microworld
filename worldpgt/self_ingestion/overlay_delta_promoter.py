"""Overlay delta promoter for Promote Overlay Delta v1.

Loads the base accepted overlay and the self-ingestion delta proposal, re-runs
the delta validator, and merges the base overlay with only the accepted (safe)
delta items into a *separate* promoted overlay artifact.

The base accepted overlay is preserved unchanged and is NEVER overwritten. The
promoted overlay is an isolated, promoted self-ingestion overlay — NOT trusted
general runtime memory (``safe_for_general_runtime`` is always False).

Deterministic, offline only. No ML, no embeddings, no network.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from worldpgt.self_ingestion.overlay_delta_validator import (
    ValidationResult,
    validate_delta,
)

_PROMOTED_META = {
    "overlay_kind": "promoted_self_ingestion_overlay_v1",
    "promoted_from": "self_ingestion_v1/overlay_delta_proposal.json",
    "base_overlay": "accepted_wiki_memory_overlay_v1.json",
    "trusted_general_runtime_memory": False,
    "safe_for_general_runtime": False,
    "note": (
        "This overlay is a PROMOTED self-ingestion overlay built by merging the "
        "base accepted wiki overlay with validator-accepted delta items. It is an "
        "isolated knowledge overlay for QA regression only and is NOT trusted "
        "general runtime memory."
    ),
}


@dataclass
class PromotionResult:
    base_items: list[dict] = field(default_factory=list)
    delta_items: list[dict] = field(default_factory=list)
    promoted_items: list[dict] = field(default_factory=list)
    validation: ValidationResult = None  # type: ignore[assignment]
    base_count: int = 0
    delta_count: int = 0
    accepted_delta_count: int = 0
    promoted_count: int = 0
    meta: dict = field(default_factory=lambda: dict(_PROMOTED_META))


def promote(
    base_overlay_path: str,
    delta_proposal_path: str,
    promoted_overlay_out_path: str,
    meta_out_path: str | None = None,
) -> PromotionResult:
    base_items = json.loads(Path(base_overlay_path).read_text(encoding="utf-8"))
    delta_items = json.loads(Path(delta_proposal_path).read_text(encoding="utf-8"))

    validation = validate_delta(delta_items, base_items)

    # Merge base + accepted delta only. Base is copied, not mutated.
    promoted_items = list(base_items) + list(validation.accepted_items)

    out = Path(promoted_overlay_out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Written as a plain list of overlay items so it is directly QA-runnable by
    # the existing overlay provider. Provenance metadata is kept in a sidecar.
    out.write_text(json.dumps(promoted_items, indent=2, ensure_ascii=False),
                   encoding="utf-8")

    meta = dict(_PROMOTED_META)
    meta.update({
        "base_overlay_items": len(base_items),
        "accepted_delta_items": len(validation.accepted_items),
        "promoted_overlay_items": len(promoted_items),
    })
    if meta_out_path:
        Path(meta_out_path).write_text(json.dumps(meta, indent=2, ensure_ascii=False),
                                       encoding="utf-8")

    return PromotionResult(
        base_items=base_items,
        delta_items=delta_items,
        promoted_items=promoted_items,
        validation=validation,
        base_count=len(base_items),
        delta_count=len(delta_items),
        accepted_delta_count=len(validation.accepted_items),
        promoted_count=len(promoted_items),
        meta=meta,
    )
