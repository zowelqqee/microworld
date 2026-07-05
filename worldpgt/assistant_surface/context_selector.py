"""Context selector for the Assistant Surface v1.

For every question this builds a Working Context Pack (read-only, via the
existing :class:`ContextPackBuilder`), validates it, and condenses it into an
:class:`AssistantContextSummary` that the orchestrator can reason over without
re-touching overlay internals.

Safety policy enforced here:
- weak-only context can never count as factual support;
- volatile / source-qualified facts are tracked separately, never as "stable";
- private / inverted / universal questions are screened upstream by the router,
  and the pack's own safety screens are surfaced as ``safety_notes``.

Read-only over the overlay. No writes, no network, no ML.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from worldpgt.assistant_surface.types import AssistantContextSummary
from worldpgt.assistant_surface.types import OVERLAY_MODE_CUSTOM_PATH, OVERLAY_MODE_PUMP_DRY_RUN
from worldpgt.context_pack.context_pack_builder import ContextPackBuilder
from worldpgt.context_pack.context_pack_validator import validate_pack
from worldpgt.context_pack.types import (
    OVERLAY_ACCEPTED,
    OVERLAY_PROMOTED,
    STABLE_STABILITIES,
    STABILITY_VOLATILE,
    TRUST_WEAK,
    WorkingContextPack,
)

_EXPERIMENTS = Path(__file__).resolve().parent.parent / "experiments"

# Overlay file resolution by assistant overlay mode.
ACCEPTED_OVERLAY_PATH = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
PROMOTED_OVERLAY_PATH = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
SNAPSHOT_DRY_RUN_OVERLAY_PATH = (
    _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
)
PUMP_DRY_RUN_OVERLAY_PATH = (
    _EXPERIMENTS / "knowledge_pump_v1" / "pump_dry_run_overlay.json"
)


def resolve_overlay(overlay_mode: str) -> tuple[str, str]:
    """Return (overlay_json_path, context_pack_overlay_mode) for ``overlay_mode``.

    The context pack builder only knows ``accepted`` / ``promoted`` as semantic
    modes; the snapshot dry-run overlay is read with the ``promoted`` semantics
    (its base) but kept clearly labelled as a proposal at the surface level.
    """

    if overlay_mode == "accepted":
        return str(ACCEPTED_OVERLAY_PATH), OVERLAY_ACCEPTED
    if overlay_mode == "promoted":
        return str(PROMOTED_OVERLAY_PATH), OVERLAY_PROMOTED
    if overlay_mode == "snapshot-dry-run":
        return str(SNAPSHOT_DRY_RUN_OVERLAY_PATH), OVERLAY_PROMOTED
    if overlay_mode == OVERLAY_MODE_PUMP_DRY_RUN:
        return str(PUMP_DRY_RUN_OVERLAY_PATH), OVERLAY_PROMOTED
    raise ValueError(f"unknown overlay mode: {overlay_mode!r}")


@lru_cache(maxsize=16)
def _cached_context_pack_builder(
    overlay_path: str,
    overlay_mode: str,
    mtime_ns: int,
    size: int,
) -> ContextPackBuilder:
    del mtime_ns, size
    return ContextPackBuilder(overlay_path, overlay_mode)


def _context_pack_builder_for(overlay_path: str, overlay_mode: str) -> ContextPackBuilder:
    stat = Path(overlay_path).stat()
    return _cached_context_pack_builder(
        overlay_path,
        overlay_mode,
        stat.st_mtime_ns,
        stat.st_size,
    )


class ContextSelector:
    """Builds and summarizes a context pack for an assistant question."""

    def __init__(
        self,
        overlay_mode: str = "promoted",
        overlay_path: str | Path | None = None,
    ) -> None:
        self.overlay_mode = overlay_mode
        if overlay_path is not None:
            self.overlay_mode = OVERLAY_MODE_CUSTOM_PATH
            self._overlay_path, self._pack_mode = str(overlay_path), OVERLAY_PROMOTED
        else:
            self._overlay_path, self._pack_mode = resolve_overlay(overlay_mode)
        self._builder = _context_pack_builder_for(self._overlay_path, self._pack_mode)

    def build_pack(self, question: str) -> WorkingContextPack:
        return self._builder.build(question)

    def select(self, question: str) -> tuple[WorkingContextPack, AssistantContextSummary]:
        pack = self.build_pack(question)
        validation = validate_pack(pack)

        # Stable definitions present?
        has_stable_definition = any(
            (d.stability in STABLE_STABILITIES) for d in pack.definitions
        ) or bool(pack.definitions)

        # Relations split by stability and trust. Weak-trust relations never
        # count as factual support.
        stable_rels = [
            r
            for r in pack.direct_relations
            if r.trust != TRUST_WEAK and r.stability in {"stable"}
        ]
        semi_rels = [
            r
            for r in pack.direct_relations
            if r.trust != TRUST_WEAK and r.stability in {"semi_stable"}
        ]

        has_explicit_path = bool(pack.candidate_paths)
        has_source_fact = bool(pack.source_facts)

        # Weak-only: relations/links exist but no stable definition/relation/path
        # and no source fact.
        has_any_strong = (
            has_stable_definition
            or bool(stable_rels)
            or bool(semi_rels)
            or has_explicit_path
            or has_source_fact
        )
        has_weak_only = bool(pack.weak_links) and not has_any_strong

        summary = AssistantContextSummary(
            overlay_mode=self.overlay_mode,
            pack_valid=validation.passed,
            matched_entities=[m.name for m in pack.matched_entities],
            candidate_paths=[
                " -> ".join(p.nodes) for p in pack.candidate_paths
            ],
            blocked_paths=[
                f"{' -> '.join(p.nodes)} ({p.reason})" for p in pack.blocked_paths
            ],
            weak_links=[
                f"{w.source_page}->{w.target} ({w.relation})" for w in pack.weak_links
            ],
            source_facts=[
                f"{s.subject} {s.predicate} {s.object} [src={s.source_name}, "
                f"stability={s.stability}]"
                for s in pack.source_facts
            ],
            definitions=[f"{d.subject}: {d.definition}" for d in pack.definitions],
            direct_relations=[
                f"{r.subject} {r.predicate} {r.object} "
                f"[stability={r.stability}, trust={r.trust}]"
                for r in pack.direct_relations
            ],
            missing_hints=[h.category for h in pack.missing_knowledge_hints],
            safety_notes=list(pack.safety_notes),
            has_stable_definition=has_stable_definition,
            has_stable_relation=bool(stable_rels),
            has_semi_stable_relation=bool(semi_rels),
            has_explicit_path=has_explicit_path,
            has_source_fact=has_source_fact,
            has_weak_only=has_weak_only,
            safe_for_answering=pack.safe_for_answering,
        )
        return pack, summary


# Convenience set re-exported for callers that need to reason about volatility.
VOLATILE = STABILITY_VOLATILE
