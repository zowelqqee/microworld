"""Converts wiki ingestion v2 candidates into an isolated memory overlay.

This builder is read-only with respect to all existing memory artifacts.
It does NOT modify:
- accepted_knowledge_memory_v1.json
- sense_memory.py
- any planner threshold or validator

Output overlay is accepted only for isolated entity QA overlay use.
safe_for_general_runtime is always False.
"""

from __future__ import annotations

import dataclasses
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from worldpgt.knowledge.wiki_memory_overlay_types import (
    SAFE_FOR_GENERAL_RUNTIME,
    OverlayContextLink,
    OverlayDefinition,
    OverlayEntity,
    OverlayRelation,
    OverlaySkipped,
    OverlaySourceFact,
)
from worldpgt.knowledge.temporal_classification import (
    classify_temporal_class,
    requires_as_of,
)


@dataclass
class OverlayBuildResult:
    items: list = field(default_factory=list)
    skipped: list[OverlaySkipped] = field(default_factory=list)
    summary: dict = field(default_factory=dict)


def as_dict(obj: object) -> object:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: as_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [as_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {k: as_dict(v) for k, v in obj.items()}
    return obj


class WikiCandidateOverlayBuilder:
    """Converts raw ingestion v2 candidate dicts into typed overlay items."""

    def build(self, candidates: list[dict]) -> OverlayBuildResult:
        items: list = []
        skipped: list[OverlaySkipped] = []

        for c in candidates:
            item_type = c.get("item_type", "")
            if item_type == "entity_card":
                result = self._convert_entity_card(c)
            elif item_type == "definition_claim":
                result = self._convert_definition(c)
            elif item_type == "relation_claim":
                result = self._convert_relation(c)
            elif item_type == "context_link":
                result = self._convert_context_link(c)
            elif item_type == "source_qualified_fact":
                result = self._convert_source_fact(c)
            else:
                skipped.append(OverlaySkipped(
                    item_type=item_type,
                    reason="unknown_item_type",
                    label_or_subject=str(c.get("label", c.get("subject", ""))),
                ))
                continue

            if isinstance(result, OverlaySkipped):
                skipped.append(result)
            else:
                items.append(result)

        summary = self._build_summary(candidates, items, skipped)
        return OverlayBuildResult(items=items, skipped=skipped, summary=summary)

    # ------------------------------------------------------------------
    # Conversion rules
    # ------------------------------------------------------------------

    def _convert_entity_card(self, c: dict) -> OverlayEntity | OverlaySkipped:
        if c.get("status") != "candidate":
            return OverlaySkipped("entity_card", "status_not_candidate", c.get("label", ""))
        if c.get("risk") != "low":
            return OverlaySkipped("entity_card", "risk_not_low", c.get("label", ""))
        if not c.get("label"):
            return OverlaySkipped("entity_card", "missing_label", "")
        if not c.get("entity_id"):
            return OverlaySkipped("entity_card", "missing_entity_id", c.get("label", ""))
        return OverlayEntity(
            entity_id=c["entity_id"],
            label=c["label"],
            aliases=list(c.get("aliases") or []),
            entity_type=c.get("entity_type", "other"),
            source_page=c.get("source_page", ""),
            source_candidate_type="entity_card",
            trust="overlay_candidate",
            risk="low",
        )

    def _convert_definition(self, c: dict) -> OverlayDefinition | OverlaySkipped:
        if c.get("risk") != "low":
            return OverlaySkipped("definition_claim", "risk_not_low", c.get("subject", ""))
        if c.get("stability") != "stable":
            return OverlaySkipped("definition_claim", "stability_not_stable", c.get("subject", ""))
        if not (c.get("subject") and c.get("object") and c.get("evidence_text")):
            return OverlaySkipped("definition_claim", "missing_required_field", c.get("subject", ""))
        return OverlayDefinition(
            subject=c["subject"],
            definition=c["object"],
            predicate=c.get("predicate", "is_a"),
            source_page=c.get("source_page", ""),
            evidence_text=c["evidence_text"],
            trust="overlay_candidate",
            risk="low",
            stability="stable",
            temporal_class="historical",
        )

    def _convert_relation(self, c: dict) -> OverlayRelation | OverlaySkipped:
        risk = c.get("risk", "")
        stability = c.get("stability", "")
        if risk not in ("low", "medium"):
            return OverlaySkipped("relation_claim", f"risk_not_allowed:{risk}", c.get("subject", ""))
        if stability not in ("stable", "semi_stable"):
            return OverlaySkipped("relation_claim", f"stability_not_allowed:{stability}", c.get("subject", ""))
        if not (c.get("subject") and c.get("object") and c.get("predicate") and c.get("evidence_text")):
            return OverlaySkipped("relation_claim", "missing_required_field", c.get("subject", ""))
        temporal_class = c.get("temporal_class") or classify_temporal_class(
            c.get("predicate"), stability
        )
        if temporal_class is None:
            return OverlaySkipped("relation_claim", "temporal_class_requires_review", c.get("subject", ""))
        if requires_as_of(temporal_class) and not c.get("as_of"):
            reason = "aggregate_requires_as_of" if temporal_class == "aggregate" else "snapshot_requires_as_of"
            return OverlaySkipped("relation_claim", reason, c.get("subject", ""))
        return OverlayRelation(
            subject=c["subject"],
            predicate=c["predicate"],
            object=c["object"],
            source_page=c.get("source_page", ""),
            evidence_text=c["evidence_text"],
            trust="overlay_candidate",
            risk=risk,  # type: ignore[arg-type]
            stability=stability,  # type: ignore[arg-type]
            temporal_class=temporal_class,  # type: ignore[arg-type]
            as_of=c.get("as_of", ""),
        )

    def _convert_context_link(self, c: dict) -> OverlayContextLink | OverlaySkipped:
        if c.get("strength") != "weak":
            return OverlaySkipped("context_link", "strength_not_weak", c.get("source_page", ""))
        if not (c.get("source_page") and c.get("surface") and c.get("target")):
            return OverlaySkipped("context_link", "missing_required_field", c.get("source_page", ""))
        return OverlayContextLink(
            source_page=c["source_page"],
            surface=c["surface"],
            target=c["target"],
            relation=c.get("relation", "mentioned_with"),
            strength="weak",
            trust="weak_context_only",
        )

    def _convert_source_fact(self, c: dict) -> OverlaySourceFact | OverlaySkipped:
        if not c.get("source_name"):
            return OverlaySkipped("source_qualified_fact", "missing_source_name", c.get("subject", ""))
        if not c.get("requires_recheck", False):
            return OverlaySkipped("source_qualified_fact", "requires_recheck_false", c.get("subject", ""))
        if c.get("stability") != "volatile":
            return OverlaySkipped("source_qualified_fact", "stability_not_volatile", c.get("subject", ""))
        if c.get("risk") != "high":
            return OverlaySkipped("source_qualified_fact", "risk_not_high", c.get("subject", ""))
        temporal_class = c.get("temporal_class") or classify_temporal_class(
            c.get("predicate"),
            c.get("stability"),
            overlay_type="overlay_source_fact",
            claim_type=c.get("claim_type"),
        )
        if temporal_class is None:
            return OverlaySkipped("source_qualified_fact", "temporal_class_requires_review", c.get("subject", ""))
        if requires_as_of(temporal_class) and not c.get("as_of"):
            reason = "aggregate_requires_as_of" if temporal_class == "aggregate" else "snapshot_requires_as_of"
            return OverlaySkipped("source_qualified_fact", reason, c.get("subject", ""))
        return OverlaySourceFact(
            subject=c.get("subject", ""),
            predicate=c.get("predicate", ""),
            object=c.get("object", ""),
            source_name=c["source_name"],
            as_of=c["as_of"],
            claim_type=c.get("claim_type", "time_sensitive_estimate"),
            temporal_class=temporal_class,  # type: ignore[arg-type]
            source_page=c.get("source_page", ""),
            evidence_text=c.get("evidence_text", ""),
            requires_recheck=True,
            trust="source_qualified_overlay_candidate",
            risk="high",
            stability="volatile",
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def _build_summary(
        self, candidates: list[dict], items: list, skipped: list[OverlaySkipped]
    ) -> dict:
        by_overlay_type: Counter[str] = Counter()
        by_risk: Counter[str] = Counter()
        by_stability: Counter[str] = Counter()
        by_temporal_class: Counter[str] = Counter()

        for item in items:
            by_overlay_type[getattr(item, "overlay_type", "unknown")] += 1
            r = getattr(item, "risk", None)
            if r:
                by_risk[r] += 1
            s = getattr(item, "stability", None)
            if s:
                by_stability[s] += 1
            tc = getattr(item, "temporal_class", None)
            if tc:
                by_temporal_class[tc] += 1

        source_facts = by_overlay_type.get("overlay_source_fact", 0)
        context_links = by_overlay_type.get("overlay_context_link", 0)
        validation_passed = len(skipped) == 0 or all(
            sk.reason not in ("missing_label", "missing_entity_id", "missing_required_field")
            for sk in skipped
        )

        return {
            "source_candidates_total": len(candidates),
            "overlay_items_total": len(items),
            "skipped_candidates_total": len(skipped),
            "by_overlay_type": dict(sorted(by_overlay_type.items())),
            "by_risk": dict(sorted(by_risk.items())),
            "by_stability": dict(sorted(by_stability.items())),
            "by_temporal_class": dict(sorted(by_temporal_class.items())),
            "source_facts_count": source_facts,
            "weak_context_links_count": context_links,
            "safe_for_general_runtime": SAFE_FOR_GENERAL_RUNTIME,
            "safe_for_entity_qa_overlay": validation_passed,
        }

    def load_candidates(self, path: str | Path) -> list[dict]:
        return json.loads(Path(path).read_text(encoding="utf-8"))
