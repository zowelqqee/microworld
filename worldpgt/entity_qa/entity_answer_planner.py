"""Entity answer planner v1.

Builds an EntityQAPlan from an analyzed question and the overlay provider.
Deterministic rule-based only. No ML. No network.
"""

from __future__ import annotations

from typing import Optional

from worldpgt.entity_qa.types import (
    AnalyzedEntityQuestion,
    EntityQAEvidence,
    EntityQAPlan,
)
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_CURRENT_AUDIT_REASON = (
    "This question asks for current/real-time data that is not available as "
    "an accepted source-qualified fact in the overlay."
)

_PERSONAL_AUDIT_REASON = (
    "This question asks for personal or unsupported information not present "
    "in the wiki overlay."
)

_NO_DATA_AUDIT_REASON = (
    "No relevant information found in the wiki overlay for this question."
)


class EntityAnswerPlanner:
    def __init__(self, provider: WikiMemoryOverlayProvider) -> None:
        self._provider = provider

    def plan(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        intent = analyzed.intent

        if analyzed.is_unsupported or intent == "unknown_or_unsupported":
            return self._audit_plan(
                analyzed,
                _CURRENT_AUDIT_REASON if analyzed.is_current_query else _PERSONAL_AUDIT_REASON,
            )

        if intent == "define_entity":
            return self._plan_define(analyzed)
        if intent == "relation_lookup":
            return self._plan_relation(analyzed)
        if intent == "link_explanation":
            return self._plan_link_explanation(analyzed)
        if intent == "source_fact_lookup":
            return self._plan_source_fact(analyzed)

        return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

    # ------------------------------------------------------------------
    # Intent-specific planners
    # ------------------------------------------------------------------

    def _plan_define(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        subject = analyzed.subject or ""
        evidence = EntityQAEvidence()

        entity = self._provider.get_entity(subject)
        definition = self._provider.get_definition(subject)

        if entity:
            evidence.overlay_items_used.append(f"overlay_entity:{entity['label']}")
        if definition:
            evidence.overlay_items_used.append(f"overlay_definition:{definition['subject']}")

        relations = self._provider.get_relations(subject)
        if relations:
            for r in relations:
                evidence.overlay_items_used.append(
                    f"overlay_relation:{r['predicate']}:{r['object']}"
                )

        if not entity and not definition:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="define_entity",
            render_args={
                "entity": entity,
                "definition": definition,
                "relations": relations,
                "subject": subject,
            },
            confidence=0.95,
        )

    def _plan_relation(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        subject = analyzed.subject or ""
        predicate = analyzed.predicate_hint
        evidence = EntityQAEvidence()

        if predicate == "founded":
            # "Who founded X?" — look for founded relations with X as object
            all_relations = [
                r for r in self._provider._relations
                if r.get("predicate") == "founded"
                and r.get("object", "").lower() == subject.lower()
            ]
            self._provider._items_used += len(all_relations)
        else:
            all_relations = self._provider.get_relations(subject, predicate)

        if not all_relations and predicate in ("known_for", "leader_of"):
            # Try any relation for this subject
            all_relations = self._provider.get_relations(subject)

        for r in all_relations:
            evidence.overlay_items_used.append(
                f"overlay_relation:{r['predicate']}:{r['object']}"
            )

        if not all_relations:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        founder_lookup = (predicate == "founded")

        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="relation_lookup",
            render_args={
                "subject": subject,
                "predicate": predicate,
                "relations": all_relations,
                "founder_lookup": founder_lookup,
            },
            confidence=0.9,
        )

    def _plan_link_explanation(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        subject = analyzed.subject or ""
        secondary = analyzed.secondary_entity or ""
        evidence = EntityQAEvidence()

        # Look for context links from subject page mentioning secondary
        links = self._provider.get_context_links(source_page=subject, target=secondary)
        if not links:
            # Try reversed
            links = self._provider.get_context_links(source_page=secondary, target=subject)
        if not links:
            links = self._provider.get_context_links(target=secondary)

        for lnk in links:
            evidence.weak_context_links_used.append(
                f"context_link:{lnk['source_page']}->{lnk['target']}"
            )

        if not links:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="link_explanation",
            render_args={
                "subject": subject,
                "secondary": secondary,
                "links": links,
            },
            confidence=0.85,
        )

    def _plan_source_fact(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        subject = analyzed.subject or ""
        predicate_hint = analyzed.predicate_hint or ""
        source_hint = analyzed.source_hint
        evidence = EntityQAEvidence()

        facts = self._provider.get_source_facts(
            subject=subject,
            source_name=source_hint,
        )

        if not facts and subject:
            # Try alias resolution
            entity = self._provider.get_entity(subject)
            if entity:
                facts = self._provider.get_source_facts(subject=entity["label"])

        for f in facts:
            evidence.source_facts_used.append(
                f"source_fact:{f['subject']}:{f['predicate']}:{f['source_name']}"
            )

        if predicate_hint in ("stability_check", "recheck_reason"):
            return EntityQAPlan(
                analyzed=analyzed,
                decision="answer",
                audit_reason=None,
                evidence=evidence,
                render_template="stability_check",
                render_args={
                    "subject": subject,
                    "facts": facts,
                    "predicate_hint": predicate_hint,
                },
                confidence=0.9,
            )

        if not facts:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="source_fact_lookup",
            render_args={
                "subject": subject,
                "source_hint": source_hint,
                "facts": facts,
            },
            confidence=0.9,
        )

    # ------------------------------------------------------------------
    # Audit helper
    # ------------------------------------------------------------------

    def _audit_plan(
        self, analyzed: AnalyzedEntityQuestion, reason: str
    ) -> EntityQAPlan:
        return EntityQAPlan(
            analyzed=analyzed,
            decision="audit",
            audit_reason=reason,
            evidence=EntityQAEvidence(),
            render_template="audit",
            render_args={"reason": reason, "question": analyzed.question},
            confidence=0.0,
        )
