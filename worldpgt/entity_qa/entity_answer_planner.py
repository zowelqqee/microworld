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


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split()).removeprefix("the ")


def _asks_who_founded(question: str) -> bool:
    return (question or "").strip().lower().startswith("who ")


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
        is_a_relations = [r for r in relations if r.get("predicate") == "is_a"]
        if relations:
            for r in relations:
                evidence.overlay_items_used.append(
                    f"overlay_relation:{r['predicate']}:{r['object']}"
                )

        if not entity and not definition and not is_a_relations:
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

        if predicate == "connection":
            # "How is X connected to Y?" — relations of X whose object is Y.
            secondary = (analyzed.secondary_entity or "").strip().lower()
            all_relations = [
                r for r in self._provider.get_relations(subject)
                if r.get("object", "").strip().lower() == secondary
            ]
            for r in all_relations:
                evidence.overlay_items_used.append(
                    f"overlay_relation:{r['predicate']}:{r['object']}"
                )
            if not all_relations:
                return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)
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
                    "founder_lookup": False,
                },
                confidence=0.9,
            )

        founder_lookup_question = predicate in {"founded_by", "founded"} and _asks_who_founded(analyzed.question)

        if predicate == "founded_by" or founder_lookup_question:
            # "Who founded X?" — prefer X founded_by Y; also support older
            # founder-as-subject facts: Y founded X.
            all_relations = self._provider.get_relations(subject, "founded_by")
            inverse = [
                r for r in self._provider.all_relations()
                if r.get("predicate") == "founded" and _norm(r.get("object", "")) == _norm(subject)
            ]
            self._provider._items_used += len(inverse)
            all_relations.extend(inverse)
        elif predicate == "founded":
            # "What did X found?" — subject is the founder.
            all_relations = [
                r for r in self._provider.all_relations()
                if r.get("predicate") == "founded"
                and _norm(r.get("subject", "")) == _norm(subject)
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

        founder_lookup = founder_lookup_question or predicate == "founded_by"

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
        # Meta question about overlay link policy — answer without a specific link.
        if analyzed.predicate_hint == "weak_link_policy":
            return EntityQAPlan(
                analyzed=analyzed,
                decision="answer",
                audit_reason=None,
                evidence=EntityQAEvidence(),
                render_template="link_policy",
                render_args={},
                confidence=0.95,
            )

        subject = analyzed.subject or ""
        secondary = analyzed.secondary_entity or ""
        evidence = EntityQAEvidence()

        # Candidate surface/target forms: the secondary as written, plus a
        # simple depluralized variant ("rockets" -> "rocket") so plural
        # phrasings still resolve to the singular overlay link target/surface.
        candidates = [secondary]
        if secondary.endswith("s") and len(secondary) > 3:
            candidates.append(secondary[:-1])

        links: list[dict] = []
        # 1. target match on the subject page
        for cand in candidates:
            links = self._provider.get_context_links(source_page=subject, target=cand)
            if links:
                break
        # 2. surface match on the subject page
        if not links:
            for cand in candidates:
                links = self._provider.get_context_links(source_page=subject, surface=cand)
                if links:
                    break
        # 3. reversed: subject mentioned on the secondary page
        if not links:
            for cand in candidates:
                links = self._provider.get_context_links(source_page=cand, target=subject)
                if links:
                    break
        # 4. target match anywhere
        if not links:
            for cand in candidates:
                links = self._provider.get_context_links(target=cand)
                if links:
                    break

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

        if predicate_hint in ("stability_check", "recheck_reason", "source_qualified_confirm"):
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
