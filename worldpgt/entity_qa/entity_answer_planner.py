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
from worldpgt.knowledge.negation import (
    classes_contradict,
    coverage_score,
    item_is_safe_negation_basis,
    well_covered,
)
from worldpgt.knowledge.ontology_traversal import find_is_a_path
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.relation_extraction_v2.relation_policy import is_current_sensitive

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

_WELL_COVERED_NOT_FOUND_REASON = (
    "not found in well-covered entity, verify externally"
)

_QUESTION_NOT_UNDERSTOOD_REASON = "question not understood"

_UNSUPPORTED_SEMANTIC_QUERY_REASON = (
    "This structured question type is not supported by the entity QA planner."
)

_ANSWERABLE_STABILITIES = frozenset({"stable", "semi_stable"})


def _norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split()).removeprefix("the ")


def _asks_who_founded(question: str) -> bool:
    return (question or "").strip().lower().startswith("who ")


def _is_answerable_relation(item: dict) -> bool:
    predicate = str(item.get("predicate") or "")
    stability = str(item.get("stability") or "")
    return (
        stability in _ANSWERABLE_STABILITIES
        and not is_current_sensitive(predicate)
        and str(item.get("risk") or "").lower() != "high"
    )


class EntityAnswerPlanner:
    def __init__(
        self,
        provider: WikiMemoryOverlayProvider,
        ontology_layer_items: list[dict] | None = None,
    ) -> None:
        self._provider = provider
        self._ontology_layer_items = list(ontology_layer_items or [])

    def plan(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        intent = analyzed.intent

        if analyzed.is_unsupported or intent == "unknown_or_unsupported":
            if analyzed.predicate_hint == "question_not_understood":
                reason = _QUESTION_NOT_UNDERSTOOD_REASON
            elif analyzed.predicate_hint == "intersection":
                reason = _UNSUPPORTED_SEMANTIC_QUERY_REASON
            else:
                reason = (
                    _CURRENT_AUDIT_REASON
                    if analyzed.is_current_query
                    else _PERSONAL_AUDIT_REASON
                )
            return self._audit_plan(
                analyzed,
                reason,
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

        if predicate == "is_a":
            return self._plan_is_a_membership(analyzed)

        if predicate == "intersection":
            return self._plan_comparative_intersection(analyzed)

        if (
            analyzed.semantic_query is not None
            and analyzed.semantic_query.unknown_position == "subject"
            and predicate
        ):
            all_relations = [
                r for r in self._provider.all_relations()
                if r.get("predicate") == predicate
                and _norm(r.get("object", "")) == _norm(subject)
                and _is_answerable_relation(r)
            ]
            self._provider._items_used += len(all_relations)
            for r in all_relations:
                evidence.overlay_items_used.append(
                    f"overlay_relation:{r['subject']}:{r['predicate']}:{r['object']}"
                )
            if not all_relations:
                return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)
            return EntityQAPlan(
                analyzed=analyzed,
                decision="answer",
                audit_reason=None,
                evidence=evidence,
                render_template="inverse_relation_lookup",
                render_args={
                    "known_object": subject,
                    "predicate": predicate,
                    "relations": all_relations,
                },
                confidence=0.9,
            )

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

    def _plan_comparative_intersection(
        self,
        analyzed: AnalyzedEntityQuestion,
    ) -> EntityQAPlan:
        subject = analyzed.subject or ""
        secondary = analyzed.secondary_entity or ""
        evidence = EntityQAEvidence()

        if not subject or not secondary:
            return self._audit_plan(analyzed, "no common facts in current overlay")

        left = self._comparison_facts(subject)
        right = self._comparison_facts(secondary)

        common_pair_keys = set(left["pairs"]).intersection(right["pairs"])
        common_class_keys = set(left["classes"]).intersection(right["classes"])

        common_pairs = [left["pairs"][key] for key in sorted(common_pair_keys)]
        common_classes = [left["classes"][key] for key in sorted(common_class_keys)]

        if not common_pairs and not common_classes:
            return self._audit_plan(analyzed, "no common facts in current overlay")

        for item in common_pairs:
            evidence.overlay_items_used.append(
                f"overlay_relation:{subject}:{item['predicate']}:{item['object']}"
            )
            evidence.overlay_items_used.append(
                f"overlay_relation:{secondary}:{item['predicate']}:{item['object']}"
            )
        for item in common_classes:
            evidence.overlay_items_used.append(f"overlay_is_a:{subject}:{item['class']}")
            evidence.overlay_items_used.append(f"overlay_is_a:{secondary}:{item['class']}")

        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="comparative_intersection",
            render_args={
                "entity_a": subject,
                "entity_b": secondary,
                "common_pairs": common_pairs,
                "common_classes": common_classes,
            },
            confidence=0.85,
        )

    def _comparison_facts(self, subject: str) -> dict[str, dict[str, dict]]:
        pairs: dict[str, dict] = {}
        classes: dict[str, dict] = {}

        definition = self._provider.get_definition(subject)
        if definition and definition.get("definition"):
            cls = str(definition["definition"])
            classes[_norm(cls)] = {"class": cls, "source": "definition"}

        for relation in self._provider.get_relations(subject):
            if not _is_answerable_relation(relation):
                continue
            predicate = str(relation.get("predicate") or "")
            obj = str(relation.get("object") or "")
            if not predicate or not obj:
                continue
            if predicate == "is_a":
                classes[_norm(obj)] = {"class": obj, "source": "relation"}
                continue
            pairs[f"{predicate}\0{_norm(obj)}"] = {
                "predicate": predicate,
                "object": obj,
                "stability": relation.get("stability"),
            }

        return {"pairs": pairs, "classes": classes}

    def _plan_is_a_membership(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        subject = analyzed.subject or ""
        target = analyzed.secondary_entity or ""
        evidence = EntityQAEvidence()

        if not subject or not target:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        overlay_items = [
            *self._provider.all_definitions(),
            *self._provider.all_relations(),
        ]
        path = find_is_a_path(
            subject,
            target,
            overlay_items,
            ontology_layer_items=self._ontology_layer_items,
        )
        if path is None:
            negation = self._find_is_a_negation(analyzed, subject, target)
            if negation is not None:
                return negation
            if well_covered(self._provider, subject):
                score = coverage_score(self._provider, subject)
                return self._audit_plan(
                    analyzed,
                    f"{_WELL_COVERED_NOT_FOUND_REASON} (coverage_score={score})",
                )
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        evidence.overlay_items_used.append(f"explicit is_a chain: {len(path)} hops")
        for edge in path:
            evidence.overlay_items_used.append(
                f"{edge.overlay_type}:{edge.subject}:is_a:{edge.object}"
            )

        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="ontology_is_a",
            render_args={
                "subject": subject,
                "target": target,
                "path": path,
                "support": f"explicit is_a chain: {len(path)} hops",
            },
            confidence=0.9,
        )

    def _find_is_a_negation(
        self,
        analyzed: AnalyzedEntityQuestion,
        subject: str,
        target: str,
    ) -> Optional[EntityQAPlan]:
        """Return an explicit negative plan for incompatible class claims."""

        evidence = EntityQAEvidence()
        candidates: list[tuple[str, str, dict]] = []

        definition = self._provider.get_definition(subject)
        if definition and definition.get("definition"):
            candidates.append(("definition", str(definition["definition"]), definition))

        for relation in self._provider.get_relations(subject, "is_a"):
            if relation.get("object"):
                candidates.append(("is_a", str(relation["object"]), relation))

        for source, actual_class, item in candidates:
            if not item_is_safe_negation_basis(item):
                continue
            if not classes_contradict(actual_class, target):
                continue
            evidence.overlay_items_used.append(
                f"{item.get('overlay_type')}:{subject}:is_a:{actual_class}"
            )
            return self._no_plan(
                analyzed=analyzed,
                subject=subject,
                target=target,
                actual_class=actual_class,
                support_kind="explicit_type_contradiction",
                support=f"explicit type contradiction from overlay {source}",
                evidence=evidence,
            )

        entity = self._provider.get_entity(subject)
        entity_type = str((entity or {}).get("entity_type") or "")
        if entity_type and classes_contradict(entity_type, target):
            entity_item = dict(entity or {})
            entity_item.setdefault("overlay_type", "overlay_entity")
            entity_item.setdefault("stability", "stable")
            if item_is_safe_negation_basis(entity_item):
                evidence.overlay_items_used.append(
                    f"overlay_entity:{subject}:entity_type:{entity_type}"
                )
                return self._no_plan(
                    analyzed=analyzed,
                    subject=subject,
                    target=target,
                    actual_class=entity_type,
                    support_kind="entity_type_mismatch",
                    support="entity type mismatch",
                    evidence=evidence,
                )

        return None

    def _no_plan(
        self,
        *,
        analyzed: AnalyzedEntityQuestion,
        subject: str,
        target: str,
        actual_class: str,
        support_kind: str,
        support: str,
        evidence: EntityQAEvidence,
    ) -> EntityQAPlan:
        return EntityQAPlan(
            analyzed=analyzed,
            decision="no",
            audit_reason=None,
            evidence=evidence,
            render_template="negated_is_a",
            render_args={
                "subject": subject,
                "target": target,
                "actual_class": actual_class,
                "support_kind": support_kind,
                "support": support,
            },
            confidence=0.85,
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
