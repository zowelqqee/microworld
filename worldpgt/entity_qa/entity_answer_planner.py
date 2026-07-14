"""Entity answer planner v1.

Builds an EntityQAPlan from an analyzed question and the overlay provider.
Deterministic rule-based only. No ML. No network.
"""

from __future__ import annotations

import re
from typing import Optional

from worldpgt.entity_qa.synthesis_engine import synthesize
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
    "an accepted source-qualified fact in my knowledge base."
)

_PERSONAL_AUDIT_REASON = (
    "This question asks for personal or unsupported information not present "
    "in my knowledge base."
)

_NO_DATA_AUDIT_REASON = (
    "No relevant information found for this question."
)

_WELL_COVERED_NOT_FOUND_REASON = (
    "not found in well-covered entity, verify externally"
)

_QUESTION_NOT_UNDERSTOOD_REASON = "question not understood"

_UNSUPPORTED_SEMANTIC_QUERY_REASON = (
    "This structured question type is not supported by the entity QA planner."
)

_ANSWERABLE_STABILITIES = frozenset({"stable", "semi_stable"})
_LED_BY_RE = re.compile(
    r"\bled\s+by\s+([A-Z][A-Za-z.'-]+(?:\s+[A-Z][A-Za-z.'-]+){0,4})\b"
)
_LANGUAGE_OR_ADJECTIVE_LOCATIONS = frozenset({
    "arabic",
    "chinese",
    "dutch",
    "english",
    "french",
    "german",
    "greek",
    "hebrew",
    "italian",
    "japanese",
    "korean",
    "latin",
    "portuguese",
    "russian",
    "spanish",
    "turkish",
})
_ABSTRACT_LOCATION_OBJECT_TOKENS = frozenset({
    "concept",
    "concepts",
    "culture",
    "discipline",
    "disciplines",
    "economics",
    "field",
    "fields",
    "language",
    "languages",
    "literature",
    "medicine",
    "philosophy",
    "practice",
    "practices",
    "religion",
    "science",
    "teachings",
    "theory",
    "tradition",
    "traditions",
})
_NEIGHBORHOOD_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_NEIGHBORHOOD_MAX_PREDICATE_GROUPS = 3
_NEIGHBORHOOD_MAX_EDGES_PER_GROUP = 2


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


def _is_directly_renderable_relation(item: dict) -> bool:
    stability = str(item.get("stability") or "")
    return (
        stability in _ANSWERABLE_STABILITIES
        and str(item.get("risk") or "").lower() != "high"
    )


def _is_bad_located_in_relation(item: dict) -> bool:
    if item.get("predicate") != "located_in":
        return False
    source_page = _norm(str(item.get("source_page") or ""))
    subject = _norm(str(item.get("subject") or ""))
    if source_page and subject and source_page != subject:
        return True
    obj = _norm(str(item.get("object") or ""))
    if not obj:
        return False
    evidence = _norm(str(item.get("evidence_text") or item.get("evidence_span") or ""))
    if " " not in obj and obj in _LANGUAGE_OR_ADJECTIVE_LOCATIONS:
        return f"in {obj}" in f" {evidence} "
    obj_tokens = set(obj.split())
    if obj_tokens.intersection(_ABSTRACT_LOCATION_OBJECT_TOKENS):
        return True
    return False


def _filter_renderable_relations(relations: list[dict]) -> list[dict]:
    return [r for r in relations if not _is_bad_located_in_relation(r)]


def _neighborhood_terms(text: str) -> set[str]:
    return {
        token.casefold()
        for token in _NEIGHBORHOOD_TOKEN_RE.findall(text or "")
        if len(token) > 2
    }


class EntityAnswerPlanner:
    def __init__(
        self,
        provider: WikiMemoryOverlayProvider,
        ontology_layer_items: list[dict] | None = None,
        inference_workspace=None,
    ) -> None:
        self._provider = provider
        self._ontology_layer_items = list(ontology_layer_items or [])
        self._inference_workspace = inference_workspace

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

        if intent == "open_synthesis":
            return self._plan_open_synthesis(analyzed)
        if intent == "define_entity":
            return self._plan_define(analyzed)
        if intent == "relation_lookup":
            return self._plan_relation(analyzed)
        if intent == "link_explanation":
            return self._plan_link_explanation(analyzed)
        if intent == "source_fact_lookup":
            return self._plan_source_fact(analyzed)

        return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

    def plan_outgoing_neighborhood(self, question: str, subject: str) -> EntityQAPlan:
        """Select a bounded direct-edge neighborhood without a predicate schema."""
        analyzed = AnalyzedEntityQuestion(
            question=question, intent="relation_lookup", subject=subject,
            predicate_hint=None, secondary_entity=None, source_hint=None,
            is_current_query=False, is_unsupported=False,
        )
        evidence = EntityQAEvidence()
        relations = [
            relation
            for relation in _filter_renderable_relations(self._provider.get_relations(subject))
            if _is_answerable_relation(relation)
        ]
        if not relations:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        question_terms = _neighborhood_terms(question) - _neighborhood_terms(subject)
        groups: dict[str, list[dict]] = {}
        for relation in relations:
            predicate = str(relation.get("predicate") or "")
            if predicate:
                groups.setdefault(predicate, []).append(relation)
        if not groups:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        def edge_score(relation: dict) -> int:
            edge_text = " ".join(
                str(relation.get(field) or "")
                for field in ("predicate", "object", "evidence_text", "evidence_span")
            )
            return len(question_terms.intersection(_neighborhood_terms(edge_text)))

        ranked_groups: list[tuple[int, int, str, list[dict]]] = []
        for predicate, members in groups.items():
            ranked_members = sorted(
                members,
                key=lambda relation: (
                    -edge_score(relation),
                    -int(relation.get("support_count") or 1),
                    str(relation.get("object") or "").casefold(),
                ),
            )
            ranked_groups.append((
                edge_score(ranked_members[0]), len(ranked_members), predicate, ranked_members,
            ))
        ranked_groups.sort(key=lambda group: (-group[0], -group[1], group[2]))

        # A broad unscored graph cannot be narrowed honestly.  A single
        # predicate group is already an unambiguous direct neighborhood.
        if ranked_groups[0][0] == 0 and len(ranked_groups) != 1:
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        selected: list[dict] = []
        eligible_groups = (
            [group for group in ranked_groups if group[0] > 0]
            if ranked_groups[0][0] > 0
            else ranked_groups[:1]
        )
        for _score, _size, _predicate, members in eligible_groups[:_NEIGHBORHOOD_MAX_PREDICATE_GROUPS]:
            selected.extend(members[:_NEIGHBORHOOD_MAX_EDGES_PER_GROUP])
        for relation in selected:
            evidence.overlay_items_used.append(
                f"overlay_relation:{relation['predicate']}:{relation['object']}"
            )
        return EntityQAPlan(
            analyzed=analyzed, decision="answer", audit_reason=None, evidence=evidence,
            render_template="relation_lookup",
            render_args={
                "subject": subject, "predicate": None, "relations": selected, "founder_lookup": False,
            },
            confidence=0.82,
        )

    # ------------------------------------------------------------------
    # Intent-specific planners
    # ------------------------------------------------------------------

    def _plan_open_synthesis(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        """Layer 3: synthesize an answer from every fact known about the entity."""
        subject = analyzed.subject or ""
        result = synthesize(
            self._provider,
            subject,
            analyzed.question,
            workspace=self._inference_workspace,
        )

        # Nothing in the graph (no entity, no keyword neighbour, no facts) — be
        # honest and audit rather than emit an empty answer.
        if not result.matched or (
            result.definition is None
            and not result.entity_type
            and not result.groups
        ):
            return self._audit_plan(analyzed, _NO_DATA_AUDIT_REASON)

        evidence = EntityQAEvidence()
        if result.definition:
            evidence.overlay_items_used.append(f"overlay_definition:{result.subject}")
        for group in result.groups:
            if group.kind == "snapshot":
                for obj in group.objects:
                    evidence.source_facts_used.append(
                        f"source_fact:{result.subject}:{group.predicate}:{group.source_name}"
                    )
            elif group.tier == "INFERRED":
                for obj in group.objects:
                    evidence.overlay_items_used.append(
                        f"inferred_rule:{group.rule}:{group.predicate}:{obj}"
                    )
            else:
                for obj in group.objects:
                    evidence.overlay_items_used.append(
                        f"overlay_relation:{group.predicate}:{obj}"
                    )

        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="open_synthesis",
            render_args={"synthesis": result, "question": analyzed.question},
            confidence=0.9,
        )

    def _plan_define(self, analyzed: AnalyzedEntityQuestion) -> EntityQAPlan:
        subject = analyzed.subject or ""
        evidence = EntityQAEvidence()

        entity = self._provider.get_entity(subject)
        definition = self._provider.get_definition(subject)

        if entity:
            evidence.overlay_items_used.append(f"overlay_entity:{entity['label']}")
        if definition:
            evidence.overlay_items_used.append(f"overlay_definition:{definition['subject']}")

        relations = _filter_renderable_relations(self._provider.get_relations(subject))
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

        founder_lookup_question = predicate in {"founded_by", "founded"} and _asks_who_founded(analyzed.question)

        if (
            analyzed.semantic_query is not None
            and analyzed.semantic_query.unknown_position == "subject"
            and predicate
            and not founder_lookup_question
        ):
            all_relations = [
                r for r in self._provider.all_relations()
                if r.get("predicate") == predicate
                and _norm(r.get("object", "")) == _norm(subject)
                and (
                    _is_answerable_relation(r)
                    or (predicate == "leader_of" and _is_directly_renderable_relation(r))
                )
            ]
            self._provider._items_used += len(all_relations)
            for r in all_relations:
                evidence.overlay_items_used.append(
                    f"overlay_relation:{r['subject']}:{r['predicate']}:{r['object']}"
                )
            if not all_relations:
                if predicate == "leader_of":
                    definition_plan = self._plan_leader_from_definition(analyzed, subject)
                    if definition_plan is not None:
                        return definition_plan
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

        all_relations = _filter_renderable_relations(all_relations)

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

    def _plan_leader_from_definition(
        self,
        analyzed: AnalyzedEntityQuestion,
        subject: str,
    ) -> Optional[EntityQAPlan]:
        definition = self._provider.get_definition(subject)
        if not definition:
            return None
        definition_text = str(definition.get("definition") or "")
        evidence_text = str(definition.get("evidence_text") or "")
        support_text = evidence_text or definition_text
        match = _LED_BY_RE.search(support_text)
        if not match:
            return None

        leader = match.group(1).strip().rstrip(".,;:")
        if not leader:
            return None

        evidence = EntityQAEvidence()
        evidence.overlay_items_used.append(f"overlay_definition:{subject}")
        return EntityQAPlan(
            analyzed=analyzed,
            decision="answer",
            audit_reason=None,
            evidence=evidence,
            render_template="definition_relation_lookup",
            render_args={
                "subject": subject,
                "predicate": "leader_of",
                "object": leader,
                "definition": definition,
            },
            confidence=0.8,
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
