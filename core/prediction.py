from dataclasses import dataclass, field
from collections import Counter
from typing import Optional, TYPE_CHECKING
from .relations import Relation

if TYPE_CHECKING:
    from .concepts import Concept
    from .similarity import SimilarityGraph

# v0.5: one supported template — the 3-step lifecycle cycle.
LIFECYCLE_RELS: tuple[str, ...] = ("produces_result", "contains", "grows_into")


@dataclass
class ChainTemplate:
    """A repeating cycle pattern extracted from the relation graph."""
    relation_sequence: tuple[str, ...]
    instances: list[list[str]]

    def slot_values(self, idx: int) -> list[str]:
        return [inst[idx] for inst in self.instances]

    def most_common_slot_value(self, idx: int) -> str | None:
        values = self.slot_values(idx)
        if not values:
            return None
        return Counter(values).most_common(1)[0][0]


@dataclass
class Prediction:
    source: str
    relation_type: str
    target: str
    confidence: float
    reason: str
    evidence: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"Prediction({self.source!r} --{self.relation_type}--> {self.target!r}, "
            f"conf={self.confidence:.2f})"
        )


class PredictionEngine:
    """
    Infers likely missing relations by structural analogy over chain templates.

    v0.9 inference priority for connector selection:
      1. Concept-aware  — find concept whose members are most similar to
                          the partial product; use that concept's pattern.
      2. Similarity     — directly find nearest complete instance via score.
      3. Majority vote  — use the most common connector across all instances.
    """

    def __init__(
        self,
        relations: list[Relation],
        similarity_graph: Optional["SimilarityGraph"] = None,
        concepts: Optional[list["Concept"]] = None,
    ) -> None:
        self._relations = relations
        self._existing: frozenset[tuple[str, str, str]] = frozenset(
            (r.source, r.relation_type, r.target) for r in relations
        )
        self._similarity_graph = similarity_graph
        self._concepts: list["Concept"] = concepts or []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_missing_links(self) -> list[Prediction]:
        return self.apply_templates_to_partial_matches()

    def find_chain_templates(self) -> list[ChainTemplate]:
        templates: list[ChainTemplate] = []
        instances = self._find_cycle_instances(LIFECYCLE_RELS)
        if instances:
            templates.append(ChainTemplate(
                relation_sequence=LIFECYCLE_RELS,
                instances=instances,
            ))
        return templates

    def apply_templates_to_partial_matches(self) -> list[Prediction]:
        predictions: list[Prediction] = []
        for template in self.find_chain_templates():
            predictions.extend(self._apply_template(template))
        return predictions

    def explain_prediction(self, prediction: Prediction) -> str:
        lines = [
            f"Prediction: {prediction.source} --{prediction.relation_type}--> {prediction.target}",
            f"  confidence : {prediction.confidence:.2f}",
            f"  reason     : {prediction.reason}",
        ]
        if prediction.evidence:
            lines.append("  based on:")
            for ev in prediction.evidence:
                lines.append(f"    • {ev}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_cycle_instances(self, rel_sequence: tuple[str, ...]) -> list[list[str]]:
        assert len(rel_sequence) == 3, "only 3-step cycles supported"
        r0, r1, r2 = rel_sequence
        instances: list[list[str]] = []
        for rel0 in self._relations:
            if rel0.relation_type != r0:
                continue
            A, B = rel0.source, rel0.target
            for rel1 in self._relations:
                if rel1.source != B or rel1.relation_type != r1:
                    continue
                C = rel1.target
                if C == A:
                    continue
                for rel2 in self._relations:
                    if rel2.source == C and rel2.relation_type == r2 and rel2.target == A:
                        instances.append([A, B, C])
        return instances

    def _apply_template(self, template: ChainTemplate) -> list[Prediction]:
        if len(template.relation_sequence) != 3 or not template.instances:
            return []

        r0, r1, r2 = template.relation_sequence
        product_to_connector: dict[str, str] = {
            inst[1]: inst[2] for inst in template.instances
        }
        complete_pairs = {(inst[0], inst[1]) for inst in template.instances}

        # Pre-build majority fallback values once.
        majority_connector = template.most_common_slot_value(2)
        n = len(template.instances)
        majority_confidence = min(0.95, 0.7 + 0.1 * (n - 1))
        majority_evidence = [
            f"{inst[0]} -{r0}-> {inst[1]} -{r1}-> {inst[2]} -{r2}-> {inst[0]}"
            for inst in template.instances
        ]

        predictions: list[Prediction] = []
        for rel in self._relations:
            if rel.relation_type != r0:
                continue
            entity, product = rel.source, rel.target
            if (entity, product) in complete_pairs:
                continue

            connector, confidence, reason, evidence = self._pick_connector(
                product=product,
                entity=entity,
                product_to_connector=product_to_connector,
                majority_connector=majority_connector,
                majority_confidence=majority_confidence,
                majority_evidence=majority_evidence,
                r0=r0, r1=r1, r2=r2,
                template=template,
            )
            if connector is None:
                continue

            if (product, r1, connector) not in self._existing:
                predictions.append(Prediction(
                    source=product,
                    relation_type=r1,
                    target=connector,
                    confidence=confidence,
                    reason=reason,
                    evidence=list(evidence),
                ))

            if (connector, r2, entity) not in self._existing:
                predictions.append(Prediction(
                    source=connector,
                    relation_type=r2,
                    target=entity,
                    confidence=confidence,
                    reason=reason,
                    evidence=list(evidence),
                ))

        return predictions

    def _pick_connector(
        self,
        product: str,
        entity: str,
        product_to_connector: dict[str, str],
        majority_connector: str | None,
        majority_confidence: float,
        majority_evidence: list[str],
        r0: str, r1: str, r2: str,
        template: ChainTemplate,
    ) -> tuple[str | None, float, str, list[str]]:
        """
        Return (connector, confidence, reason, evidence) using v0.9 priority:
          1. Concept-aware  — direct membership OR similarity-to-members.
                              Does NOT require a similarity_graph when the
                              product is already a concept member itself.
          2. Similarity     — nearest complete instance via similarity score.
          3. Majority vote  — most common connector across all instances.
        """
        # ── 1. Concept-aware ────────────────────────────────────────────
        if self._concepts:
            concept, score = self._find_best_concept(product)
            if concept is not None and score > 0.0:
                connector = self._concept_connector(concept)
                if connector is not None:
                    confidence = min(0.95, 0.85 + 0.02 * len(concept.members))
                    reason = f"concept match: {concept.id}"
                    evidence = [
                        f"{inst[0]} -{r0}-> {inst[1]} -{r1}-> {inst[2]} -{r2}-> {inst[0]}"
                        for inst in template.instances
                        if inst[2] == connector
                    ]
                    return connector, confidence, reason, evidence

        # ── 2. Direct similarity ─────────────────────────────────────────
        if self._similarity_graph is not None:
            known_products = list(product_to_connector.keys())
            best_match, score = self._similarity_graph.most_similar(product, known_products)
            if best_match is not None and score > 0.0:
                connector = product_to_connector[best_match]
                confidence = min(0.95, 0.65 + 0.30 * score)
                reason = (
                    f"nearest lifecycle match: {product} is similar to "
                    f"{best_match} (score={score:.2f})"
                )
                evidence = [
                    f"{inst[0]} -{r0}-> {inst[1]} -{r1}-> {inst[2]} -{r2}-> {inst[0]}"
                    for inst in template.instances
                    if inst[1] == best_match
                ]
                return connector, confidence, reason, evidence

        # ── 3. Majority vote fallback ────────────────────────────────────
        if majority_connector is None:
            return None, 0.0, "", []
        reason = (
            f"fallback majority template [{r0} → {r1} → {r2}]: "
            f"{entity} {r0} {product}, analogous to "
            + " | ".join(f"{inst[0]}->{inst[1]}" for inst in template.instances)
        )
        return majority_connector, majority_confidence, reason, majority_evidence

    def _find_best_concept(self, product: str) -> tuple[Optional["Concept"], float]:
        """
        Among all relation-group concepts, return the one best matching *product*.

        Priority per concept:
          - Direct membership (product in concept.members) → score 1.0, no similarity needed.
          - Similarity-based: max similarity between product and any concept member.

        Returns (None, 0.0) if no concept matches.
        """
        best_concept = None
        best_score = 0.0
        for concept in self._concepts:
            if concept.kind != "relation_group":
                continue
            # Direct membership is the strongest signal — no similarity required.
            if product in concept.members:
                return concept, 1.0
            # Fall back to similarity lookup.
            if self._similarity_graph is not None:
                _, score = self._similarity_graph.most_similar(product, concept.members)
                if score > best_score:
                    best_score = score
                    best_concept = concept
        return best_concept, best_score

    def _concept_connector(self, concept: "Concept") -> str | None:
        """Extract the target value from a 'rel_type -> target' pattern label."""
        for pattern in concept.common_relations:
            parts = pattern.split(" -> ")
            if len(parts) == 2:
                return parts[1]
        return None
