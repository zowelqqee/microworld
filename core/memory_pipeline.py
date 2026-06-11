"""Application-level memory/reasoning pipeline wrapper for Microworld."""
from __future__ import annotations

from dataclasses import dataclass, field

from .concepts import Concept
from .node_quality import node_quality
from .pattern_prediction import PatternBasedPredictor, PatternPrediction
from .patterns import Pattern, PatternDiscoveryEngine
from .relation_trust import DEFAULT_RELATION_TRUST
from .relations import Relation
from .trust_learning import LABEL_SCORES, label_score
from .world import World

DEFAULT_PIPELINE_RELATION_TRUST: dict[str, float] = {
    **DEFAULT_RELATION_TRUST,
    "uses": 0.90,
    "extends": 0.80,
    "capable_of": 0.85,
    "has_property": 0.80,
    "made_of": DEFAULT_RELATION_TRUST["made_of"],
    "learned_from": 0.75,
    "evaluates": 0.75,
    "affect": 0.75,
    "lowers": 0.70,
    "rejects": 0.70,
}


@dataclass
class PipelineAuditRecord:
    prediction: PatternPrediction
    label: str
    score: float


@dataclass
class PipelineSleepReport:
    concepts: list[Concept]
    patterns: list[Pattern]
    structural_similarities: list[tuple[str, str, float]]
    relation_trust: dict[str, float]
    low_quality_nodes: dict[str, float] = field(default_factory=dict)


@dataclass
class TrustUpdateReport:
    before: dict[str, float]
    after: dict[str, float]
    learned: dict[str, float]
    counts: dict[str, int]


class MicroworldPipeline:
    """
    Small application-level API over World + pattern reasoning + audit learning.

    This wrapper keeps the core World storage API unchanged while exposing the
    memory loop as observe -> sleep -> predict -> audit -> learn -> re-predict.
    """

    def __init__(
        self,
        relation_trust: dict[str, float] | None = None,
        prediction_min_count: int = 1,
        prediction_threshold: float = 0.4,
        use_node_quality: bool = True,
        min_node_quality: float = 0.3,
    ) -> None:
        self.world = World()
        self.relation_trust = dict(relation_trust or DEFAULT_PIPELINE_RELATION_TRUST)
        self.prediction_min_count = prediction_min_count
        self.prediction_threshold = prediction_threshold
        self.use_node_quality = use_node_quality
        self.min_node_quality = min_node_quality
        self.audit_records: list[PipelineAuditRecord] = []
        self.sleep_report: PipelineSleepReport | None = None

    def observe(
        self,
        source: str,
        relation_type: str,
        target: str,
        evidence: str | list[str] | None = None,
    ) -> Relation:
        """Insert an explicit observed relation."""
        evidence_list = _evidence_list(evidence)
        self.world._ensure_object(source)
        self.world._ensure_object(target)
        for relation in self.world._relations:
            if (
                relation.source == source
                and relation.relation_type == relation_type
                and relation.target == target
            ):
                for item in evidence_list:
                    if item not in relation.evidence:
                        relation.evidence.append(item)
                return relation
        relation = Relation(
            source=source,
            relation_type=relation_type,
            target=target,
            evidence=evidence_list,
        )
        self.world.add_relation(relation)
        return relation

    def sleep(self) -> PipelineSleepReport:
        """Run consolidation-style discovery over the current world graph."""
        concepts = self.world.discover_concepts()
        similarities = self.world.discover_structural_similarities(min_score=0.5)
        patterns = PatternDiscoveryEngine(
            self.world.get_relations()
        ).discover_relation_bigrams(min_count=self.prediction_min_count)
        low_quality_nodes = {
            obj.name: node_quality(obj.name)
            for obj in self.world.get_objects()
            if node_quality(obj.name) < self.min_node_quality
        }
        self.sleep_report = PipelineSleepReport(
            concepts=concepts,
            patterns=patterns,
            structural_similarities=similarities,
            relation_trust=dict(self.relation_trust),
            low_quality_nodes=low_quality_nodes,
        )
        return self.sleep_report

    def predict(self) -> list[PatternPrediction]:
        """Generate transitive and allowlisted mixed-pattern predictions."""
        predictor = PatternBasedPredictor(self.world.get_relations())
        transitive = predictor.predict_from_bigrams(
            min_count=self.prediction_min_count,
            min_confidence=0.0,
            hub_penalty=True,
            relation_trust=self.relation_trust,
            use_node_quality=self.use_node_quality,
            min_node_quality=self.min_node_quality,
            use_relation_drift=True,
        )
        mixed = predictor.predict_from_mixed_bigrams(
            min_count=self.prediction_min_count,
            min_confidence=0.0,
            hub_penalty=True,
            relation_trust=self.relation_trust,
            use_node_quality=self.use_node_quality,
            min_node_quality=self.min_node_quality,
        )
        return _dedupe_predictions(transitive + mixed)

    def audit(self, prediction: PatternPrediction, label: str) -> PipelineAuditRecord:
        """Record human feedback for a prediction."""
        score = label_score(label)
        if score is None:
            raise ValueError("audit label must not be empty")
        record = PipelineAuditRecord(
            prediction=prediction,
            label=label.strip().lower(),
            score=score,
        )
        self.audit_records.append(record)
        return record

    def learn_from_audit(self) -> TrustUpdateReport:
        """Update relation trust by averaging explicit human audit labels."""
        before = dict(self.relation_trust)
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for record in self.audit_records:
            relation = record.prediction.relation_type
            sums[relation] = sums.get(relation, 0.0) + record.score
            counts[relation] = counts.get(relation, 0) + 1

        learned = {
            relation: round(total / counts[relation], 6)
            for relation, total in sorted(sums.items())
        }
        self.relation_trust.update(learned)
        return TrustUpdateReport(
            before=before,
            after=dict(self.relation_trust),
            learned=learned,
            counts=counts,
        )

    def explain(self, prediction: PatternPrediction) -> str:
        """Return a compact explanation for a prediction."""
        evidence = ", ".join(prediction.evidence) if prediction.evidence else "none"
        return (
            f"{prediction.source} --{prediction.relation_type}--> {prediction.target}\n"
            f"confidence={prediction.confidence:.3f}\n"
            f"evidence={evidence}\n"
            f"reason={prediction.reason}"
        )


def accepted_predictions(
    predictions: list[PatternPrediction],
    threshold: float,
) -> list[PatternPrediction]:
    return [prediction for prediction in predictions if prediction.confidence >= threshold]


def prediction_key(prediction: PatternPrediction) -> tuple[str, str, str]:
    return (prediction.source, prediction.relation_type, prediction.target)


def compare_prediction_runs(
    before: list[PatternPrediction],
    after: list[PatternPrediction],
    threshold: float,
) -> tuple[list[PatternPrediction], list[PatternPrediction], list[tuple[PatternPrediction, float]]]:
    before_map = {prediction_key(prediction): prediction for prediction in before}
    after_map = {prediction_key(prediction): prediction for prediction in after}
    survived: list[PatternPrediction] = []
    suppressed: list[PatternPrediction] = []
    changed: list[tuple[PatternPrediction, float]] = []

    for key, before_prediction in before_map.items():
        after_prediction = after_map.get(key)
        after_confidence = after_prediction.confidence if after_prediction else 0.0
        if before_prediction.confidence >= threshold and after_confidence >= threshold:
            survived.append(after_prediction)
        if before_prediction.confidence >= threshold and after_confidence < threshold:
            suppressed.append(before_prediction)
        if after_prediction and abs(after_confidence - before_prediction.confidence) > 1e-9:
            changed.append((after_prediction, after_confidence - before_prediction.confidence))

    survived.sort(key=lambda prediction: (-prediction.confidence, prediction.relation_type))
    suppressed.sort(key=lambda prediction: (-prediction.confidence, prediction.relation_type))
    changed.sort(key=lambda item: (item[1], item[0].relation_type, item[0].source))
    return survived, suppressed, changed


def _dedupe_predictions(predictions: list[PatternPrediction]) -> list[PatternPrediction]:
    best: dict[tuple[str, str, str], PatternPrediction] = {}
    for prediction in predictions:
        key = prediction_key(prediction)
        existing = best.get(key)
        if existing is None or prediction.confidence > existing.confidence:
            best[key] = prediction
    return sorted(
        best.values(),
        key=lambda prediction: (
            -prediction.confidence,
            prediction.relation_type,
            prediction.source,
            prediction.target,
        ),
    )


def _evidence_list(evidence: str | list[str] | None) -> list[str]:
    if evidence is None:
        return []
    if isinstance(evidence, str):
        return [evidence]
    return list(evidence)


__all__ = [
    "DEFAULT_PIPELINE_RELATION_TRUST",
    "LABEL_SCORES",
    "MicroworldPipeline",
    "PipelineAuditRecord",
    "PipelineSleepReport",
    "TrustUpdateReport",
    "accepted_predictions",
    "compare_prediction_runs",
    "prediction_key",
]
