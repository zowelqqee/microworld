"""
Pattern-based link predictor.

Uses relation bigrams discovered by PatternDiscoveryEngine to infer missing
direct links.

Transitive same-relation prediction:

    A --r--> B --r--> C   →   predict  A --r--> C

Mixed manually allowlisted prediction:

    A --r1--> B --r2--> C  →  predict  A --r3--> C

Hub penalty
-----------
High-degree intermediate nodes (hubs like "air", "glass", "wood") produce many
spurious transitive predictions.  When hub_penalty=True the confidence of each
chain is adjusted by:

    hub_factor = sqrt(10 / max(10, intermediate_degree))

where intermediate_degree is the total degree (in + out) of the intermediate
node B.  Chains through a degree-10 node are unaffected; chains through a
degree-90 node are penalised by a factor of ~0.33.

If multiple intermediate nodes lead to the same (src, rel, tgt) conclusion the
penalty is computed from the *best* (lowest-degree) intermediate node seen,
so the prediction benefits from the most specific evidence available.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .patterns import PatternDiscoveryEngine
from .relation_trust import get_trust, DEFAULT_RELATION_TRUST, UNKNOWN_RELATION_TRUST
from .node_quality import node_quality
from .reasoning_relations import is_relation_enabled
from .relation_drift import DEFAULT_DRIFT_PENALTY_TABLE, classify_made_of_drift

if TYPE_CHECKING:
    from .relations import Relation

DEFAULT_MIXED_BIGRAM_RULES: dict[tuple[str, str], str] = {
    ("is_a", "capable_of"): "capable_of",
    ("is_a", "has_property"): "has_property",
    ("is_a", "used_for"): "used_for",
    ("is_a", "has_a"): "has_a",
    ("part_of", "made_of"): "made_of",
}


@dataclass
class PatternPrediction:
    source:        str
    relation_type: str
    target:        str
    confidence:    float
    reason:        str
    evidence:      list[str] = field(default_factory=list)  # intermediate nodes
    drift_type:    str | None = None
    drift_penalty: float = 1.0


@dataclass
class PatternEvaluationResult:
    hidden_count:     int
    prediction_count: int
    true_positives:   int
    false_positives:  int
    false_negatives:  int
    precision:        float
    recall:           float
    f1:               float

    def __repr__(self) -> str:
        return (
            f"PatternEvaluationResult("
            f"P={self.precision:.3f}, R={self.recall:.3f}, F1={self.f1:.3f} | "
            f"TP={self.true_positives}, FP={self.false_positives}, FN={self.false_negatives})"
        )


class PatternBasedPredictor:
    """
    Infers missing links from frequent relation bigrams.

    Does not touch the lifecycle PredictionEngine.
    """

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = list(relations)
        self._existing: frozenset[tuple[str, str, str]] = frozenset(
            (r.source, r.relation_type, r.target) for r in self._relations
        )
        self._outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for r in self._relations:
            self._outgoing[r.source].append((r.relation_type, r.target))

        # Node degree tables (built once, used for hub penalty / filtering)
        out_deg: dict[str, int] = defaultdict(int)
        in_deg:  dict[str, int] = defaultdict(int)
        for r in self._relations:
            out_deg[r.source] += 1
            in_deg[r.target]  += 1
        all_nodes = set(out_deg) | set(in_deg)
        self._total_degree: dict[str, int] = {
            n: out_deg[n] + in_deg[n] for n in all_nodes
        }

        # How many times each node appears as an intermediate in a 2-hop chain
        all_chain_count: dict[str, int] = defaultdict(int)
        chain_count: dict[str, int] = defaultdict(int)
        for r1 in self._relations:
            for r2_type, _ in self._outgoing[r1.target]:
                all_chain_count[r1.target] += 1
                if r2_type == r1.relation_type:
                    chain_count[r1.target] += 1
        self._all_chain_intermediate_count: dict[str, int] = dict(all_chain_count)
        self._chain_intermediate_count: dict[str, int] = dict(chain_count)

    # ------------------------------------------------------------------
    # Public degree accessors
    # ------------------------------------------------------------------

    def get_total_degree(self, node: str) -> int:
        """Total (in + out) degree of *node*."""
        return self._total_degree.get(node, 0)

    def get_chain_intermediate_count(self, node: str) -> int:
        """Number of same-relation 2-hop chains in which *node* is the middle."""
        return self._chain_intermediate_count.get(node, 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict_from_bigrams(
        self,
        min_count: int = 5,
        min_confidence: float = 0.5,
        max_intermediate_degree: int | None = None,
        max_intermediate_count: int | None = None,
        hub_penalty: bool = True,
        relation_trust: dict[str, float] | None = None,
        use_node_quality: bool = False,
        min_node_quality: float = 0.3,
        include_disabled_relations: bool = False,
        use_relation_drift: bool = False,
        drift_penalty_table: dict[str, float] | None = None,
    ) -> list[PatternPrediction]:
        """
        For each frequent transitive bigram (r, r) predict A --r--> C wherever:
        - A --r--> B --r--> C exists
        - A --r--> C does NOT already exist
        - pattern count >= min_count  AND  derived confidence >= min_confidence

        Parameters
        ----------
        max_intermediate_degree
            Skip chains where B has total_degree > this value.
        max_intermediate_count
            Skip chains where B appears as middle node more than this many times.
        hub_penalty
            If True, scale confidence by sqrt(10 / max(10, degree_of_B)).
        relation_trust
            Dict mapping relation_type -> trust prior in (0, 1].  When provided,
            final_confidence = base_confidence * hub_factor * trust * nq.
            Pass an empty dict {} to disable trust scaling while keeping the
            parameter explicit; pass None (default) to skip trust entirely.
        use_node_quality
            If True, apply node_quality() to source, intermediate, and target.
            Chains where any node scores below min_node_quality are skipped.
            Surviving chains have confidence multiplied by
            min(src_quality, via_quality, tgt_quality).
        min_node_quality
            Hard threshold: skip chains containing nodes with quality < this.

        Formula:
            base_confidence = min(0.95, 0.5 + 0.05 * log(count + 1))
            hub_factor      = sqrt(10 / max(10, degree_of_B))   if hub_penalty
                            = 1.0                                otherwise
            trust           = relation_trust[rel]  (or UNKNOWN_RELATION_TRUST)
            nq              = min(src_q, via_q, tgt_q)           if use_node_quality
                            = 1.0                                otherwise
            final           = base_confidence * hub_factor * trust * nq  (capped at 0.95)

        Deduplicates by (source, relation_type, target); the evidence list
        collects all distinct intermediate nodes (up to 5) when multiple
        paths reach the same conclusion.  Confidence is set by the
        *best-scoring* intermediate seen (highest hub_factor × nq_via).
        """
        discovery = PatternDiscoveryEngine(self._relations)
        bigrams = discovery.discover_relation_bigrams(min_count=min_count)

        # Keep only same-relation (transitive) patterns
        transitive: dict[str, int] = {}   # rel_type -> pattern count
        for p in bigrams:
            if (
                p.relations[0] == p.relations[1]
                and is_relation_enabled(p.relations[0], include_disabled_relations)
            ):
                transitive[p.relations[0]] = p.count

        # (src, rel, tgt) -> PatternPrediction (for dedup + evidence accumulation)
        preds: dict[tuple[str, str, str], PatternPrediction] = {}
        # best combined scale factor (hub × nq_via) seen per key; used to pick
        # the highest-quality intermediate when multiple paths exist.
        best_scale: dict[tuple[str, str, str], float] = {}
        # best intermediate metadata for the reason string
        best_via_info: dict[tuple[str, str, str], tuple[str, int]] = {}

        # node quality cache: computed once per node per call
        _nq: dict[str, float] = {}

        def _cached_nq(name: str) -> float:
            if name not in _nq:
                _nq[name] = node_quality(name)
            return _nq[name]

        drift_penalties = DEFAULT_DRIFT_PENALTY_TABLE if drift_penalty_table is None else drift_penalty_table

        for rel_type, count in transitive.items():
            base_conf = min(0.95, 0.5 + 0.05 * math.log(count + 1))
            # filter on base confidence before any scaling (scaling can only reduce)
            if base_conf < min_confidence:
                continue

            trust = get_trust(rel_type, relation_trust) if relation_trust is not None else 1.0

            for r1 in self._relations:
                if r1.relation_type != rel_type:
                    continue
                src        = r1.source
                intermediate = r1.target
                deg = self._total_degree.get(intermediate, 0)

                if max_intermediate_degree is not None and deg > max_intermediate_degree:
                    continue

                chain_cnt = self._chain_intermediate_count.get(intermediate, 0)
                if max_intermediate_count is not None and chain_cnt > max_intermediate_count:
                    continue

                # node quality checks for source + intermediate (target checked below)
                if use_node_quality:
                    src_q = _cached_nq(src)
                    via_q = _cached_nq(intermediate)
                    if src_q < min_node_quality or via_q < min_node_quality:
                        continue
                else:
                    src_q = 1.0
                    via_q = 1.0

                hub_factor = math.sqrt(10.0 / max(10.0, deg)) if hub_penalty else 1.0
                # combined per-intermediate scale (hub × via quality)
                chain_scale = hub_factor * via_q

                for r2_type, r2_tgt in self._outgoing[intermediate]:
                    if r2_type != rel_type:
                        continue
                    if src == r2_tgt:                # skip self-loop
                        continue
                    key = (src, rel_type, r2_tgt)
                    if key in self._existing:         # already in graph
                        continue

                    if use_node_quality:
                        tgt_q = _cached_nq(r2_tgt)
                        if tgt_q < min_node_quality:
                            continue
                        nq_factor = min(src_q, via_q, tgt_q)
                    else:
                        tgt_q    = 1.0
                        nq_factor = 1.0

                    drift_type, drift_penalty = _drift_info(
                        rel_type,
                        src,
                        intermediate,
                        r2_tgt,
                        use_relation_drift,
                        drift_penalties,
                    )
                    # total scale for this chain (hub × nq of weakest node)
                    total_scale = hub_factor * nq_factor * drift_penalty
                    conf = min(0.95, base_conf * total_scale * trust)

                    if key in preds:
                        # accumulate evidence
                        existing_pred = preds[key]
                        if intermediate not in existing_pred.evidence and len(existing_pred.evidence) < 5:
                            existing_pred.evidence.append(intermediate)
                        # upgrade confidence if this chain has a better combined scale
                        if total_scale > best_scale.get(key, 0.0):
                            best_scale[key]    = total_scale
                            best_via_info[key] = (intermediate, deg)
                            existing_pred.confidence = conf
                            existing_pred.reason = _make_reason(
                                rel_type, count, intermediate, deg,
                                hub_factor, hub_penalty,
                                trust, relation_trust is not None,
                                nq_factor, use_node_quality,
                                drift_type, drift_penalty,
                            )
                            existing_pred.drift_type = drift_type
                            existing_pred.drift_penalty = drift_penalty
                    else:
                        reason = _make_reason(
                            rel_type, count, intermediate, deg,
                            hub_factor, hub_penalty,
                            trust, relation_trust is not None,
                            nq_factor, use_node_quality,
                            drift_type, drift_penalty,
                        )
                        preds[key] = PatternPrediction(
                            source=src,
                            relation_type=rel_type,
                            target=r2_tgt,
                            confidence=conf,
                            reason=reason,
                            evidence=[intermediate],
                            drift_type=drift_type,
                            drift_penalty=drift_penalty,
                        )
                        best_scale[key]    = total_scale
                        best_via_info[key] = (intermediate, deg)

        # apply min_confidence filter AFTER all scaling
        result = [p for p in preds.values() if p.confidence >= min_confidence]
        return sorted(result, key=lambda p: (-p.confidence, p.source, p.target))

    def predict_from_mixed_bigrams(
        self,
        min_count: int = 5,
        min_confidence: float = 0.5,
        allowed_rules: dict[tuple[str, str], str] | None = None,
        max_intermediate_degree: int | None = None,
        max_intermediate_count: int | None = None,
        hub_penalty: bool = True,
        relation_trust: dict[str, float] | None = None,
        use_node_quality: bool = False,
        min_node_quality: float = 0.3,
        include_disabled_relations: bool = False,
    ) -> list[PatternPrediction]:
        """
        Predict allowlisted mixed-relation closures from frequent bigrams.

        For an allowed rule (r1, r2) -> r3, predicts A --r3--> C wherever
        A --r1--> B --r2--> C exists and A --r3--> C is not already present.
        The default allowlist is deliberately small and manually chosen.
        """
        rules = DEFAULT_MIXED_BIGRAM_RULES if allowed_rules is None else allowed_rules
        if not rules:
            return []

        discovery = PatternDiscoveryEngine(self._relations)
        bigrams = discovery.discover_relation_bigrams(min_count=min_count)
        mixed_counts: dict[tuple[str, str], int] = {
            p.relations: p.count
            for p in bigrams
            if (
                len(p.relations) == 2
                and p.relations in rules
                and is_relation_enabled(p.relations[0], include_disabled_relations)
                and is_relation_enabled(p.relations[1], include_disabled_relations)
                and is_relation_enabled(rules[p.relations], include_disabled_relations)
            )
        }

        preds: dict[tuple[str, str, str], PatternPrediction] = {}
        best_conf: dict[tuple[str, str, str], float] = {}
        _nq: dict[str, float] = {}

        def _cached_nq(name: str) -> float:
            if name not in _nq:
                _nq[name] = node_quality(name)
            return _nq[name]

        for (r1_type, r2_type), count in mixed_counts.items():
            output_relation = rules[(r1_type, r2_type)]
            base_conf = min(0.95, 0.5 + 0.05 * math.log(count + 1))
            if base_conf < min_confidence:
                continue

            trust = (
                get_trust(output_relation, relation_trust)
                if relation_trust is not None
                else 1.0
            )

            for r1 in self._relations:
                if r1.relation_type != r1_type:
                    continue

                src = r1.source
                intermediate = r1.target
                deg = self._total_degree.get(intermediate, 0)

                if max_intermediate_degree is not None and deg > max_intermediate_degree:
                    continue

                chain_cnt = self._all_chain_intermediate_count.get(intermediate, 0)
                if max_intermediate_count is not None and chain_cnt > max_intermediate_count:
                    continue

                if use_node_quality:
                    src_q = _cached_nq(src)
                    via_q = _cached_nq(intermediate)
                    if src_q < min_node_quality or via_q < min_node_quality:
                        continue
                else:
                    src_q = 1.0
                    via_q = 1.0

                hub_factor = math.sqrt(10.0 / max(10.0, deg)) if hub_penalty else 1.0

                for actual_r2_type, r2_tgt in self._outgoing[intermediate]:
                    if actual_r2_type != r2_type:
                        continue
                    if src == r2_tgt:
                        continue

                    key = (src, output_relation, r2_tgt)
                    if key in self._existing:
                        continue

                    if use_node_quality:
                        tgt_q = _cached_nq(r2_tgt)
                        if tgt_q < min_node_quality:
                            continue
                        nq_factor = min(src_q, via_q, tgt_q)
                    else:
                        nq_factor = 1.0

                    total_scale = hub_factor * nq_factor
                    conf = min(0.95, base_conf * total_scale * trust)

                    reason = _make_mixed_reason(
                        r1_type,
                        r2_type,
                        output_relation,
                        count,
                        intermediate,
                        deg,
                        hub_factor,
                        trust,
                        nq_factor,
                    )

                    if key in preds:
                        existing_pred = preds[key]
                        if intermediate not in existing_pred.evidence and len(existing_pred.evidence) < 5:
                            existing_pred.evidence.append(intermediate)
                        if conf > best_conf.get(key, 0.0):
                            best_conf[key] = conf
                            existing_pred.confidence = conf
                            existing_pred.reason = reason
                    else:
                        preds[key] = PatternPrediction(
                            source=src,
                            relation_type=output_relation,
                            target=r2_tgt,
                            confidence=conf,
                            reason=reason,
                            evidence=[intermediate],
                        )
                        best_conf[key] = conf

        result = [p for p in preds.values() if p.confidence >= min_confidence]
        return sorted(result, key=lambda p: (-p.confidence, p.source, p.target))

    def explain_prediction(self, prediction: PatternPrediction) -> str:
        """Return a human-readable explanation for one prediction."""
        via = " | ".join(prediction.evidence)
        return (
            f"{prediction.source} --{prediction.relation_type}--> {prediction.target}\n"
            f"  conf   : {prediction.confidence:.3f}\n"
            f"  reason : {prediction.reason}\n"
            f"  via    : {via}"
        )


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------

def _make_reason(
    rel_type: str,
    count: int,
    via: str,
    degree: int,
    hub_factor: float,
    hub_penalty_on: bool,
    trust: float = 1.0,
    trust_on: bool = False,
    nq: float = 1.0,
    nq_on: bool = False,
    drift_type: str | None = None,
    drift_penalty: float = 1.0,
) -> str:
    parts = [f"count={count}"]
    if hub_penalty_on:
        parts += [f"via={via}", f"degree={degree}", f"hub_penalty={hub_factor:.2f}"]
    if trust_on:
        parts.append(f"trust={trust:.3f}")
    if nq_on:
        parts.append(f"nq={nq:.3f}")
    if drift_type is not None:
        parts.append(f"drift={drift_type}")
        parts.append(f"drift_penalty={drift_penalty:.2f}")
    if len(parts) == 1:
        return f"transitive pattern: {rel_type} -> {rel_type} ({parts[0]})"
    return f"transitive pattern: {rel_type} -> {rel_type} ({', '.join(parts)})"


def _drift_info(
    rel_type: str,
    source: str,
    intermediate: str,
    target: str,
    use_relation_drift: bool,
    drift_penalty_table: dict[str, float],
) -> tuple[str | None, float]:
    if not use_relation_drift or rel_type != "made_of":
        return None, 1.0
    drift_type = classify_made_of_drift(source, intermediate, target)
    if drift_type is None:
        return None, drift_penalty_table.get("none", 1.0)
    return drift_type, drift_penalty_table.get(drift_type, 1.0)


def _make_mixed_reason(
    r1_type: str,
    r2_type: str,
    output_relation: str,
    count: int,
    via: str,
    degree: int,
    hub_factor: float,
    trust: float,
    nq: float,
) -> str:
    return (
        f"mixed pattern: {r1_type} -> {r2_type} => {output_relation} "
        f"(count={count}, via={via}, degree={degree}, "
        f"hub_penalty={hub_factor:.2f}, trust={trust:.3f}, nq={nq:.3f})"
    )


# ------------------------------------------------------------------
# Evaluation helper
# ------------------------------------------------------------------

def evaluate_pattern_prediction_recovery(
    world,
    relation_types: set[str] | None = None,
    max_hidden: int = 100,
    min_count: int = 5,
) -> PatternEvaluationResult:
    """
    Hide known transitive-closure edges, run PatternBasedPredictor, score.

    Ground truth: (A, r, C) triples that
      (a) already exist in the graph, AND
      (b) satisfy A --r--> B --r--> C for at least one B.

    After hiding those triples (leaving the 2-hop chains intact) the
    predictor should be able to recover them from the transitive bigram.

    Parameters
    ----------
    world          : World instance
    relation_types : only consider these relation types; None = all
    max_hidden     : cap on how many edges to hide (first N deterministically)
    min_count      : forwarded to predict_from_bigrams
    """
    from .world import World

    rels = world.get_relations()
    existing = {(r.source, r.relation_type, r.target) for r in rels}

    outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for r in rels:
        outgoing[r.source].append((r.relation_type, r.target))

    # Collect ground-truth transitive triples (deduplicated, deterministic order)
    seen_candidates: set[tuple[str, str, str]] = set()
    candidates: list[tuple[str, str, str]] = []
    for r1 in rels:
        rt = r1.relation_type
        if relation_types is not None and rt not in relation_types:
            continue
        for r2_type, r2_tgt in outgoing[r1.target]:
            if r2_type != rt:
                continue
            if r1.source == r2_tgt:
                continue
            triple = (r1.source, rt, r2_tgt)
            if triple in existing and triple not in seen_candidates:
                seen_candidates.add(triple)
                candidates.append(triple)

    hidden = candidates[:max_hidden]
    hidden_keys: set[tuple[str, str, str]] = set(hidden)

    # Build modified world without hidden edges
    visible = [r for r in rels if (r.source, r.relation_type, r.target) not in hidden_keys]
    w2 = World(normalizer=world._normalizer)
    for r in visible:
        w2._relations.append(r)
        w2._ensure_object(r.source)
        w2._ensure_object(r.target)

    # Predict and score
    predictor = PatternBasedPredictor(w2.get_relations())
    preds = predictor.predict_from_bigrams(min_count=min_count)
    pred_keys = {(p.source, p.relation_type, p.target) for p in preds}

    tp = len(hidden_keys & pred_keys)
    fp = len(pred_keys - hidden_keys)
    fn = len(hidden_keys - pred_keys)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    return PatternEvaluationResult(
        hidden_count=len(hidden),
        prediction_count=len(preds),
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        precision=prec,
        recall=rec,
        f1=f1,
    )
