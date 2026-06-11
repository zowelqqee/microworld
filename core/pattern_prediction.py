"""
Pattern-based link predictor.

Uses transitive relation bigrams discovered by PatternDiscoveryEngine to
infer missing direct links:

    A --r--> B --r--> C   →   predict  A --r--> C

Only same-relation (transitive) bigrams are used.  Mixed-relation patterns
are not yet supported.

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

if TYPE_CHECKING:
    from .relations import Relation


@dataclass
class PatternPrediction:
    source:        str
    relation_type: str
    target:        str
    confidence:    float
    reason:        str
    evidence:      list[str] = field(default_factory=list)  # intermediate nodes


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
    Infers missing transitive links from frequent same-relation bigrams.

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
        chain_count: dict[str, int] = defaultdict(int)
        for r1 in self._relations:
            for r2_type, _ in self._outgoing[r1.target]:
                if r2_type == r1.relation_type:
                    chain_count[r1.target] += 1
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
            final_confidence = base_confidence * hub_factor * trust.
            Pass an empty dict {} to disable trust scaling while keeping the
            parameter explicit; pass None (default) to skip trust entirely.

        Formula:
            base_confidence = min(0.95, 0.5 + 0.05 * log(count + 1))
            hub_factor      = sqrt(10 / max(10, degree_of_B))   if hub_penalty
                            = 1.0                                otherwise
            trust           = relation_trust[rel]  (or UNKNOWN_RELATION_TRUST)
            final           = base_confidence * hub_factor * trust   (capped at base)

        Deduplicates by (source, relation_type, target); the evidence list
        collects all distinct intermediate nodes (up to 5) when multiple
        paths reach the same conclusion.  When hub_penalty is True the
        confidence is set by the *best* (least-penalised) intermediate seen.
        """
        discovery = PatternDiscoveryEngine(self._relations)
        bigrams = discovery.discover_relation_bigrams(min_count=min_count)

        # Keep only same-relation (transitive) patterns
        transitive: dict[str, int] = {}   # rel_type -> pattern count
        for p in bigrams:
            if p.relations[0] == p.relations[1]:
                transitive[p.relations[0]] = p.count

        # (src, rel, tgt) -> PatternPrediction (for dedup + evidence accumulation)
        preds: dict[tuple[str, str, str], PatternPrediction] = {}
        # track best (lowest) penalty factor per prediction key when hub_penalty=True
        best_hub_factor: dict[tuple[str, str, str], float] = {}
        # track degree of best intermediate for reason string
        best_via_degree: dict[tuple[str, str, str], tuple[str, int]] = {}

        for rel_type, count in transitive.items():
            base_conf = min(0.95, 0.5 + 0.05 * math.log(count + 1))
            # We filter on base confidence before any scaling because scaling
            # can only reduce confidence further.
            if base_conf < min_confidence:
                continue

            trust = get_trust(rel_type, relation_trust) if relation_trust is not None else 1.0

            for r1 in self._relations:
                if r1.relation_type != rel_type:
                    continue
                intermediate = r1.target
                deg = self._total_degree.get(intermediate, 0)

                if max_intermediate_degree is not None and deg > max_intermediate_degree:
                    continue

                chain_cnt = self._chain_intermediate_count.get(intermediate, 0)
                if max_intermediate_count is not None and chain_cnt > max_intermediate_count:
                    continue

                if hub_penalty:
                    hub_factor = math.sqrt(10.0 / max(10.0, deg))
                else:
                    hub_factor = 1.0

                for r2_type, r2_tgt in self._outgoing[intermediate]:
                    if r2_type != rel_type:
                        continue
                    if r1.source == r2_tgt:          # skip self-loop
                        continue
                    key = (r1.source, rel_type, r2_tgt)
                    if key in self._existing:         # already in graph
                        continue

                    if key in preds:
                        # accumulate evidence
                        existing_pred = preds[key]
                        if intermediate not in existing_pred.evidence and len(existing_pred.evidence) < 5:
                            existing_pred.evidence.append(intermediate)
                        # update confidence upward if this path has a better hub factor
                        if hub_penalty and hub_factor > best_hub_factor.get(key, 0.0):
                            best_hub_factor[key] = hub_factor
                            best_via_degree[key] = (intermediate, deg)
                            new_conf = min(0.95, base_conf * hub_factor * trust)
                            existing_pred.confidence = new_conf
                            existing_pred.reason = _make_reason(
                                rel_type, count, intermediate, deg,
                                hub_factor, hub_penalty, trust, relation_trust is not None,
                            )
                    else:
                        conf = base_conf * hub_factor * trust
                        if not hub_penalty:
                            conf = base_conf * trust
                        conf = min(0.95, conf)
                        reason = _make_reason(
                            rel_type, count, intermediate, deg,
                            hub_factor, hub_penalty, trust, relation_trust is not None,
                        )
                        preds[key] = PatternPrediction(
                            source=r1.source,
                            relation_type=rel_type,
                            target=r2_tgt,
                            confidence=conf,
                            reason=reason,
                            evidence=[intermediate],
                        )
                        if hub_penalty:
                            best_hub_factor[key] = hub_factor
                            best_via_degree[key] = (intermediate, deg)

        # Apply min_confidence filter AFTER all scaling
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
) -> str:
    parts = [f"count={count}"]
    if hub_penalty_on:
        parts += [f"via={via}", f"degree={degree}", f"hub_penalty={hub_factor:.2f}"]
    if trust_on:
        parts.append(f"trust={trust:.3f}")
    if len(parts) == 1:
        return f"transitive pattern: {rel_type} -> {rel_type} ({parts[0]})"
    return f"transitive pattern: {rel_type} -> {rel_type} ({', '.join(parts)})"


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
