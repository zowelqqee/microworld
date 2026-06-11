"""
Exploratory relation proposal from observed 2-hop graph patterns.

This module does not replace PatternBasedPredictor.  It learns, from direct
edges already present in the graph, which relation labels commonly close a
2-hop chain:

    A --r1--> B --r2--> C
    A --r_out--> C

and then proposes candidate relation labels for novel A-C pairs.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .node_quality import node_quality
from .reasoning_relations import is_relation_enabled

if TYPE_CHECKING:
    from .relations import Relation

RelationRule = tuple[str, int, int, float]


@dataclass
class RelationProposal:
    source: str
    target: str
    proposed_relation: str
    confidence: float
    reason: str
    evidence: list[str] = field(default_factory=list)
    original_relation: str | None = None


class RelationProposalEngine:
    """Learn and propose relation labels for 2-hop A-B-C chains."""

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = list(relations)
        self._existing: frozenset[tuple[str, str, str]] = frozenset(
            (r.source, r.relation_type, r.target) for r in self._relations
        )

        self._outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        self._direct_relations: dict[tuple[str, str], list[str]] = defaultdict(list)
        out_deg: dict[str, int] = defaultdict(int)
        in_deg: dict[str, int] = defaultdict(int)

        for r in self._relations:
            self._outgoing[r.source].append((r.relation_type, r.target))
            self._direct_relations[(r.source, r.target)].append(r.relation_type)
            out_deg[r.source] += 1
            in_deg[r.target] += 1

        all_nodes = set(out_deg) | set(in_deg)
        self._total_degree: dict[str, int] = {
            node: out_deg[node] + in_deg[node] for node in all_nodes
        }
        self._relation_out_degree: dict[tuple[str, str], int] = defaultdict(int)
        for source, edges in self._outgoing.items():
            per_relation: dict[str, int] = defaultdict(int)
            for relation_type, _ in edges:
                per_relation[relation_type] += 1
            for relation_type, count in per_relation.items():
                self._relation_out_degree[(source, relation_type)] = count

    def discover_relation_rules(
        self,
        min_count: int = 3,
        min_rule_total: int = 10,
        rule_alpha: float = 1.0,
        rule_beta: float = 5.0,
        include_disabled_relations: bool = False,
    ) -> dict[tuple[str, str], list[RelationRule]]:
        """
        Learn direct-edge closure labels for each observed 2-hop relation pattern.

        Returns:
            {(r1, r2): [(r_out, count, total, confidence), ...]}

        confidence = (count + rule_alpha) / (total + rule_beta).  The total is
        observed direct A-C closure edges for this chain pattern.  If an A-C
        pair has multiple direct relation labels, each direct edge contributes
        one observation to the denominator.
        """
        counts: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        totals: dict[tuple[str, str], int] = defaultdict(int)

        for r1 in self._relations:
            if not is_relation_enabled(r1.relation_type, include_disabled_relations):
                continue
            for r2_type, c_node in self._outgoing[r1.target]:
                if not is_relation_enabled(r2_type, include_disabled_relations):
                    continue
                pattern = (r1.relation_type, r2_type)
                direct_relations = self._direct_relations.get((r1.source, c_node), [])
                for out_rel in direct_relations:
                    if not is_relation_enabled(out_rel, include_disabled_relations):
                        continue
                    counts[pattern][out_rel] += 1
                    totals[pattern] += 1

        rules: dict[tuple[str, str], list[RelationRule]] = {}
        for pattern, rel_counts in counts.items():
            total = totals[pattern]
            if total < min_rule_total:
                continue
            candidates = [
                (out_rel, count, total, _smoothed_confidence(count, total, rule_alpha, rule_beta))
                for out_rel, count in rel_counts.items()
                if count >= min_count
            ]
            if candidates:
                rules[pattern] = sorted(
                    candidates,
                    key=lambda item: (-item[3], -item[1], item[0]),
                )
        return rules

    def propose_relations(
        self,
        min_count: int = 3,
        min_confidence: float = 0.4,
        min_rule_total: int = 10,
        rule_alpha: float = 1.0,
        rule_beta: float = 5.0,
        max_intermediate_degree: int | None = None,
        max_intermediate_relation_fanout: int | None = None,
        relation_trust: dict[str, float] | None = None,
        use_node_quality: bool = False,
        min_node_quality: float = 0.3,
        include_disabled_relations: bool = False,
    ) -> list[RelationProposal]:
        """Propose learned output relation labels for novel 2-hop closures."""
        rules = self.discover_relation_rules(
            min_count=min_count,
            min_rule_total=min_rule_total,
            rule_alpha=rule_alpha,
            rule_beta=rule_beta,
            include_disabled_relations=include_disabled_relations,
        )
        proposals: dict[tuple[str, str, str], RelationProposal] = {}
        best_conf: dict[tuple[str, str, str], float] = {}
        _nq: dict[str, float] = {}

        def _cached_nq(name: str) -> float:
            if name not in _nq:
                _nq[name] = node_quality(name)
            return _nq[name]

        for r1 in self._relations:
            if not is_relation_enabled(r1.relation_type, include_disabled_relations):
                continue
            src = r1.source
            intermediate = r1.target
            degree = self._total_degree.get(intermediate, 0)

            if max_intermediate_degree is not None and degree > max_intermediate_degree:
                continue

            if use_node_quality:
                src_q = _cached_nq(src)
                via_q = _cached_nq(intermediate)
                if src_q < min_node_quality or via_q < min_node_quality:
                    continue
            else:
                src_q = 1.0
                via_q = 1.0

            hub_factor = math.sqrt(10.0 / max(10.0, degree))

            for r2_type, target in self._outgoing[intermediate]:
                if not is_relation_enabled(r2_type, include_disabled_relations):
                    continue
                pattern = (r1.relation_type, r2_type)
                candidates = rules.get(pattern)
                if not candidates:
                    continue
                if src == target:
                    continue
                fanout = self._relation_out_degree.get((intermediate, r2_type), 0)
                if (
                    max_intermediate_relation_fanout is not None
                    and fanout > max_intermediate_relation_fanout
                ):
                    continue

                if use_node_quality:
                    tgt_q = _cached_nq(target)
                    if tgt_q < min_node_quality:
                        continue
                    nq_factor = min(src_q, via_q, tgt_q)
                else:
                    nq_factor = 1.0

                for out_rel, count, total, rule_conf in candidates:
                    if not is_relation_enabled(out_rel, include_disabled_relations):
                        continue
                    if (src, out_rel, target) in self._existing:
                        continue

                    trust = _relation_trust(out_rel, relation_trust)
                    conf = rule_conf * hub_factor * trust * nq_factor
                    if conf < min_confidence:
                        continue

                    key = (src, out_rel, target)
                    reason = (
                        f"learned relation rule: {pattern[0]} -> {pattern[1]} => {out_rel} "
                        f"(support={count}/{total}, smoothed_conf={rule_conf:.3f}, fanout={fanout})"
                    )

                    if key in proposals:
                        existing = proposals[key]
                        if intermediate not in existing.evidence and len(existing.evidence) < 5:
                            existing.evidence.append(intermediate)
                        if conf > best_conf.get(key, 0.0):
                            best_conf[key] = conf
                            existing.confidence = conf
                            existing.reason = reason
                            existing.original_relation = r2_type
                    else:
                        proposals[key] = RelationProposal(
                            source=src,
                            target=target,
                            proposed_relation=out_rel,
                            confidence=conf,
                            reason=reason,
                            evidence=[intermediate],
                            original_relation=r2_type,
                        )
                        best_conf[key] = conf

        return sorted(
            proposals.values(),
            key=lambda p: (-p.confidence, p.proposed_relation, p.source, p.target),
        )


def _relation_trust(
    relation_type: str,
    relation_trust: dict[str, float] | None,
) -> float:
    if relation_trust is None:
        return 1.0
    raw = relation_trust.get(relation_type, 1.0)
    return max(0.0, min(1.0, raw))


def _smoothed_confidence(
    count: int,
    total: int,
    alpha: float,
    beta: float,
) -> float:
    denominator = total + beta
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, (count + alpha) / denominator))
