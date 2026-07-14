"""Answer-behavior patterns: evidence-scoped multi-step answer planning.

Given a question and its resolved target entities, walk the *local* evidence
graph (relations that carry an evidence span and a source reference) and build
an inspectable :class:`AnswerPlan`: an ordered set of content blocks, each tied
to one concrete graph edge, with a full deterministic score breakdown, the
rejected candidates, and the reason the walk stopped.

Design constraints (mirrors the layer contract):

* Structural only.  Selection never keys on entity names, domains, or literal
  predicate strings — it uses token overlap, node connectivity, evidence
  quality fields (``stability`` / ``risk`` / source counts), novelty, and
  repetition, all of which are schema-level properties of any overlay edge.
* Never invents a claim.  A block exists only if a concrete edge with an
  evidence span and a source reference supports it; no edge -> no block.
* A single reliable link stays a single block — the caller keeps the original
  short answer in that case (see :func:`plan_is_expansion`).
* Conflicting alternatives (same subject/predicate, evidence-backed but
  mutually incompatible objects) are surfaced as an uncertainty block, never
  as one confident claim.
* Deterministic.  No ML.  No network.  Accepted/promoted memory and their
  rules are untouched: this module only *reads* whatever items it is given.
"""

from __future__ import annotations

import re
import time
from array import array
from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import os
import sqlite3
from collections import OrderedDict
from threading import RLock, local

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")

# Generic function/question words only — no domain, entity, or predicate terms.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "by", "at", "as", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "can", "could", "will", "would", "should", "may",
    "might", "have", "has", "had", "that", "this", "these", "those", "it",
    "its", "their", "there", "what", "which", "who", "whom", "whose", "when",
    "where", "why", "how", "about", "tell", "me", "know", "you", "your",
    "please", "not", "no", "than", "then", "them", "they", "from", "into",
    "any", "all", "some", "such", "also", "more", "most", "other", "others",
})

# A node name is *non-referential* when it only denotes inside the source
# document's own discourse rather than naming a standalone thing:
#   - a bare pronoun / determiner used as the whole name ("it", "we", "the");
#   - a phrase led by a first-person or demonstrative deictic ("our approach",
#     "this method", "these results").
# This is a general linguistic property (person/demonstrative deixis), not a
# domain, entity, or predicate rule. A leading definite/indefinite article
# ("the Python language", "a Weibull model") is NOT deixis and stays.
_DEICTIC_LEADING = frozenset({
    "our", "my", "we", "us", "i", "this", "these", "those", "that",
})
_BARE_NONREFERENTIAL = frozenset({
    "it", "its", "we", "us", "the", "they", "them", "this", "these", "those",
})


def _is_referential_node(name: str) -> bool:
    normalized = _norm(name)
    if not normalized:
        return False
    if normalized in _BARE_NONREFERENTIAL:
        return False
    return normalized.split(" ", 1)[0] not in _DEICTIC_LEADING


# Score weights — module-level so every plan reports the exact formula used.
SCORE_WEIGHTS: dict[str, float] = {
    "direct_relevance": 0.25,
    "evidence_quality": 0.20,
    "source_corroboration": 0.10,
    "graph_connectivity": 0.20,
    "information_gain": 0.25,
    "relation_diversity": 0.10,
    "redundancy_penalty": -0.25,
    "unsupported_leap_penalty": -0.50,
    "length_value_tradeoff": -1.0,
}

THRESHOLDS: dict[str, float] = {
    "min_first_step_total": 0.35,
    # Continuation steps rarely overlap the question wording directly (their
    # relevance is carried through connectivity), so their floor sits below
    # the first-step floor — but still high enough that weak, redundant, or
    # softly-attached leaps are cut.
    "min_next_step_total": 0.34,
    "min_information_gain": 0.25,
    # Cost per additional block: later steps must earn their place.
    "length_cost_per_block": 0.05,
}

_MAX_BLOCKS_DEFAULT = 4
_MAX_REJECTED_PER_ROUND = 3
_SOFT_ATTACH_MIN_SHARED = 2
# A soft attachment needs near-complete containment of one side's name in the
# other; sharing a couple of common words (e.g. "natural language") is not a
# graph link and must not license a hop.
_SOFT_ATTACH_MIN_RATIO = 0.75


def _norm(text: object) -> str:
    return " ".join(str(text or "").casefold().split())


def _tokens(text: object) -> frozenset[str]:
    raw = _TOKEN_RE.findall(_norm(text))
    # Exact lexical overlap only.  Morphological normalization is deliberately
    # out of scope for this structural layer: it would need language-aware
    # policy, and a false match is worse than slightly lower relevance.
    return frozenset(t for t in raw if len(t) > 2 and t not in _STOPWORDS)


# --------------------------------------------------------------------------- #
# Inspectable plan objects
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class EvidenceEdge:
    """One evidence-backed relation edge from the local graph."""

    evidence_id: str
    subject: str
    predicate: str
    object: str
    evidence_text: str
    sources: tuple[str, ...]
    stability: str
    risk: str
    trust: str
    corroboration: str  # "independent_sources" | "single_source"
    support_count: int
    # These keys are shared directly with graph dictionaries.  Keeping them
    # once on the edge avoids repeated normalization on warm queries and a
    # second retained copy of every unique node string in index construction.
    subject_norm: str = field(init=False)
    object_norm: str = field(init=False)
    predicate_norm: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "subject_norm", _norm(self.subject))
        object.__setattr__(self, "object_norm", _norm(self.object))
        object.__setattr__(self, "predicate_norm", _norm(self.predicate))

    def content_tokens(self) -> frozenset[str]:
        return _tokens(f"{self.subject} {self.predicate} {self.object}")

    def payload_tokens(self) -> frozenset[str]:
        """Tokens of the object only — the new fact content a block introduces.

        Novelty accounting uses this, not :meth:`content_tokens`: the subject is
        the (already-covered) target and the predicate is a structural relation
        label, so counting either would over- or under-state information gain
        (a literal predicate like ``is_a`` must never read as new content).
        """
        return _tokens(self.object)

    def all_tokens(self) -> frozenset[str]:
        return self.content_tokens() | _tokens(self.evidence_text)

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "object": self.object,
            "evidence_text": self.evidence_text,
            "sources": list(self.sources),
            "stability": self.stability,
            "risk": self.risk,
            "trust": self.trust,
            "corroboration": self.corroboration,
            "support_count": self.support_count,
        }


@dataclass(frozen=True)
class ScoreBreakdown:
    direct_relevance: float
    evidence_quality: float
    source_corroboration: float
    graph_connectivity: float
    information_gain: float
    relation_diversity: float
    redundancy_penalty: float
    unsupported_leap_penalty: float
    length_value_tradeoff: float
    total: float

    def to_dict(self) -> dict:
        return {
            "direct_relevance": round(self.direct_relevance, 4),
            "evidence_quality": round(self.evidence_quality, 4),
            "source_corroboration": round(self.source_corroboration, 4),
            "graph_connectivity": round(self.graph_connectivity, 4),
            "information_gain": round(self.information_gain, 4),
            "relation_diversity": round(self.relation_diversity, 4),
            "redundancy_penalty": round(self.redundancy_penalty, 4),
            "unsupported_leap_penalty": round(self.unsupported_leap_penalty, 4),
            "length_value_tradeoff": round(self.length_value_tradeoff, 4),
            "total": round(self.total, 4),
        }


@dataclass
class PlanningInstrumentation:
    """Optional, request-local counters and timings for planner benchmarks.

    The object is deliberately supplied by the caller instead of kept in a
    module-level collector.  That keeps serving deterministic and avoids a
    shared mutable metrics sink between requests.
    """

    edges_scanned: int = 0
    candidate_evaluations: int = 0
    candidates_discarded: int = 0
    selected_steps: int = 0
    candidate_fetch_ns: int = 0
    scoring_ns: int = 0
    diversity_ns: int = 0
    plan_assembly_ns: int = 0
    _visited_edge_ids: set[str] = field(default_factory=set, repr=False)
    _visited_nodes: set[str] = field(default_factory=set, repr=False)

    def record_edge_scan(self, edge: EvidenceEdge) -> None:
        self.edges_scanned += 1
        self._visited_edge_ids.add(edge.evidence_id)

    def record_node(self, node: str) -> None:
        if node:
            self._visited_nodes.add(node)

    def to_dict(self, *, planning_ns: int, rendering_ns: int = 0) -> dict:
        """A JSON-safe snapshot; all durations are milliseconds."""
        return {
            "edges_scanned": self.edges_scanned,
            "visited_edges": len(self._visited_edge_ids),
            "visited_nodes": len(self._visited_nodes),
            "candidate_evaluations": self.candidate_evaluations,
            "candidates_discarded": self.candidates_discarded,
            "selected_steps": self.selected_steps,
            "branch_pruning_rate": round(
                self.candidates_discarded / max(1, self.candidate_evaluations), 4
            ),
            "candidate_fetch_ms": round(self.candidate_fetch_ns / 1_000_000, 4),
            "scoring_ms": round(self.scoring_ns / 1_000_000, 4),
            "diversity_penalty_ms": round(self.diversity_ns / 1_000_000, 4),
            "plan_assembly_ms": round(self.plan_assembly_ns / 1_000_000, 4),
            "planning_ms": round(planning_ns / 1_000_000, 4),
            "rendering_ms": round(rendering_ns / 1_000_000, 4),
        }


@dataclass(frozen=True)
class GraphStep:
    """One selected movement through the local graph."""

    index: int
    edge: EvidenceEdge
    attach_node: str        # normalized node the step attaches to
    attach_kind: str        # "target" | "plan_node"
    attach_mode: str        # "exact" | "contained" | "soft"
    introduced_by_block: int  # block index that introduced the attach node (-1 = target)
    score: ScoreBreakdown

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "edge": self.edge.to_dict(),
            "attach_node": self.attach_node,
            "attach_kind": self.attach_kind,
            "attach_mode": self.attach_mode,
            "introduced_by_block": self.introduced_by_block,
            "score": self.score.to_dict(),
        }


@dataclass(frozen=True)
class ContentBlock:
    """A semantic step of the answer, traceable to one concrete graph edge.

    ``kind`` is derived from graph structure only:
      direct_claim              — first step, incident to a question target;
      explanatory_continuation  — attaches to a node introduced by an earlier block;
      sibling_elaboration       — another evidence-backed facet of a target;
      uncertainty_note          — evidence-backed but conflicting alternatives exist.
    """

    kind: str
    step: GraphStep
    cautions: tuple[str, ...] = ()
    alternatives: tuple[EvidenceEdge, ...] = ()

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "cautions": list(self.cautions),
            "alternatives": [edge.to_dict() for edge in self.alternatives],
            "step": self.step.to_dict(),
        }


@dataclass(frozen=True)
class RejectedCandidate:
    evidence_id: str
    reason: str
    total: float

    def to_dict(self) -> dict:
        return {
            "evidence_id": self.evidence_id,
            "reason": self.reason,
            "total": round(self.total, 4),
        }


@dataclass
class AnswerPlan:
    """The inspectable output of the answer-behavior layer."""

    question: str
    targets: tuple[str, ...]
    blocks: tuple[ContentBlock, ...]
    rejected: tuple[RejectedCandidate, ...] = ()
    ordering_rationale: tuple[str, ...] = ()
    stop_reason: str = ""

    def evidence_ids(self) -> list[str]:
        return [block.step.edge.evidence_id for block in self.blocks]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "targets": list(self.targets),
            "blocks": [block.to_dict() for block in self.blocks],
            "rejected_candidates": [r.to_dict() for r in self.rejected],
            "ordering_rationale": list(self.ordering_rationale),
            "stop_reason": self.stop_reason,
            "score_weights": dict(SCORE_WEIGHTS),
            "thresholds": dict(THRESHOLDS),
        }

    def trace(self) -> dict:
        """question -> local graph -> chosen plan -> evidence per block."""
        return {
            "question": self.question,
            "targets": list(self.targets),
            "steps": [
                {
                    "block": i,
                    "kind": block.kind,
                    "claim": (
                        f"{block.step.edge.subject} | {block.step.edge.predicate} "
                        f"| {block.step.edge.object}"
                    ),
                    "attach": f"{block.step.attach_kind}:{block.step.attach_node}"
                              f" ({block.step.attach_mode})",
                    "evidence_id": block.step.edge.evidence_id,
                    "evidence_text": block.step.edge.evidence_text,
                    "sources": list(block.step.edge.sources),
                    "cautions": list(block.cautions),
                    "score": block.step.score.to_dict(),
                }
                for i, block in enumerate(self.blocks)
            ],
            "stop_reason": self.stop_reason,
            "ordering_rationale": list(self.ordering_rationale),
        }


def plan_is_expansion(plan: AnswerPlan | None) -> bool:
    """Whether the plan justifies replacing a short single-fact answer.

    One reliable link never becomes a longer answer: expansion needs at least
    two independently evidence-backed blocks.
    """
    return plan is not None and len(plan.blocks) >= 2


# --------------------------------------------------------------------------- #
# Edge extraction
# --------------------------------------------------------------------------- #

def _edge_from_item(item: dict) -> EvidenceEdge | None:
    if not isinstance(item, dict):
        return None
    overlay_type = item.get("overlay_type")
    if overlay_type != "overlay_relation":
        return None
    obj = str(item.get("object") or "").strip()
    subject = str(item.get("subject") or "").strip()
    predicate = str(item.get("predicate") or "").strip()
    evidence_text = " ".join(
        str(item.get("evidence_text") or item.get("evidence_span") or "").split()
    )
    if not subject or not predicate or not obj or not evidence_text:
        return None
    if _norm(subject) == _norm(obj):
        return None
    # A claim anchored on (or pointing at) a non-referential deictic node
    # carries nothing retrievable about a stable entity — drop it outright.
    if not _is_referential_node(subject) or not _is_referential_node(obj):
        return None
    sources = sorted({
        str(url).strip()
        for url in [item.get("source_url"), *(item.get("supporting_sources") or [])]
        if str(url or "").strip()
    })
    if not sources:
        page = str(item.get("source_page") or "").strip()
        if not page:
            return None
        sources = [f"source_page:{page}"]
    quality = item.get("evidence_quality") or {}
    corroboration = str(quality.get("corroboration") or "")
    if corroboration not in {"independent_sources", "single_source"}:
        corroboration = "independent_sources" if len(sources) > 1 else "single_source"
    evidence_id = f"edge:{_norm(subject)}|{_norm(predicate)}|{_norm(obj)}"
    return EvidenceEdge(
        evidence_id=evidence_id,
        subject=subject,
        predicate=predicate,
        object=obj,
        evidence_text=evidence_text,
        sources=tuple(sources),
        stability=str(item.get("stability") or ""),
        risk=str(item.get("risk") or "").lower(),
        trust=str(item.get("trust") or ""),
        corroboration=corroboration,
        support_count=max(1, int(item.get("support_count") or 1)),
    )


def prepare_evidence_edges(overlay_items: list[dict]) -> tuple[EvidenceEdge, ...]:
    """Extract and deduplicate evidence edges once, for compatibility callers.

    New serving callers should prefer :func:`prepare_evidence_graph`: it adds
    universal structural indices so a query visits only edges connected to its
    resolved frontier.  This tuple API remains supported for small callers and
    preserves its existing return contract.
    """
    return tuple(_evidence_edges(overlay_items))


def _evidence_edges(overlay_items: list[dict]) -> list[EvidenceEdge]:
    by_id: dict[str, EvidenceEdge] = {}
    for item in overlay_items:
        edge = _edge_from_item(item)
        if edge is None:
            continue
        existing = by_id.get(edge.evidence_id)
        if existing is None:
            by_id[edge.evidence_id] = edge
            continue
        sources = tuple(sorted(set(existing.sources) | set(edge.sources)))
        by_id[edge.evidence_id] = EvidenceEdge(
            evidence_id=existing.evidence_id,
            subject=existing.subject,
            predicate=existing.predicate,
            object=existing.object,
            evidence_text=existing.evidence_text,
            sources=sources,
            stability=existing.stability or edge.stability,
            risk=existing.risk or edge.risk,
            trust=existing.trust or edge.trust,
            corroboration=(
                "independent_sources" if len(sources) > 1 else existing.corroboration
            ),
            support_count=existing.support_count + edge.support_count,
        )
    return [by_id[key] for key in sorted(by_id)]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #

_STABILITY_QUALITY = {"stable": 0.30, "semi_stable": 0.20}
_RISK_QUALITY = {"low": 0.30, "": 0.15, "medium": 0.15}


def _evidence_quality(edge: EvidenceEdge) -> float:
    quality = 0.40  # evidence span present (guaranteed by construction)
    quality += _STABILITY_QUALITY.get(edge.stability, 0.05)
    quality += _RISK_QUALITY.get(edge.risk, 0.0)
    return min(1.0, quality)


def _source_corroboration(edge: EvidenceEdge) -> float:
    if edge.corroboration == "independent_sources" or len(edge.sources) > 1:
        return 1.0
    return 0.5


@dataclass(frozen=True)
class _Attachment:
    node: str
    kind: str   # "target" | "plan_node"
    mode: str   # "exact" | "soft"
    introduced_by_block: int


def _find_attachment(
    edge: EvidenceEdge,
    plan_nodes: list[str],
    node_tokens: dict[str, frozenset[str]],
    node_origin: dict[str, int],
    target_nodes: set[str],
) -> _Attachment | None:
    for node in plan_nodes:
        if edge.subject_norm == node or edge.object_norm == node:
            kind = "target" if node in target_nodes else "plan_node"
            return _Attachment(node, kind, "exact", node_origin.get(node, -1))
    # Containment: the node's full name appears verbatim inside one side of
    # the edge ("calibrated gates" within "the calibrated gates of the outer
    # manifold").  Multi-word names only — a single word contained somewhere
    # is not a graph link.
    for node in plan_nodes:
        if len(node_tokens[node]) < 2:
            continue
        padded_node = f" {node} "
        for side_norm in (edge.subject_norm, edge.object_norm):
            if padded_node in f" {side_norm} ":
                kind = "target" if node in target_nodes else "plan_node"
                return _Attachment(node, kind, "contained", node_origin.get(node, -1))
    best: tuple[float, _Attachment] | None = None
    for node in plan_nodes:
        n_tokens = node_tokens[node]
        if not n_tokens:
            continue
        for side_tokens in (_tokens(edge.subject), _tokens(edge.object)):
            if not side_tokens:
                continue
            shared = len(n_tokens & side_tokens)
            if shared < _SOFT_ATTACH_MIN_SHARED:
                continue
            ratio = shared / min(len(n_tokens), len(side_tokens))
            if ratio < _SOFT_ATTACH_MIN_RATIO:
                continue
            kind = "target" if node in target_nodes else "plan_node"
            candidate = _Attachment(node, kind, "soft", node_origin.get(node, -1))
            if best is None or ratio > best[0]:
                best = (ratio, candidate)
    return best[1] if best is not None else None


_CONNECTIVITY = {
    ("target", "exact"): 1.0,
    ("plan_node", "exact"): 0.80,
    ("target", "contained"): 0.75,
    ("plan_node", "contained"): 0.60,
    ("target", "soft"): 0.55,
    ("plan_node", "soft"): 0.45,
}

# Leap penalty per attachment mode: exact node identity carries none, verbatim
# containment a small one, bag-of-tokens overlap the largest.
_LEAP_BY_MODE = {"exact": 0.0, "contained": 0.15, "soft": 0.30}


def _score_candidate(
    edge: EvidenceEdge,
    attachment: _Attachment,
    *,
    question_terms: frozenset[str],
    covered_tokens: set[str],
    used_attach_pairs: set[tuple[str, str]],
    step_index: int,
    instrumentation: PlanningInstrumentation | None = None,
) -> ScoreBreakdown:
    started_ns = time.perf_counter_ns()
    edge_terms = edge.all_tokens()
    overlap = len(question_terms & edge_terms)
    denom = max(1, min(len(question_terms), len(edge_terms))) if question_terms else 1
    direct_relevance = min(1.0, overlap / denom) if question_terms else 0.0

    quality = _evidence_quality(edge)
    corroboration = _source_corroboration(edge)
    connectivity = _CONNECTIVITY[(attachment.kind, attachment.mode)]

    payload = edge.payload_tokens()
    new_tokens = payload - covered_tokens
    information_gain = len(new_tokens) / max(1, len(payload))
    redundancy = 1.0 - information_gain

    # Reusing the same relation on the same node reads as flat enumeration
    # ("X enables A. X enables B. X enables C."). A fresh relation at the node
    # earns a small diversity bonus so a coherent, varied walk is preferred —
    # keyed on structural predicate identity, never on any literal label.
    diversity_started_ns = time.perf_counter_ns()
    relation_diversity = (
        0.0 if (attachment.node, edge.predicate_norm) in used_attach_pairs else 1.0
    )
    if instrumentation is not None:
        instrumentation.diversity_ns += time.perf_counter_ns() - diversity_started_ns

    leap = _LEAP_BY_MODE.get(attachment.mode, 0.30)
    length_cost = THRESHOLDS["length_cost_per_block"] * step_index

    total = (
        SCORE_WEIGHTS["direct_relevance"] * direct_relevance
        + SCORE_WEIGHTS["evidence_quality"] * quality
        + SCORE_WEIGHTS["source_corroboration"] * corroboration
        + SCORE_WEIGHTS["graph_connectivity"] * connectivity
        + SCORE_WEIGHTS["information_gain"] * information_gain
        + SCORE_WEIGHTS["relation_diversity"] * relation_diversity
        + SCORE_WEIGHTS["redundancy_penalty"] * redundancy
        + SCORE_WEIGHTS["unsupported_leap_penalty"] * leap
        + SCORE_WEIGHTS["length_value_tradeoff"] * length_cost
    )
    result = ScoreBreakdown(
        direct_relevance=direct_relevance,
        evidence_quality=quality,
        source_corroboration=corroboration,
        graph_connectivity=connectivity,
        information_gain=information_gain,
        relation_diversity=relation_diversity,
        redundancy_penalty=redundancy,
        unsupported_leap_penalty=leap,
        length_value_tradeoff=length_cost,
        total=total,
    )
    if instrumentation is not None:
        instrumentation.scoring_ns += time.perf_counter_ns() - started_ns
    return result


# --------------------------------------------------------------------------- #
# Conflict detection
# --------------------------------------------------------------------------- #

# A relation counts as effectively single-valued when, across the whole local
# graph, subjects using it carry close to one distinct object each — measured
# both on average and at the maximum observed fan-out (a single subject with
# three or more distinct objects proves the relation naturally holds many
# complementary facts, so disjoint objects are breadth, not conflict).
# Single-valuedness is a *learned* property: with fewer observed subjects than
# the minimum below there is not enough evidence to call the relation
# single-valued at all, and disjoint objects default to breadth.
_FUNCTIONAL_RELATION_MAX_RATIO = 1.34
_FUNCTIONAL_RELATION_MAX_FANOUT = 2
_FUNCTIONAL_RELATION_MIN_SUBJECTS = 3


def _relation_functionality(
    edges: list[EvidenceEdge],
) -> dict[str, tuple[float, int, int]]:
    objects_per_subject: dict[str, dict[str, set[str]]] = {}
    for edge in edges:
        objects_per_subject.setdefault(edge.predicate_norm, {}).setdefault(
            edge.subject_norm, set()
        ).add(edge.object_norm)
    return {
        predicate: (
            sum(len(objects) for objects in subjects.values()) / max(1, len(subjects)),
            max(len(objects) for objects in subjects.values()),
            len(subjects),
        )
        for predicate, subjects in objects_per_subject.items()
    }


def _conflicting_alternatives(
    edge: EvidenceEdge,
    edges_by_subject_predicate: dict[tuple[str, str], object],
    relation_functionality: dict[str, tuple[float, int, int]],
    edge_lookup: tuple[EvidenceEdge, ...] | None = None,
) -> tuple[EvidenceEdge, ...]:
    """Evidence-backed edges asserting an incompatible object for the same
    subject/predicate pair.

    Purely structural, three conditions: the relation was observed on enough
    distinct subjects to learn its shape at all, it behaves as single-valued
    across the graph (functionality ratio / max fan-out), AND the object
    descriptions share no content tokens.  No predicate semantics are
    consulted."""
    ratio, max_fanout, subject_count = relation_functionality.get(
        edge.predicate_norm, (1.0, 1, 0)
    )
    if (
        subject_count < _FUNCTIONAL_RELATION_MIN_SUBJECTS
        or ratio > _FUNCTIONAL_RELATION_MAX_RATIO
        or max_fanout > _FUNCTIONAL_RELATION_MAX_FANOUT
    ):
        return ()
    group = edges_by_subject_predicate.get((edge.subject_norm, edge.predicate_norm), ())
    members = _posting_ids(group) if isinstance(group, (int, array)) else group
    own_tokens = _tokens(edge.object)
    alternatives: list[EvidenceEdge] = []
    for member in members:
        other = edge_lookup[member] if edge_lookup is not None else member
        if (
            other.evidence_id != edge.evidence_id
            and not (own_tokens & _tokens(other.object))
        ):
            alternatives.append(other)
    return tuple(alternatives[:2])


@dataclass(frozen=True)
class PreparedEvidenceGraph:
    """Immutable structural indices over evidence edges.

    Indices are derived entirely from normalized node identity and tokens:
    there are no entity-, domain-, or predicate-specific branches.  Exact
    adjacency is the normal fast path.  The token index only retrieves a small
    *superset* for the existing multi-word containment and strict soft-attach
    checks; :func:`_find_attachment` remains the final semantic guard.
    """

    edges: tuple[EvidenceEdge, ...]
    # Postings store compact uint32 edge ids, never duplicate Edge references.
    # ``array('I')`` makes a 10m-edge index practical where Python int/list
    # postings would otherwise dominate memory.
    adjacency: dict[str, int | array]
    token_index: dict[str, int | array]
    by_subject_predicate: dict[tuple[str, str], int | array]
    relation_functionality: dict[str, tuple[float, int, int]]

    def __iter__(self):
        return iter(self.edges)

    def __len__(self) -> int:
        return len(self.edges)

    def candidate_edges(self, plan_nodes: list[str]) -> tuple[EvidenceEdge, ...]:
        """Return exact or viable fuzzy neighbours of the current frontier.

        A fuzzy candidate must share at least two content tokens with a plan
        node, which is a necessary condition for both containment and the
        configured soft-attach policy.  This removes the old full-overlay scan
        without weakening the final attachment thresholds.
        """
        candidate_ids: set[int] = set()
        for node in plan_nodes:
            candidate_ids.update(_posting_ids(self.adjacency.get(node)))

            tokens = _tokens(node)
            if len(tokens) < _SOFT_ATTACH_MIN_SHARED:
                continue
            shared_counts: dict[int, int] = {}
            for token in tokens:
                for edge_id in _posting_ids(self.token_index.get(token)):
                    shared_counts[edge_id] = shared_counts.get(edge_id, 0) + 1
            for edge_id, count in shared_counts.items():
                if count >= _SOFT_ATTACH_MIN_SHARED:
                    candidate_ids.add(edge_id)
        return tuple(
            self.edges[edge_id]
            for edge_id in sorted(candidate_ids, key=lambda item: self.edges[item].evidence_id)
        )


def _posting_ids(posting: int | array | None) -> tuple[int, ...] | array:
    """View a compact posting without allocating a singleton container."""
    if posting is None:
        return ()
    return (posting,) if isinstance(posting, int) else posting


def _compact_posting(values: list[int]) -> int | array:
    """A singleton needs one integer; fan-out uses a compact uint32 array."""
    return values[0] if len(values) == 1 else array("I", values)


def prepare_evidence_graph(overlay_items: list[dict]) -> PreparedEvidenceGraph:
    """Build the reusable local-neighbourhood index at overlay load time."""
    edges = tuple(_evidence_edges(overlay_items))
    adjacency_lists: dict[str, list[int]] = {}
    token_lists: dict[str, list[int]] = {}
    subject_predicate_lists: dict[tuple[str, str], list[int]] = {}
    for edge_id, edge in enumerate(edges):
        for node in {edge.subject_norm, edge.object_norm}:
            adjacency_lists.setdefault(node, []).append(edge_id)
        # A side with fewer than two content tokens can never satisfy either
        # containment or soft attachment: both require two shared tokens.
        # Exact identity is already covered by ``adjacency``, so skipping these
        # postings preserves behaviour while avoiding millions of useless
        # unique-token keys in title-heavy overlays.
        for side in (edge.subject, edge.object):
            side_tokens = _tokens(side)
            if len(side_tokens) < _SOFT_ATTACH_MIN_SHARED:
                continue
            for token in side_tokens:
                token_lists.setdefault(token, []).append(edge_id)
        subject_predicate_lists.setdefault(
            (edge.subject_norm, edge.predicate_norm), []
        ).append(edge_id)
    return PreparedEvidenceGraph(
        edges=edges,
        adjacency={key: _compact_posting(value) for key, value in adjacency_lists.items()},
        token_index={key: _compact_posting(value) for key, value in token_lists.items()},
        by_subject_predicate={
            key: _compact_posting(value) for key, value in subject_predicate_lists.items()
        },
        relation_functionality=_relation_functionality(list(edges)),
    )


_PERSISTENT_INDEX_SCHEMA_VERSION = "1"
_SQLITE_VARIABLE_CHUNK = 900


def _overlay_fingerprint(overlay_items: list[dict]) -> str:
    """Stable fingerprint for a derived index; never a memory promotion key."""
    digest = sha256()
    for item in overlay_items:
        if not isinstance(item, dict) or item.get("overlay_type") != "overlay_relation":
            continue
        digest.update(json.dumps(item, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _edge_from_sql_row(row: tuple) -> EvidenceEdge:
    return EvidenceEdge(
        evidence_id=row[1],
        subject=row[2],
        predicate=row[3],
        object=row[4],
        evidence_text=row[5],
        sources=tuple(json.loads(row[6])),
        stability=row[7],
        risk=row[8],
        trust=row[9],
        corroboration=row[10],
        support_count=int(row[11]),
    )


@dataclass
class PersistentEvidenceGraph:
    """Disk-backed evidence graph that materializes only local candidates.

    SQLite's B-trees keep the adjacency and token postings persistent on disk.
    The OS page cache may retain hot pages, but Python no longer owns a full
    duplicate edge/index graph in RAM after startup.
    """

    path: Path
    edge_count: int
    relation_functionality: dict[str, tuple[float, int, int]]
    frontier_cache_capacity: int = 512
    _connection_local: local = field(default_factory=local, init=False, repr=False)
    _frontier_cache: OrderedDict[str, tuple[int, ...]] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _cache_lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __len__(self) -> int:
        return self.edge_count

    def __iter__(self):
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT edge_id, evidence_id, subject, predicate, object, evidence_text, "
                "sources_json, stability, risk, trust, corroboration, support_count "
                "FROM edges ORDER BY evidence_id"
            )
            yield from (_edge_from_sql_row(row) for row in rows)

    def has_node(self, node: str) -> bool:
        return bool(self._frontier_ids(node))

    def close(self) -> None:
        """Close this thread's SQLite handle when a loaded overlay is replaced."""
        connection = getattr(self._connection_local, "connection", None)
        if connection is not None:
            connection.close()
            del self._connection_local.connection

    def _connection(self) -> sqlite3.Connection:
        connection = getattr(self._connection_local, "connection", None)
        if connection is None:
            connection = sqlite3.connect(self.path)
            connection.execute("PRAGMA query_only=ON")
            # Keep SQLite's own page cache bounded; the LRU below stores only
            # compact IDs, never evidence text or full edge objects.
            connection.execute("PRAGMA cache_size=-8192")
            self._connection_local.connection = connection
        return connection

    def _frontier_ids(self, node: str) -> tuple[int, ...]:
        with self._cache_lock:
            cached = self._frontier_cache.get(node)
            if cached is not None:
                self._frontier_cache.move_to_end(node)
                return cached

        connection = self._connection()
        edge_ids = {
            row[0] for row in connection.execute(
                "SELECT edge_id FROM adjacency WHERE node = ?", (node,)
            )
        }
        tokens = _tokens(node)
        if len(tokens) >= _SOFT_ATTACH_MIN_SHARED:
            placeholders = ",".join("?" for _ in tokens)
            edge_ids.update(
                row[0] for row in connection.execute(
                    "SELECT edge_id FROM token_posting WHERE token IN ("
                    f"{placeholders}) GROUP BY edge_id "
                    "HAVING COUNT(DISTINCT token) >= ?",
                    [*sorted(tokens), _SOFT_ATTACH_MIN_SHARED],
                )
            )
        result = tuple(sorted(edge_ids))
        with self._cache_lock:
            self._frontier_cache[node] = result
            self._frontier_cache.move_to_end(node)
            while len(self._frontier_cache) > self.frontier_cache_capacity:
                self._frontier_cache.popitem(last=False)
        return result

    def _edges_for_ids(
        self, connection: sqlite3.Connection, edge_ids: set[int]
    ) -> tuple[EvidenceEdge, ...]:
        if not edge_ids:
            return ()
        rows: list[tuple] = []
        ordered_ids = sorted(edge_ids)
        for start in range(0, len(ordered_ids), _SQLITE_VARIABLE_CHUNK):
            chunk = ordered_ids[start:start + _SQLITE_VARIABLE_CHUNK]
            placeholders = ",".join("?" for _ in chunk)
            rows.extend(connection.execute(
                "SELECT edge_id, evidence_id, subject, predicate, object, evidence_text, "
                "sources_json, stability, risk, trust, corroboration, support_count "
                f"FROM edges WHERE edge_id IN ({placeholders})",
                chunk,
            ))
        return tuple(sorted(
            (_edge_from_sql_row(row) for row in rows), key=lambda edge: edge.evidence_id
        ))

    def candidate_edges(self, plan_nodes: list[str]) -> tuple[EvidenceEdge, ...]:
        edge_ids: set[int] = set()
        for node in plan_nodes:
            edge_ids.update(self._frontier_ids(node))
        return self._edges_for_ids(self._connection(), edge_ids)

    def conflicting_alternatives(self, edge: EvidenceEdge) -> tuple[EvidenceEdge, ...]:
        ratio, max_fanout, subject_count = self.relation_functionality.get(
            edge.predicate_norm, (1.0, 1, 0)
        )
        if (
            subject_count < _FUNCTIONAL_RELATION_MIN_SUBJECTS
            or ratio > _FUNCTIONAL_RELATION_MAX_RATIO
            or max_fanout > _FUNCTIONAL_RELATION_MAX_FANOUT
        ):
            return ()
        rows = self._connection().execute(
            "SELECT edge_id, evidence_id, subject, predicate, object, evidence_text, "
            "sources_json, stability, risk, trust, corroboration, support_count "
            "FROM edges WHERE subject_norm = ? AND predicate_norm = ? "
            "AND evidence_id != ? ORDER BY evidence_id",
            (edge.subject_norm, edge.predicate_norm, edge.evidence_id),
        )
        own_tokens = _tokens(edge.object)
        alternatives = [
            other for other in (_edge_from_sql_row(row) for row in rows)
            if not (own_tokens & _tokens(other.object))
        ]
        return tuple(alternatives[:2])


def prepare_persistent_evidence_graph(
    overlay_items: list[dict], path: str | Path, *, source_fingerprint: str | None = None,
) -> PersistentEvidenceGraph:
    """Open a valid disk index or atomically rebuild it from an overlay.

    This is a serving cache, never accepted memory.  Its fingerprint is over
    the provided relation overlay, so a changed campaign invalidates it.  A
    caller that already owns a trusted file mtime/size token may pass it to
    avoid re-serializing a large in-memory overlay on every cache-hit startup.
    """
    index_path = Path(path)
    fingerprint = source_fingerprint or _overlay_fingerprint(overlay_items)
    if index_path.is_file():
        try:
            with sqlite3.connect(index_path) as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            if (
                metadata.get("schema_version") == _PERSISTENT_INDEX_SCHEMA_VERSION
                and metadata.get("fingerprint") == fingerprint
            ):
                return PersistentEvidenceGraph(
                    path=index_path,
                    edge_count=int(metadata["edge_count"]),
                    relation_functionality={
                        key: tuple(value)
                        for key, value in json.loads(metadata["relation_functionality"]).items()
                    },
                )
        except (sqlite3.DatabaseError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            pass

    edges = _evidence_edges(overlay_items)
    relation_functionality = _relation_functionality(edges)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    with sqlite3.connect(temporary) as connection:
        connection.executescript("""
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE edges (
                edge_id INTEGER PRIMARY KEY, evidence_id TEXT NOT NULL UNIQUE,
                subject TEXT NOT NULL, predicate TEXT NOT NULL, object TEXT NOT NULL,
                evidence_text TEXT NOT NULL, sources_json TEXT NOT NULL,
                stability TEXT NOT NULL, risk TEXT NOT NULL, trust TEXT NOT NULL,
                corroboration TEXT NOT NULL, support_count INTEGER NOT NULL,
                subject_norm TEXT NOT NULL, predicate_norm TEXT NOT NULL
            );
            CREATE TABLE adjacency (node TEXT NOT NULL, edge_id INTEGER NOT NULL);
            CREATE INDEX adjacency_node ON adjacency(node);
            CREATE TABLE token_posting (token TEXT NOT NULL, edge_id INTEGER NOT NULL);
            CREATE INDEX token_posting_token ON token_posting(token);
            CREATE INDEX edges_subject_predicate ON edges(subject_norm, predicate_norm);
        """)
        for edge_id, edge in enumerate(edges):
            connection.execute(
                "INSERT INTO edges VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    edge_id, edge.evidence_id, edge.subject, edge.predicate, edge.object,
                    edge.evidence_text, json.dumps(edge.sources), edge.stability, edge.risk,
                    edge.trust, edge.corroboration, edge.support_count, edge.subject_norm,
                    edge.predicate_norm,
                ),
            )
            connection.executemany(
                "INSERT INTO adjacency VALUES (?, ?)",
                [(node, edge_id) for node in {edge.subject_norm, edge.object_norm}],
            )
            for side in (edge.subject, edge.object):
                tokens = _tokens(side)
                if len(tokens) >= _SOFT_ATTACH_MIN_SHARED:
                    connection.executemany(
                        "INSERT INTO token_posting VALUES (?, ?)",
                        [(token, edge_id) for token in tokens],
                    )
        metadata = {
            "schema_version": _PERSISTENT_INDEX_SCHEMA_VERSION,
            "fingerprint": fingerprint,
            "edge_count": str(len(edges)),
            "relation_functionality": json.dumps(relation_functionality),
        }
        connection.executemany(
            "INSERT INTO metadata VALUES (?, ?)", metadata.items()
        )
    os.replace(temporary, index_path)
    return PersistentEvidenceGraph(
        path=index_path,
        edge_count=len(edges),
        relation_functionality=relation_functionality,
    )


# --------------------------------------------------------------------------- #
# Plan builder
# --------------------------------------------------------------------------- #

def build_answer_plan(
    question: str,
    overlay_items: list[dict],
    *,
    targets: list[str],
    max_blocks: int = _MAX_BLOCKS_DEFAULT,
    prepared_edges: (
        tuple[EvidenceEdge, ...] | PreparedEvidenceGraph | PersistentEvidenceGraph | None
    ) = None,
    instrumentation: PlanningInstrumentation | None = None,
) -> AnswerPlan | None:
    """Build an inspectable AnswerPlan or return None when no valid plan exists.

    ``targets`` are the question's resolved entities (from the semantic parser
    or surface index) — this function never guesses targets from text shape.
    ``prepared_edges`` may be a compatibility tuple from
    :func:`prepare_evidence_edges` or a :class:`PreparedEvidenceGraph` from
    :func:`prepare_evidence_graph`.  The graph form is the serving fast path:
    it avoids a whole-overlay scan on every planning step.
    """
    graph = prepared_edges if isinstance(prepared_edges, PreparedEvidenceGraph) else None
    persistent_graph = (
        prepared_edges if isinstance(prepared_edges, PersistentEvidenceGraph) else None
    )
    # ``PreparedEvidenceGraph.edges`` is already immutable.  Keep the tuple
    # by reference: copying a million-edge tuple into a list per query would
    # reintroduce whole-store latency even though candidate retrieval is local.
    edges = graph.edges if graph is not None else (
        () if persistent_graph is not None
        else list(prepared_edges) if prepared_edges is not None
        else _evidence_edges(overlay_items)
    )
    if (persistent_graph is not None and not persistent_graph.edge_count) or (
        persistent_graph is None and not edges
    ) or not targets:
        return None

    if graph is not None:
        adjacency = graph.adjacency
        by_subject_predicate = graph.by_subject_predicate
        relation_functionality = graph.relation_functionality
    elif persistent_graph is not None:
        adjacency = None
        by_subject_predicate = None
        relation_functionality = persistent_graph.relation_functionality
    else:
        adjacency: dict[str, list[EvidenceEdge]] = {}
        by_subject_predicate: dict[tuple[str, str], list[EvidenceEdge]] = {}
        for edge in edges:
            adjacency.setdefault(edge.subject_norm, []).append(edge)
            adjacency.setdefault(edge.object_norm, []).append(edge)
            by_subject_predicate.setdefault(
                (edge.subject_norm, edge.predicate_norm), []
            ).append(edge)
        relation_functionality = _relation_functionality(edges)

    target_nodes = {
        _norm(t)
        for t in targets
        if (
            persistent_graph.has_node(_norm(t))
            if persistent_graph is not None else _norm(t) in adjacency
        )
    }
    if not target_nodes:
        return None

    question_terms = _tokens(question)
    plan_nodes: list[str] = sorted(target_nodes)
    node_tokens: dict[str, frozenset[str]] = {n: _tokens(n) for n in plan_nodes}
    node_origin: dict[str, int] = {n: -1 for n in plan_nodes}
    if instrumentation is not None:
        for node in plan_nodes:
            instrumentation.record_node(node)
    covered_tokens: set[str] = set().union(*(node_tokens[n] for n in plan_nodes))

    blocks: list[ContentBlock] = []
    rejected: list[RejectedCandidate] = []
    rationale: list[str] = []
    used_attach_pairs: set[tuple[str, str]] = set()
    selected_ids: set[str] = set()
    stop_reason = ""

    while len(blocks) < max_blocks:
        step_index = len(blocks)
        candidates: list[tuple[EvidenceEdge, _Attachment]] = []
        fetch_started_ns = time.perf_counter_ns()
        candidate_edges = (
            graph.candidate_edges(plan_nodes) if graph is not None
            else persistent_graph.candidate_edges(plan_nodes)
            if persistent_graph is not None else edges
        )
        for edge in candidate_edges:
            if instrumentation is not None:
                instrumentation.record_edge_scan(edge)
            if edge.evidence_id in selected_ids:
                continue
            attachment = _find_attachment(
                edge, plan_nodes, node_tokens, node_origin, target_nodes
            )
            if attachment is None:
                continue
            if step_index == 0 and attachment.kind != "target":
                continue
            candidates.append((edge, attachment))
        if instrumentation is not None:
            instrumentation.candidate_fetch_ns += time.perf_counter_ns() - fetch_started_ns
            instrumentation.candidate_evaluations += len(candidates)

        scored: list[tuple[ScoreBreakdown, EvidenceEdge, _Attachment]] = []
        for edge, attachment in candidates:
            breakdown = _score_candidate(
                edge,
                attachment,
                question_terms=question_terms,
                covered_tokens=covered_tokens,
                used_attach_pairs=used_attach_pairs,
                step_index=step_index,
                instrumentation=instrumentation,
            )
            scored.append((breakdown, edge, attachment))

        if not scored:
            stop_reason = "no_connected_evidence_candidates"
            break

        scored.sort(key=lambda item: (-item[0].total, item[1].evidence_id))
        best_score, best_edge, best_attachment = scored[0]

        threshold = (
            THRESHOLDS["min_first_step_total"]
            if step_index == 0
            else THRESHOLDS["min_next_step_total"]
        )
        below_threshold = best_score.total < threshold
        below_gain = (
            step_index > 0
            and best_score.information_gain < THRESHOLDS["min_information_gain"]
        )
        no_direct_relevance = step_index == 0 and best_score.direct_relevance <= 0.0
        if below_threshold or below_gain or no_direct_relevance:
            if step_index == 0:
                return None
            if instrumentation is not None:
                instrumentation.candidates_discarded += len(scored)
            reason = (
                "below_information_gain_threshold" if below_gain
                else "below_step_total_threshold"
            )
            rejected.append(
                RejectedCandidate(best_edge.evidence_id, reason, best_score.total)
            )
            stop_reason = (
                f"next best step ({best_edge.evidence_id}) scored "
                f"{best_score.total:.3f}"
                + (
                    f" with information gain {best_score.information_gain:.3f}"
                    if below_gain else ""
                )
                + f", under the continue threshold — answer value would not grow"
            )
            break

        for runner_score, runner_edge, _att in scored[1:1 + _MAX_REJECTED_PER_ROUND]:
            rejected.append(
                RejectedCandidate(
                    runner_edge.evidence_id,
                    "outscored_by_selected_step",
                    runner_score.total,
                )
            )

        if instrumentation is not None:
            instrumentation.candidates_discarded += max(0, len(scored) - 1)

        assembly_started_ns = time.perf_counter_ns()
        step = GraphStep(
            index=step_index,
            edge=best_edge,
            attach_node=best_attachment.node,
            attach_kind=best_attachment.kind,
            attach_mode=best_attachment.mode,
            introduced_by_block=best_attachment.introduced_by_block,
            score=best_score,
        )
        alternatives = (
            persistent_graph.conflicting_alternatives(best_edge)
            if persistent_graph is not None else _conflicting_alternatives(
                best_edge,
                by_subject_predicate,
                relation_functionality,
                edge_lookup=graph.edges if graph is not None else None,
            )
        )
        cautions: list[str] = []
        if alternatives:
            cautions.append("conflicting_alternatives")
        if _source_corroboration(best_edge) < 1.0:
            cautions.append("single_source")
        kind = _block_kind(step, alternatives)
        blocks.append(
            ContentBlock(
                kind=kind,
                step=step,
                cautions=tuple(cautions),
                alternatives=alternatives,
            )
        )
        if instrumentation is not None:
            instrumentation.selected_steps += 1
        rationale.append(
            f"block {step_index + 1}: {kind} attaching to "
            f"{best_attachment.kind} '{best_attachment.node}' "
            f"({best_attachment.mode}); total={best_score.total:.3f}"
        )

        selected_ids.add(best_edge.evidence_id)
        # Alternatives are already surfaced inside the uncertainty block, so
        # they must not come back as separate (mirrored) claims later.
        for alternative in alternatives:
            selected_ids.add(alternative.evidence_id)
            covered_tokens |= alternative.payload_tokens()
        used_attach_pairs.add((best_attachment.node, best_edge.predicate_norm))
        # Coverage is span-aware: a selected edge surfaces its whole evidence
        # sentence, so a later edge whose content is already stated inside that
        # span carries no new information even when their triples differ (e.g.
        # a relation and a descriptor extracted from one rich source sentence).
        covered_tokens |= best_edge.payload_tokens()
        covered_tokens |= _tokens(best_edge.evidence_text)
        for node in (best_edge.subject_norm, best_edge.object_norm):
            if instrumentation is not None:
                instrumentation.record_node(node)
            if node not in node_tokens:
                plan_nodes.append(node)
                node_tokens[node] = _tokens(node)
                node_origin[node] = step_index
        if instrumentation is not None:
            instrumentation.plan_assembly_ns += time.perf_counter_ns() - assembly_started_ns
    else:
        stop_reason = "max_blocks_reached"

    if not blocks:
        return None
    return AnswerPlan(
        question=question,
        targets=tuple(sorted(target_nodes)),
        blocks=tuple(blocks),
        rejected=tuple(rejected),
        ordering_rationale=tuple(rationale),
        stop_reason=stop_reason or "no_connected_evidence_candidates",
    )


def _block_kind(step: GraphStep, alternatives: tuple[EvidenceEdge, ...]) -> str:
    if alternatives:
        return "uncertainty_note"
    if step.index == 0:
        return "direct_claim"
    if step.attach_kind == "target":
        return "sibling_elaboration"
    return "explanatory_continuation"
