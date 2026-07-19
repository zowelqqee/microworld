"""Isolated, explicit compositional-query experiment (AND and CHAIN only).

This module deliberately has no API/server imports.  Callers provide the
evidence slice they want evaluated, making it suitable for A/B benchmarks and
preventing accidental changes to the production multi-evidence route.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Literal

from worldpgt.multihop_qa.path_validator import validate_hop_safety
from worldpgt.multihop_qa.types import HopEdge

Operator = Literal["AND", "CHAIN"]


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


@dataclass(frozen=True)
class RelationRequest:
    predicate: str


@dataclass(frozen=True)
class AndQuery:
    subject: str
    relations: tuple[RelationRequest, ...] = ()
    # Empty relations is the deliberately bounded meaning of "two key facts":
    # return every predicate group available in the supplied evidence slice.
    operator: Operator = "AND"


@dataclass(frozen=True)
class ChainQuery:
    subject: str
    first_predicate: str
    second_predicate: str
    operator: Operator = "CHAIN"


@dataclass(frozen=True)
class EvidenceRef:
    edge: HopEdge
    evidence_id: str

    def to_dict(self) -> dict:
        return {"evidence_id": self.evidence_id, **self.edge.to_detail_dict()}


@dataclass
class CompositionalPlan:
    operator: Operator
    decision: Literal["answer", "audit"]
    components: list[list[EvidenceRef]] = field(default_factory=list)
    audit_reason: str | None = None

    @property
    def evidence(self) -> list[EvidenceRef]:
        return [ref for component in self.components for ref in component]

    def to_dict(self) -> dict:
        return {"operator": self.operator, "decision": self.decision,
                "audit_reason": self.audit_reason,
                "components": [[ref.to_dict() for ref in c] for c in self.components]}


def _edge(row: dict) -> EvidenceRef | None:
    subject, predicate, obj = (str(row.get(k) or "") for k in ("subject", "predicate", "object"))
    if not (subject and predicate and obj):
        return None
    edge = HopEdge(subject, predicate, obj, overlay_type=str(row.get("overlay_type") or "overlay_relation"),
                   trust=str(row.get("trust") or ""), stability=str(row.get("stability") or ""),
                   risk=str(row.get("risk") or ""), source_page=str(row.get("source_page") or row.get("source_url") or ""),
                   temporal_class=str(row.get("temporal_class") or ""), as_of=str(row.get("as_of") or ""))
    eid = str(row.get("evidence_id") or row.get("id") or f"edge:{_norm(subject)}|{predicate}|{_norm(obj)}")
    return EvidenceRef(edge, eid)


class CompositionalGrammar:
    def __init__(self, relations: list[dict]) -> None:
        self._edges = [edge for row in relations if (edge := _edge(row)) and edge.edge.overlay_type == "overlay_relation"]

    def execute(self, query: AndQuery | ChainQuery) -> CompositionalPlan:
        return self._and(query) if isinstance(query, AndQuery) else self._chain(query)

    def _safe(self, refs: list[EvidenceRef]) -> str | None:
        for ref in refs:
            valid, reason = validate_hop_safety(ref.edge)
            if not valid:
                return f"unsafe_component:{reason}"
        return None

    def _and(self, query: AndQuery) -> CompositionalPlan:
        subject_edges = [r for r in self._edges if _norm(r.edge.subject) == _norm(query.subject)]
        requested = [r.predicate for r in query.relations]
        predicates = requested or sorted({r.edge.predicate for r in subject_edges})
        if len(predicates) < 2:
            return CompositionalPlan("AND", "audit", audit_reason="and_requires_at_least_two_predicates")
        components = [[r for r in subject_edges if r.edge.predicate == predicate] for predicate in predicates]
        missing = [predicate for predicate, refs in zip(predicates, components) if not refs]
        if missing:
            return CompositionalPlan("AND", "audit", audit_reason="missing_predicate_support:" + ",".join(missing))
        if reason := self._safe([r for c in components for r in c]):
            return CompositionalPlan("AND", "audit", audit_reason=reason)
        return CompositionalPlan("AND", "answer", components)

    def _chain(self, query: ChainQuery) -> CompositionalPlan:
        first = [r for r in self._edges if _norm(r.edge.subject) == _norm(query.subject) and r.edge.predicate == query.first_predicate]
        if not first:
            return CompositionalPlan("CHAIN", "audit", audit_reason="missing_first_hop_support")
        paths = []
        for hop1 in first:
            for hop2 in self._edges:
                if _norm(hop2.edge.subject) == _norm(hop1.edge.object) and hop2.edge.predicate == query.second_predicate:
                    paths.append([hop1, hop2])
        if not paths:
            return CompositionalPlan("CHAIN", "audit", audit_reason="missing_second_hop_support")
        if reason := self._safe(paths[0]):
            return CompositionalPlan("CHAIN", "audit", audit_reason=reason)
        return CompositionalPlan("CHAIN", "answer", [paths[0]])


_EXPLICIT_AND = re.compile(r"^For (?P<subject>.+?), what are its (?P<predicates>.+?) relations\?$", re.I)
_CHAIN = re.compile(r"^Who (?P<second>[\w_]+) what (?P<subject>.+?) (?P<first>[\w_]+)\?*$", re.I)
_PREDICATE_CUES = {
    "developed_by": ("developed", "engineered", "created"), "used_for": ("used for", "application", "employed"),
    "product_of": ("manufactured", "manufacturer"), "founded_by": ("founded",),
    "headquartered_in": ("headquartered",), "published_by": ("published", "publisher"),
    "created_by": ("created",), "owned_by": ("owns", "owned"),
}


def parse_candidate(question: str, relations: list[dict]) -> AndQuery | ChainQuery | None:
    """Conservative parser for structural markers; unknown wording declines."""
    if match := _EXPLICIT_AND.match(question.strip()):
        available = sorted({str(r.get("predicate")) for r in relations})
        text = match.group("predicates").casefold()
        requested = [p for p in available if p.replace("_", " ").casefold() in text]
        if len(requested) >= 2:
            return AndQuery(match.group("subject"), tuple(RelationRequest(p) for p in requested))
    # Existing held-outs predate the canonical marker and use paraphrased
    # predicate cues. This is a per-predicate lexical adapter, never a pair
    # enum; operators remain generic for every matched combination.
    subject = next((str(r.get("subject")) for r in relations if str(r.get("subject")) and _norm(str(r.get("subject"))) in _norm(question)), None)
    if subject and " and " in question.casefold():
        text = question.casefold()
        requested = [p for p in sorted({str(r.get("predicate")) for r in relations}) if any(cue in text for cue in _PREDICATE_CUES.get(p, (p.replace("_", " "),)))]
        if len(requested) >= 2:
            return AndQuery(subject, tuple(RelationRequest(p) for p in requested))
    if match := _CHAIN.match(question.strip()):
        return ChainQuery(match.group("subject"), match.group("first"), match.group("second"))
    if question.strip().casefold().startswith("tell me two key relations about "):
        return AndQuery(question.strip()[len("Tell me two key relations about "):].rstrip("?. "))
    return None
