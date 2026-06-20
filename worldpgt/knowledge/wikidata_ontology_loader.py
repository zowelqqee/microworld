"""Read-only Wikidata P279 ontology layer builder.

The loader starts from existing explicit ``is_a`` objects that have no outgoing
``is_a`` edge in the local overlay, resolves those class labels through
Wikidata search, and walks only ``P279`` (subclass of) edges for a bounded
number of hops. The result is a separate list of overlay_relation items; callers
can keep it as a read-only layer or merge it with a proposal overlay for QA.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from worldpgt.multihop_qa.path_validator import validate_hop_safety
from worldpgt.multihop_qa.types import HopEdge
from worldpgt.self_ingestion.overlay_delta_validator import validate_delta

WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
P279 = "P279"

_VERBISH_OR_CURRENT = re.compile(
    r"\b(?:current|currently|planned|proposed|under development|operated by|"
    r"developed by|produced by|built by|launched by|designed by|after \d{4})\b",
    re.IGNORECASE,
)
_NOISY_CHARS = re.compile(r"[()[\]{}:;]")
_YEAR_RE = re.compile(r"\b(?:18|19|20)\d{2}\b")
_CLASS_HEAD_TERMS = frozenset({
    "agency",
    "businessman",
    "businesswoman",
    "city",
    "company",
    "corporation",
    "country",
    "engineer",
    "manufacturer",
    "magazine",
    "organization",
    "publication",
    "state",
    "vehicle",
})


def _norm(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip()).casefold()


def _is_a_object(item: dict) -> str:
    if item.get("overlay_type") == "overlay_relation" and item.get("predicate") == "is_a":
        return str(item.get("object") or "").strip()
    if item.get("overlay_type") == "overlay_definition" and item.get("predicate", "is_a") == "is_a":
        return str(item.get("definition") or "").strip()
    return ""


def _is_a_subject(item: dict) -> str:
    if item.get("overlay_type") == "overlay_relation" and item.get("predicate") == "is_a":
        return str(item.get("subject") or "").strip()
    if item.get("overlay_type") == "overlay_definition" and item.get("predicate", "is_a") == "is_a":
        return str(item.get("subject") or "").strip()
    return ""


def empty_is_a_object_labels(overlay_items: list[dict]) -> list[str]:
    """Return ``is_a`` objects that do not themselves have outgoing ``is_a``."""

    subjects = {_norm(_is_a_subject(item)) for item in overlay_items if _is_a_subject(item)}
    objects: dict[str, str] = {}
    for item in overlay_items:
        obj = _is_a_object(item)
        if obj:
            objects.setdefault(_norm(obj), obj)
    return sorted(
        [label for key, label in objects.items() if key and key not in subjects],
        key=str.casefold,
    )


def looks_like_stable_class_label(label: str) -> bool:
    """Conservative filter for labels worth sending to Wikidata search."""

    clean = label.strip()
    if not clean:
        return False
    words = clean.split()
    if len(words) > 6:
        return False
    if _NOISY_CHARS.search(clean) or _YEAR_RE.search(clean):
        return False
    if _VERBISH_OR_CURRENT.search(clean):
        return False
    if any(ch.isdigit() for ch in clean):
        return False
    if clean.endswith((",", "-", " of", " in", " to", " by", " for")):
        return False
    return True


def _class_label_score(label: str) -> tuple[int, int, str]:
    words = label.casefold().split()
    head_match = 0 if any(term in words for term in _CLASS_HEAD_TERMS) else 1
    return (head_match, len(words), label.casefold())


@dataclass(frozen=True)
class WikidataSearchHit:
    qid: str
    label: str
    description: str = ""


@dataclass(frozen=True)
class WikidataP279Edge:
    subject_qid: str
    subject_label: str
    object_qid: str
    object_label: str


class WikidataOntologyClient(Protocol):
    def search_class(self, label: str) -> WikidataSearchHit | None:
        ...

    def p279_edges(self, qid: str, *, max_edges: int = 4) -> list[WikidataP279Edge]:
        ...


class WikidataApiClient:
    """Tiny Wikidata API client using ``wbsearchentities`` and ``wbgetentities``."""

    def __init__(self, *, api_url: str = WIKIDATA_API_URL, sleep_seconds: float = 0.25) -> None:
        self.api_url = api_url
        self.sleep_seconds = sleep_seconds
        self._entity_cache: dict[str, dict] = {}
        self._search_cache: dict[str, WikidataSearchHit | None] = {}

    def _get(self, params: dict[str, str]) -> dict:
        query = urllib.parse.urlencode(params)
        req = urllib.request.Request(
            f"{self.api_url}?{query}",
            headers={"User-Agent": "mini-worldgrad-wikidata-ontology-loader/1.0"},
        )
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code != 429 or attempt == 3:
                    raise
                time.sleep(max(1.0, self.sleep_seconds) * (attempt + 1))
        else:
            raise RuntimeError("Wikidata request failed") from last_error
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return data

    def search_class(self, label: str) -> WikidataSearchHit | None:
        key = _norm(label)
        if key in self._search_cache:
            return self._search_cache[key]
        data = self._get({
            "action": "wbsearchentities",
            "search": label,
            "language": "en",
            "format": "json",
            "limit": "3",
            "type": "item",
        })
        for row in data.get("search", []):
            qid = str(row.get("id") or "")
            found_label = str(row.get("label") or "")
            if qid.startswith("Q") and found_label:
                hit = WikidataSearchHit(
                    qid=qid,
                    label=found_label,
                    description=str(row.get("description") or ""),
                )
                self._search_cache[key] = hit
                return hit
        self._search_cache[key] = None
        return None

    def _entities(self, qids: list[str]) -> dict[str, dict]:
        missing = [qid for qid in qids if qid not in self._entity_cache]
        if missing:
            data = self._get({
                "action": "wbgetentities",
                "ids": "|".join(missing),
                "props": "labels|claims",
                "languages": "en",
                "format": "json",
            })
            for entity_qid, entity in data.get("entities", {}).items():
                self._entity_cache[entity_qid] = entity
        return {qid: self._entity_cache.get(qid, {}) for qid in qids}

    def _entity(self, qid: str) -> dict:
        return self._entities([qid]).get(qid, {})

    def _labels(self, qids: list[str]) -> dict[str, str]:
        entities = self._entities(qids)
        return {
            qid: str(entity.get("labels", {}).get("en", {}).get("value") or qid)
            for qid, entity in entities.items()
        }

    def _label(self, qid: str) -> str:
        if qid in self._entity_cache:
            entity = self._entity_cache[qid]
            return str(entity.get("labels", {}).get("en", {}).get("value") or qid)
        return self._labels([qid]).get(qid, qid)

    def p279_edges(self, qid: str, *, max_edges: int = 4) -> list[WikidataP279Edge]:
        entity = self._entity(qid)
        subject_label = self._label(qid)
        object_ids: list[str] = []
        for claim in entity.get("claims", {}).get(P279, []):
            mainsnak = claim.get("mainsnak", {})
            value = mainsnak.get("datavalue", {}).get("value", {})
            object_id = value.get("id") if isinstance(value, dict) else None
            if not isinstance(object_id, str) or not object_id.startswith("Q"):
                continue
            object_ids.append(object_id)
            if len(object_ids) >= max_edges:
                break

        object_labels = self._labels(object_ids) if object_ids else {}
        out: list[WikidataP279Edge] = []
        for object_id in object_ids:
            out.append(WikidataP279Edge(
                subject_qid=qid,
                subject_label=subject_label,
                object_qid=object_id,
                object_label=object_labels.get(object_id, object_id),
            ))
        return out


@dataclass
class OntologyLoadReport:
    seed_empty_label_count: int = 0
    searched_label_count: int = 0
    resolved_seed_count: int = 0
    unresolved_seed_labels: list[str] = field(default_factory=list)
    rejected_seed_labels: list[str] = field(default_factory=list)
    raw_edge_count: int = 0
    accepted_edge_count: int = 0
    rejected_edge_count: int = 0
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    max_depth: int = 0

    def to_dict(self) -> dict:
        return {
            "seed_empty_label_count": self.seed_empty_label_count,
            "searched_label_count": self.searched_label_count,
            "resolved_seed_count": self.resolved_seed_count,
            "unresolved_seed_labels": list(self.unresolved_seed_labels),
            "rejected_seed_labels": list(self.rejected_seed_labels),
            "raw_edge_count": self.raw_edge_count,
            "accepted_edge_count": self.accepted_edge_count,
            "rejected_edge_count": self.rejected_edge_count,
            "rejected_reasons": dict(self.rejected_reasons),
            "max_depth": self.max_depth,
        }


def _edge_to_overlay(edge: WikidataP279Edge, *, depth: int, seed_label: str) -> dict:
    subject_label = seed_label if depth == 1 else edge.subject_label
    return {
        "overlay_type": "overlay_relation",
        "subject": subject_label,
        "predicate": "is_a",
        "object": edge.object_label,
        "source_page": "Wikidata",
        "evidence_text": (
            f"Wikidata P279 subclass of: {edge.subject_label} ({edge.subject_qid}) "
            f"-> {edge.object_label} ({edge.object_qid})"
        ),
        "trust": "wikidata_p279_ontology",
        "risk": "low",
        "stability": "stable",
        "wikidata_property": P279,
        "subject_qid": edge.subject_qid,
        "wikidata_subject_label": edge.subject_label,
        "object_qid": edge.object_qid,
        "ontology_depth": depth,
        "ontology_seed_label": seed_label,
    }


def validate_ontology_layer(items: list[dict], base_items: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Apply safety checks and the existing overlay delta validator."""

    prechecked: list[dict] = []
    rejected: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    for item in items:
        subject = str(item.get("subject") or "").strip()
        obj = str(item.get("object") or "").strip()
        if _norm(subject) == _norm(obj):
            rejected["self_loop"] = rejected.get("self_loop", 0) + 1
            continue
        hop = HopEdge(
            subject=subject,
            predicate="is_a",
            object=obj,
            overlay_type="overlay_relation",
            trust=str(item.get("trust") or ""),
            stability=str(item.get("stability") or ""),
            risk=str(item.get("risk") or ""),
            source_page=str(item.get("source_page") or ""),
        )
        valid, reason = validate_hop_safety(hop)
        if not valid:
            key = reason or "path_safety_rejected"
            rejected[key] = rejected.get(key, 0) + 1
            continue
        key = (_norm(subject), _norm(obj))
        if key in seen:
            rejected["duplicate_in_layer"] = rejected.get("duplicate_in_layer", 0) + 1
            continue
        seen.add(key)
        prechecked.append(item)

    validation = validate_delta(prechecked, base_items)
    for row in validation.rejected_items:
        reason = str(row.get("reason") or "overlay_delta_validator_rejected")
        rejected[reason] = rejected.get(reason, 0) + 1
    return validation.accepted_items, dict(sorted(rejected.items()))


def build_wikidata_p279_ontology_layer(
    overlay_items: list[dict],
    client: WikidataOntologyClient,
    *,
    max_depth: int = 3,
    max_seed_labels: int = 80,
    max_edges_per_node: int = 4,
) -> tuple[list[dict], OntologyLoadReport]:
    """Build a bounded Wikidata P279 ontology layer for empty class labels."""

    empty_labels = empty_is_a_object_labels(overlay_items)
    report = OntologyLoadReport(seed_empty_label_count=len(empty_labels), max_depth=max_depth)
    candidate_labels = [
        label for label in empty_labels
        if looks_like_stable_class_label(label)
    ]
    rejected = [
        label for label in empty_labels
        if not looks_like_stable_class_label(label)
    ]
    report.rejected_seed_labels.extend(rejected)
    seed_labels: list[str] = []
    for label in sorted(candidate_labels, key=_class_label_score):
        seed_labels.append(label)
        if len(seed_labels) >= max_seed_labels:
            break

    report.searched_label_count = len(seed_labels)
    queue: deque[tuple[str, str, int, str]] = deque()
    for label in seed_labels:
        hit = client.search_class(label)
        if hit is None:
            report.unresolved_seed_labels.append(label)
            continue
        report.resolved_seed_count += 1
        queue.append((hit.qid, hit.label, 0, label))

    raw_items: list[dict] = []
    visited: set[str] = set()
    while queue:
        qid, label, depth, seed_label = queue.popleft()
        if depth >= max_depth or qid in visited:
            continue
        visited.add(qid)
        for edge in client.p279_edges(qid, max_edges=max_edges_per_node):
            raw_items.append(_edge_to_overlay(edge, depth=depth + 1, seed_label=seed_label))
            queue.append((edge.object_qid, edge.object_label, depth + 1, seed_label))

    report.raw_edge_count = len(raw_items)
    accepted, rejected = validate_ontology_layer(raw_items, overlay_items)
    report.accepted_edge_count = len(accepted)
    report.rejected_edge_count = sum(rejected.values())
    report.rejected_reasons = rejected
    return accepted, report


def write_ontology_layer_artifacts(
    *,
    base_overlay_path: str | Path,
    layer_items: list[dict],
    report: OntologyLoadReport,
    out_dir: str | Path,
) -> dict:
    base_path = Path(base_overlay_path)
    base_items = json.loads(base_path.read_text(encoding="utf-8"))
    out = Path(out_dir)
    layer_path = out / "wikidata_p279_ontology_layer.json"
    merged_path = out / "overlay_with_wikidata_p279_ontology.json"
    report_path = out / "wikidata_p279_ontology_report.json"
    out.mkdir(parents=True, exist_ok=True)
    layer_path.write_text(json.dumps(layer_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    merged_path.write_text(json.dumps([*base_items, *layer_items], indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    report_data = {
        **report.to_dict(),
        "base_overlay_path": str(base_path),
        "base_overlay_items": len(base_items),
        "layer_path": str(layer_path),
        "merged_overlay_path": str(merged_path),
        "merged_overlay_items": len(base_items) + len(layer_items),
        "read_only_layer": True,
    }
    report_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report_data
