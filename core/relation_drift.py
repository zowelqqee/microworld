"""
Relation drift analysis for transitive-style reasoning chains.

The predictor can find meaningful A-C connections even when repeating the same
relation label is too crude.  This module is an exploratory audit helper for
finding where relation composition changes semantic level.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .reasoning_relations import is_relation_enabled

if TYPE_CHECKING:
    from .relations import Relation

DIRECT_MATERIAL = "direct_material"
RAW_MATERIAL = "raw_material"
ATOMIC_COMPONENT = "atomic_component"
ABSTRACT_COMPONENT = "abstract_component"
UNKNOWN_MATERIAL = "unknown"

MATERIAL_CATEGORIES = {
    DIRECT_MATERIAL,
    RAW_MATERIAL,
    ATOMIC_COMPONENT,
    ABSTRACT_COMPONENT,
}

DEFAULT_DRIFT_PENALTY_TABLE: dict[str, float] = {
    "none": 1.0,
    DIRECT_MATERIAL: 1.0,
    RAW_MATERIAL: 0.85,
    ATOMIC_COMPONENT: 0.65,
    ABSTRACT_COMPONENT: 0.70,
}

_ATOMIC_COMPONENTS = {
    "aluminium", "aluminum", "aminoacids", "atom", "atoms", "calcium",
    "carbon", "chlorine", "copper", "electron", "electrons", "gold",
    "hydrogen", "iron", "lead", "molecules", "nitrogen", "oxygen",
    "proton", "protons", "silicon", "silver", "sodium",
    "subatomic_particles", "zinc",
}

_RAW_MATERIALS = {
    "bamboo", "coal", "cotton", "flax", "hemp", "hide", "leather",
    "log", "logs", "milk", "ore", "plant", "plants", "sand", "silica",
    "stone", "sugar", "tree", "trees", "wheat", "wood",
}

_DIRECT_MATERIALS = {
    "brick", "bricks", "cardboard", "cellulose", "ceramic", "clay",
    "cloth", "concrete", "fabric", "flesh", "glass", "haemoglobin",
    "hemoglobin", "metal", "music", "paper", "plastic", "protein", "sounds", "steel",
    "textile", "water",
}

_ABSTRACT_MARKERS = {
    "action", "belief", "concept", "countries", "country", "ideology",
    "culture", "ideal", "ideals", "knowledge", "memories", "members",
    "pain", "political_ideology", "principle", "suffering", "violence",
}


@dataclass
class RelationDepthPath:
    relation_type: str
    nodes: tuple[str, ...]

    @property
    def depth(self) -> int:
        return len(self.nodes) - 1

    @property
    def source(self) -> str:
        return self.nodes[0]

    @property
    def target(self) -> str:
        return self.nodes[-1]

    @property
    def evidence(self) -> list[str]:
        return list(self.nodes[1:-1])


@dataclass
class RelationDriftExample:
    relation_type: str
    source: str
    target: str
    path: tuple[str, ...]
    path_length: int
    categories: tuple[str, ...]
    drift: str
    audit_label: str | None = None


@dataclass
class RelationDriftReport:
    relation_type: str
    support: int
    drift_support: int
    reviewed: int = 0
    useful: int = 0
    wrong: int = 0
    audit_accuracy: float | None = None
    examples: list[RelationDriftExample] = field(default_factory=list)


class RelationDriftEngine:
    """Discover depth paths and likely semantic-level drift."""

    def __init__(self, relations: list[Relation]) -> None:
        self._relations = list(relations)
        self._outgoing: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for relation in self._relations:
            self._outgoing[relation.source].append(
                (relation.relation_type, relation.target)
            )

    def discover_relation_depths(
        self,
        max_depth: int = 3,
        include_disabled_relations: bool = False,
    ) -> dict[int, list[RelationDepthPath]]:
        """Return same-relation paths grouped by edge depth."""
        paths_by_depth: dict[int, list[RelationDepthPath]] = {
            depth: [] for depth in range(1, max_depth + 1)
        }
        for relation in self._relations:
            if not is_relation_enabled(relation.relation_type, include_disabled_relations):
                continue
            start_nodes = (relation.source, relation.target)
            paths_by_depth[1].append(
                RelationDepthPath(relation.relation_type, start_nodes)
            )
            self._walk_same_relation(
                relation.relation_type,
                start_nodes,
                max_depth,
                paths_by_depth,
                include_disabled_relations,
            )
        return paths_by_depth

    def detect_relation_drift(
        self,
        max_depth: int = 3,
        include_disabled_relations: bool = False,
        audit_rows: list[dict] | None = None,
    ) -> list[RelationDriftExample]:
        """Detect paths where same-relation composition changes semantic level."""
        audit_labels = _audit_label_index(audit_rows or [])
        paths = self.discover_relation_depths(
            max_depth=max_depth,
            include_disabled_relations=include_disabled_relations,
        )
        examples: list[RelationDriftExample] = []
        for depth in range(2, max_depth + 1):
            for path in paths.get(depth, []):
                drift, categories = _path_drift(path.relation_type, path.nodes)
                if drift is None:
                    continue
                examples.append(
                    RelationDriftExample(
                        relation_type=path.relation_type,
                        source=path.source,
                        target=path.target,
                        path=path.nodes,
                        path_length=path.depth,
                        categories=categories,
                        drift=drift,
                        audit_label=audit_labels.get(
                            (path.source, path.relation_type, path.target)
                        ),
                    )
                )
        return sorted(
            examples,
            key=lambda ex: (ex.relation_type, ex.path_length, ex.source, ex.target),
        )

    def build_report(
        self,
        max_depth: int = 3,
        include_disabled_relations: bool = False,
        audit_rows: list[dict] | None = None,
        max_examples_per_relation: int = 8,
    ) -> list[RelationDriftReport]:
        """Return per-relation drift support and audit accuracy."""
        paths = self.discover_relation_depths(
            max_depth=max_depth,
            include_disabled_relations=include_disabled_relations,
        )
        support: dict[str, int] = defaultdict(int)
        for depth in range(2, max_depth + 1):
            for path in paths.get(depth, []):
                support[path.relation_type] += 1

        drift_examples = self.detect_relation_drift(
            max_depth=max_depth,
            include_disabled_relations=include_disabled_relations,
            audit_rows=audit_rows,
        )
        drift_by_relation: dict[str, list[RelationDriftExample]] = defaultdict(list)
        for example in drift_examples:
            drift_by_relation[example.relation_type].append(example)

        audit = _audit_accuracy_by_relation(audit_rows or [])
        relation_types = set(support) | set(drift_by_relation) | set(audit)
        reports: list[RelationDriftReport] = []
        for relation_type in sorted(relation_types):
            reviewed, useful, wrong, accuracy = audit.get(relation_type, (0, 0, 0, None))
            reports.append(
                RelationDriftReport(
                    relation_type=relation_type,
                    support=support.get(relation_type, 0),
                    drift_support=len(drift_by_relation.get(relation_type, [])),
                    reviewed=reviewed,
                    useful=useful,
                    wrong=wrong,
                    audit_accuracy=accuracy,
                    examples=drift_by_relation.get(relation_type, [])[:max_examples_per_relation],
                )
            )
        return sorted(
            reports,
            key=lambda report: (-report.drift_support, report.relation_type),
        )

    def _walk_same_relation(
        self,
        relation_type: str,
        nodes: tuple[str, ...],
        max_depth: int,
        paths_by_depth: dict[int, list[RelationDepthPath]],
        include_disabled_relations: bool,
    ) -> None:
        depth = len(nodes) - 1
        if depth >= max_depth:
            return
        current = nodes[-1]
        for next_relation, next_node in self._outgoing[current]:
            if next_relation != relation_type:
                continue
            if not is_relation_enabled(next_relation, include_disabled_relations):
                continue
            if next_node in nodes:
                continue
            next_nodes = (*nodes, next_node)
            next_depth = len(next_nodes) - 1
            paths_by_depth[next_depth].append(
                RelationDepthPath(relation_type, next_nodes)
            )
            self._walk_same_relation(
                relation_type,
                next_nodes,
                max_depth,
                paths_by_depth,
                include_disabled_relations,
            )


def material_category(node: str) -> str:
    """Classify a node into a coarse material semantic level."""
    name = node.lower()
    tokens = set(name.split("_"))
    if name in _ATOMIC_COMPONENTS or tokens & _ATOMIC_COMPONENTS:
        return ATOMIC_COMPONENT
    if name in _ABSTRACT_MARKERS or tokens & _ABSTRACT_MARKERS:
        return ABSTRACT_COMPONENT
    if name in _RAW_MATERIALS or tokens & _RAW_MATERIALS:
        return RAW_MATERIAL
    if name in _DIRECT_MATERIALS or tokens & _DIRECT_MATERIALS:
        return DIRECT_MATERIAL
    return UNKNOWN_MATERIAL


def classify_made_of_drift(
    source: str,
    intermediate: str,
    target: str,
) -> str | None:
    """Return the target material level when a made_of chain drifts."""
    target_category = material_category(target)
    if target_category in {RAW_MATERIAL, ATOMIC_COMPONENT, ABSTRACT_COMPONENT}:
        return target_category
    _drift, categories = _path_drift("made_of", (source, intermediate, target))
    if _drift is None:
        return None
    return None if target_category == UNKNOWN_MATERIAL else target_category


def read_audit_rows(path: str) -> list[dict]:
    """Read an audit CSV with comma/semicolon delimiter detection."""
    with open(path, newline="", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = ","
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _path_drift(
    relation_type: str,
    nodes: tuple[str, ...],
) -> tuple[str | None, tuple[str, ...]]:
    if relation_type != "made_of":
        return None, tuple(UNKNOWN_MATERIAL for _ in nodes)
    categories = tuple(material_category(node) for node in nodes)
    known = [category for category in categories[1:] if category != UNKNOWN_MATERIAL]
    if len(known) < 2:
        return None, categories
    transitions = []
    for left, right in zip(categories[1:-1], categories[2:]):
        if UNKNOWN_MATERIAL in {left, right}:
            continue
        if left != right:
            transitions.append(f"{left}->{right}")
    if not transitions:
        return None, categories
    return ",".join(transitions), categories


def _audit_label_index(rows: list[dict]) -> dict[tuple[str, str, str], str]:
    labels: dict[tuple[str, str, str], str] = {}
    for row in rows:
        label = _normalize_label(row.get("manual_label", ""))
        if not label:
            continue
        relation_type = row.get("relation_type") or row.get("proposed_relation")
        if not relation_type:
            continue
        labels[(row.get("source", ""), relation_type, row.get("target", ""))] = label
    return labels


def _audit_accuracy_by_relation(
    rows: list[dict],
) -> dict[str, tuple[int, int, int, float | None]]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        label = _normalize_label(row.get("manual_label", ""))
        if not label:
            continue
        relation_type = row.get("relation_type") or row.get("proposed_relation")
        if not relation_type:
            continue
        counts[relation_type]["reviewed"] += 1
        if label in {"correct", "plausible"}:
            counts[relation_type]["useful"] += 1
        if label == "wrong":
            counts[relation_type]["wrong"] += 1

    result: dict[str, tuple[int, int, int, float | None]] = {}
    for relation_type, relation_counts in counts.items():
        reviewed = relation_counts["reviewed"]
        useful = relation_counts["useful"]
        wrong = relation_counts["wrong"]
        accuracy = useful / reviewed if reviewed else None
        result[relation_type] = (reviewed, useful, wrong, accuracy)
    return result


def _normalize_label(raw: str) -> str:
    label = raw.strip().lower()
    aliases = {
        "plusable": "plausible",
        "plausable": "plausible",
        "posible": "plausible",
        "true": "correct",
        "yes": "correct",
        "false": "wrong",
        "no": "wrong",
    }
    label = aliases.get(label, label)
    return label if label in {"correct", "plausible", "wrong", "unclear"} else ""
