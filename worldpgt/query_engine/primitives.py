"""Composable query execution primitives.

Each primitive is a frozen dataclass describing one step of a QueryPlan.
No I/O, no side effects — pure descriptions of operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Target = Literal["object", "subject"]
CompareMode = Literal["intersection", "difference", "size_compare"]
AggregateOp = Literal["count", "list", "earliest", "latest"]


@dataclass(frozen=True)
class Find:
    """Find overlay_relation items matching a pattern.

    target="object": (subject, predicate, ?)  — known subject, find objects.
    target="subject": (?, predicate, known_value) — known object, find subjects.
    predicate=None matches any predicate.
    """
    subject: str | None
    predicate: str | None
    target: Target = "object"
    known_value: str | None = None
    result_key: str = "find"


@dataclass(frozen=True)
class Filter:
    """Filter a prior Find/Filter result.

    condition keys:
      "predicate"  — for each result's object entity, check a secondary relation
                     exists with this predicate in the overlay.
      "stability"  — keep only items with this stability value.
      "temporal_class" — keep only items with this temporal_class value.
    """
    source_key: str
    condition: dict
    result_key: str = "filter"


@dataclass(frozen=True)
class Count:
    """Count items in a prior step's result list."""
    source_key: str
    result_key: str = "count"


@dataclass(frozen=True)
class Compare:
    """Compare two result sets.

    mode="intersection"  — (predicate, object) pairs in both A and B.
    mode="difference"    — pairs in A but not B.
    mode="size_compare"  — which set is larger.
    """
    key_a: str
    key_b: str
    mode: CompareMode
    label_a: str = "A"
    label_b: str = "B"
    result_key: str = "compare"


@dataclass(frozen=True)
class Traverse:
    """2-hop relation chain through overlay relations.

    path: (predicate_hop1, predicate_hop2)
    Returns list of {"via": B, "endpoint": C} dicts.
    """
    entity: str
    path: tuple[str, str]
    result_key: str = "traverse"


@dataclass(frozen=True)
class Classify:
    """is_a membership check via ontology traversal."""
    entity: str
    target_class: str
    result_key: str = "classify"


@dataclass(frozen=True)
class Aggregate:
    """Aggregate a result list.

    operation="count"    — count items.
    operation="list"     — list object values.
    operation="earliest" — first item (by list order, no timestamp support).
    operation="latest"   — last item.
    """
    source_key: str
    operation: AggregateOp
    result_key: str = "aggregate"


Primitive = Find | Filter | Count | Compare | Traverse | Classify | Aggregate
