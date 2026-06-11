"""
Utilities for loading external relation datasets into a World.
"""
import csv
from .relations import Relation
from .world import World


def load_relations_csv(path: str) -> list[dict[str, str]]:
    """
    Load a CSV with columns: source, relation_type, target.
    Returns a list of row dicts; all values are strings.
    """
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                {
                    "source":        row["source"],
                    "relation_type": row["relation_type"],
                    "target":        row["target"],
                }
            )
    return rows


def build_world_from_relations(rows: list[dict[str, str]]) -> World:
    """
    Build a World directly from relation row dicts (bypasses the Russian parser).

    Each row must have keys: source, relation_type, target.
    Duplicate (source, relation_type, target) triples are silently dropped.
    """
    w = World()
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        src = row["source"]
        rel = row["relation_type"]
        tgt = row["target"]
        key = (src, rel, tgt)
        if key in seen:
            continue
        seen.add(key)
        w._ensure_object(src)
        w._ensure_object(tgt)
        w._relations.append(
            Relation(source=src, relation_type=rel, target=tgt, evidence=["csv"])
        )
    return w
