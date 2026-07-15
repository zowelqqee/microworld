from __future__ import annotations

from pathlib import Path

from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.reasoning.graph_input import GraphInputLayer
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex


def _index(tmp_path: Path, items: list[dict]) -> EntitySurfaceIndex:
    empty = tmp_path / "empty.json"
    overlay = tmp_path / "overlay.json"
    empty.write_text("[]", encoding="utf-8")
    import json
    overlay.write_text(json.dumps(items), encoding="utf-8")
    return EntitySurfaceIndex(
        accepted_overlay_path=empty,
        promoted_overlay_path=overlay,
        snapshot_overlay_path=empty,
        graph_input=GraphInputLayer.from_overlay_items(items),
    )


def test_graph_input_layer_exposes_relation_nodes_to_the_semantic_parser(tmp_path: Path):
    items = [{
        "overlay_type": "overlay_relation",
        "subject": "Quartz relay",
        "predicate": "enables",
        "object": "signal routing",
    }]
    index = _index(tmp_path, items)

    parsed = parse_semantic_query("What is known about Quartz relay?", index)

    assert index.resolve("Quartz relay") == "Quartz relay"
    assert parsed.entity_a == "Quartz relay"


def test_graph_input_layer_rejects_deictic_relation_nodes(tmp_path: Path):
    items = [{
        "overlay_type": "overlay_relation",
        "subject": "Our approach",
        "predicate": "enables",
        "object": "signal routing",
    }]
    index = _index(tmp_path, items)

    assert index.resolve("Our approach") is None
    assert parse_semantic_query("What is known about Our approach?", index).entity_a is None
