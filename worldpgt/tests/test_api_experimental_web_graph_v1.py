from __future__ import annotations

import json
from pathlib import Path

from worldpgt.api import server


def test_experimental_graph_discovery_uses_completed_campaign_artifacts(tmp_path: Path, monkeypatch):
    filename = "open_web_campaign_evidence_grounded_graph_overlay.json"
    first = tmp_path / "campaign_zzz" / filename
    second = tmp_path / "campaign_aaa" / filename
    ignored = tmp_path / "campaign_draft" / "open_web_campaign_exploratory_graph_overlay.json"
    first.parent.mkdir()
    second.parent.mkdir()
    ignored.parent.mkdir()
    first.write_text("[]", encoding="utf-8")
    second.write_text("[]", encoding="utf-8")
    ignored.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(server, "_EXPERIMENTAL_WEB_CAMPAIGN_ROOT", tmp_path)

    paths = server._available_experimental_web_graph_paths()
    fingerprint = server._experimental_web_graph_fingerprint(paths)

    assert paths == (second, first)
    assert [Path(item[0]) for item in fingerprint] == [second, first]


def test_main_ui_overlay_composition_preserves_base_and_experimental_items(tmp_path: Path, monkeypatch):
    base_path = tmp_path / "base.json"
    graph_path = tmp_path / "graph.json"
    target_path = tmp_path / "composed.json"
    base_item = {"overlay_type": "overlay_definition", "subject": "Base", "definition": "base definition"}
    graph_item = {"overlay_type": "overlay_relation", "subject": "Graph", "predicate": "uses", "object": "edges"}
    base_path.write_text(json.dumps([base_item]), encoding="utf-8")
    graph_path.write_text(json.dumps([graph_item, base_item]), encoding="utf-8")
    monkeypatch.setattr(server, "_MAIN_UI_COMPOSED_OVERLAY_PATH", target_path)

    composed_path = server._compose_main_ui_overlay(base_path, graph_path)
    rows = json.loads(composed_path.read_text(encoding="utf-8"))

    assert rows == [base_item, graph_item]


def test_main_ui_overlay_composition_merges_multiple_campaign_graphs(tmp_path: Path, monkeypatch):
    base_path = tmp_path / "base.json"
    first_graph_path = tmp_path / "first_graph.json"
    second_graph_path = tmp_path / "second_graph.json"
    target_path = tmp_path / "composed.json"
    base_item = {"overlay_type": "overlay_definition", "subject": "Base", "definition": "base definition"}
    first_item = {"overlay_type": "overlay_relation", "subject": "First", "predicate": "uses", "object": "one"}
    second_item = {"overlay_type": "overlay_relation", "subject": "Second", "predicate": "enables", "object": "two"}
    base_path.write_text(json.dumps([base_item]), encoding="utf-8")
    first_graph_path.write_text(json.dumps([first_item]), encoding="utf-8")
    second_graph_path.write_text(json.dumps([first_item, second_item]), encoding="utf-8")
    monkeypatch.setattr(server, "_MAIN_UI_COMPOSED_OVERLAY_PATH", target_path)

    composed_path = server._compose_main_ui_overlay(base_path, (first_graph_path, second_graph_path))
    rows = json.loads(composed_path.read_text(encoding="utf-8"))

    assert rows == [base_item, first_item, second_item]


def test_startup_uses_explicit_graph_paths_without_discovering_other_campaigns(tmp_path: Path, monkeypatch):
    base_path = tmp_path / "base.json"
    selected_path = tmp_path / "selected.json"
    composed_path = tmp_path / "composed.json"
    base_path.write_text(json.dumps([{
        "overlay_type": "overlay_definition", "subject": "Base", "definition": "base definition",
    }]), encoding="utf-8")
    selected_path.write_text(json.dumps([{
        "overlay_type": "overlay_relation", "subject": "Heldout", "predicate": "used_for", "object": "testing",
    }]), encoding="utf-8")
    monkeypatch.setattr(server, "resolve_overlay", lambda _mode: (str(base_path), None))
    monkeypatch.setattr(server, "_MAIN_UI_COMPOSED_OVERLAY_PATH", composed_path)
    monkeypatch.setattr(server, "_available_experimental_web_graph_paths", lambda: (_ for _ in ()).throw(AssertionError("discovery must not run")))

    server._startup(
        "pump-dry-run",
        experimental_graph_paths=[selected_path],
        warm_phrase_graph_on_startup=False,
    )

    assert server._experimental_web_graph["paths"] == [str(selected_path)]
    assert server._experimental_web_graph["item_count"] == 1
    assert server._fact_count == 2


def test_experimental_graph_merge_combines_aliases_and_independent_relation_sources():
    entities_and_relations = [
        {
            "overlay_type": "overlay_entity", "label": "Large Language Model", "aliases": ["LLM"],
            "experimental_tier": "evidence_grounded_abstract_relation_v1",
        },
        {
            "overlay_type": "overlay_entity", "label": "large language model", "aliases": ["Language Model"],
            "experimental_tier": "evidence_grounded_abstract_relation_v1",
        },
        {
            "overlay_type": "overlay_relation", "subject": "Large Language Model", "predicate": "uses",
            "object": "attention mechanisms", "source_url": "https://example.test/a",
            "support_count": 1, "supporting_sources": ["https://example.test/a"],
            "evidence_text": "LLM uses attention mechanisms.",
            "experimental_tier": "evidence_grounded_abstract_relation_v1",
        },
        {
            "overlay_type": "overlay_relation", "subject": "large language model", "predicate": "uses",
            "object": "attention mechanisms", "source_url": "https://example.test/b",
            "support_count": 1, "supporting_sources": ["https://example.test/b"],
            "evidence_text": "Large Language Model uses attention mechanisms.",
            "experimental_tier": "evidence_grounded_abstract_relation_v1",
        },
    ]

    merged = server._merge_experimental_graph_items(entities_and_relations)

    assert len(merged) == 2
    entity, relation = merged
    assert entity["aliases"] == ["LLM", "Language Model"]
    assert relation["supporting_source_count"] == 2
    assert relation["evidence_quality"]["corroboration"] == "independent_sources"


def test_optional_reasoning_failure_keeps_base_qa_available(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("derived pattern failed")

    monkeypatch.setattr(server, "try_answer_reasoning", fail)

    assert server._run_optional_reasoning("What does AetherWall use?", [], [], None, None) is None
