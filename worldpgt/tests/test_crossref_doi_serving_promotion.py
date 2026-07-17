from worldpgt.api import server
from worldpgt.knowledge_pump.crossref_doi_seed import extract_doi_relation_rows
from worldpgt.knowledge_pump.crossref_doi_serving_promotion import build_crossref_doi_serving_overlay


def _item() -> dict:
    return {
        "DOI": "10.1000/serving",
        "title": ["A Serving DOI Work"],
        "publisher": "Example Press",
        "author": [{"given": "Ada", "family": "Lovelace"}],
    }


def test_crossref_precision_accepted_rows_promote_only_to_experimental_serving_graph():
    rows = extract_doi_relation_rows(_item(), topic_bucket="computing")
    overlay, summary = build_crossref_doi_serving_overlay(rows)

    assert summary["accepted_memory_modified"] is False
    assert summary["promoted_wiki_overlay_modified"] is False
    assert summary["serving_experimental_overlay_modified"] is True
    assert summary["serving_relation_count"] == 2
    assert {row["experimental_tier"] for row in overlay} == {"evidence_grounded_structured_relation_v1"}
    assert all(row["serving_status"] == "user_authorized_experimental" for row in overlay)


def test_server_merges_structured_serving_relations_with_same_evidence_semantics():
    rows = extract_doi_relation_rows(_item(), topic_bucket="computing")
    overlay, _summary = build_crossref_doi_serving_overlay(rows)
    duplicate = {**overlay[0], "source_url": "https://api.crossref.org/works/other", "supporting_sources": []}

    merged = server._merge_experimental_graph_items([overlay[0], duplicate, overlay[1]])
    assert len(merged) == 2
    assert merged[0]["supporting_source_count"] == 2
