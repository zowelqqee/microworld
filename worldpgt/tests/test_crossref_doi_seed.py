from worldpgt.knowledge_pump.crossref_doi_seed import (
    extract_doi_relation_rows,
    fetch_crossref_doi_records,
    select_multi_predicate_doi_rows,
)
from worldpgt.knowledge_pump.crossref_doi_gate import validate_crossref_doi_proposals


def _item(doi: str, title: str = "A DOI work") -> dict:
    return {
        "DOI": doi,
        "title": [title],
        "publisher": "Example Press",
        "author": [{"given": "Ada", "family": "Lovelace"}],
    }


def test_crossref_doi_rows_are_directly_provenanced_and_proposal_only():
    rows = extract_doi_relation_rows(_item("10.1000/example"), topic_bucket="computing")

    assert {(row["predicate"], row["object"]) for row in rows} == {
        ("created_by", "Ada Lovelace"),
        ("published_by", "Example Press"),
    }
    assert all(row["source_kind"] == "crossref_doi" for row in rows)
    assert all(row["safe_for_general_runtime"] is False for row in rows)
    assert all(row["object"] in row["evidence_text"] for row in rows)
    assert all(row["source_url"].startswith("https://api.crossref.org/works/") for row in rows)


def test_crossref_doi_selector_requires_two_explicit_predicate_groups_and_deduplicates_title():
    dense = extract_doi_relation_rows(_item("10.1000/dense"), topic_bucket="computing")
    duplicate_title = extract_doi_relation_rows(_item("10.1000/duplicate"), topic_bucket="computing")
    sparse = extract_doi_relation_rows({"DOI": "10.1000/sparse", "title": ["Sparse"], "author": [{"family": "Solo"}]}, topic_bucket="computing")

    rows, manifest = select_multi_predicate_doi_rows([*dense, *duplicate_title, *sparse], max_entities=10)
    assert len(manifest) == 1
    assert manifest[0]["predicate_groups"] == ["created_by", "published_by"]
    assert {row["predicate"] for row in rows} == {"created_by", "published_by"}


def test_crossref_doi_fetch_is_bounded_deduplicated_and_preserves_errors():
    calls = []

    def fetch(url: str) -> dict:
        calls.append(url)
        if "second" in url:
            raise RuntimeError("temporary outage")
        return {"message": {"items": [_item("10.1000/shared"), _item("10.1000/unique")]}}

    records, report = fetch_crossref_doi_records(
        [("a", "first"), ("b", "second"), ("c", "third")],
        max_queries=2,
        records_per_query=3,
        request_delay_sec=0,
        user_agent="test-agent",
        get_json=fetch,
    )
    assert len(calls) == 2
    assert len(records) == 2
    assert report["requested_query_count"] == 2
    assert len(report["errors"]) == 1


def test_crossref_doi_gate_preserves_only_precision_accepted_multi_predicate_entities():
    rows = extract_doi_relation_rows(_item("10.1000/gated", "A DOI Work"), topic_bucket="computing")
    report = validate_crossref_doi_proposals(rows)

    assert report["proposal_only"] is True
    assert report["accepted_memory_modified"] is False
    assert report["passed_source_gate"] == 2
    assert report["passed_precision_gate"] == 2
    assert report["entities_with_second_relation_group_after_gate"] == 1
