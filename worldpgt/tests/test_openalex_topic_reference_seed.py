from worldpgt.knowledge_pump.open_web_pump import _open_web_source_gate
from worldpgt.knowledge_pump.openalex_topic_reference_gate import validate_openalex_topic_reference_proposals
from worldpgt.knowledge_pump.openalex_topic_reference_seed import (
    extract_topic_reference_rows,
    fetch_diverse_openalex_records,
    select_topic_reference_entities,
)


def _work(identifier: str, title: str, *, references=(), topics=()) -> dict:
    return {
        "id": f"https://openalex.org/{identifier}",
        "display_name": title,
        "doi": f"https://doi.org/10.1000/{identifier.casefold()}",
        "topics": list(topics),
        "referenced_works": [f"https://openalex.org/{reference}" for reference in references],
    }


def test_openalex_extractor_uses_topic_and_citation_not_creator_or_publisher():
    work = _work(
        "W1", "A Diverse Work", references=("W2",),
        topics=({"display_name": "Graph Theory", "score": 0.9},),
    )
    reference = _work("W2", "A Referenced Work")
    rows = extract_topic_reference_rows([work], {"W2": reference})

    assert {row["predicate"] for row in rows} == {"has_topic", "references_work"}
    assert all(row["source_kind"] == "openalex_api" for row in rows)
    assert _open_web_source_gate(rows)["accepted"] == rows
    assert all(row["object"] in row["evidence_text"] for row in rows)


def test_openalex_selector_requires_the_structurally_diverse_pair():
    diverse = extract_topic_reference_rows([
        _work("W1", "A Diverse Work", references=("W2",), topics=({"display_name": "Graph Theory", "score": 0.9},))
    ], {"W2": _work("W2", "A Referenced Work")})
    incomplete = extract_topic_reference_rows([
        _work("W3", "A Topic Only Work", topics=({"display_name": "Metadata", "score": 0.5},))
    ], {})
    rows, manifest = select_topic_reference_entities([*diverse, *incomplete], max_entities=10)

    assert len(manifest) == 1
    assert manifest[0]["predicate_groups"] == ["has_topic", "references_work"]
    assert {row["predicate"] for row in rows} == {"has_topic", "references_work"}


def test_openalex_fetch_is_bounded_to_seed_and_one_reference_per_work():
    calls = []

    def fetch(url: str) -> dict:
        calls.append(url)
        if "doi:10.1000%2Fa" in url:
            return _work("W1", "A Work", references=("W2",), topics=({"display_name": "Topic", "score": 1},))
        if "doi:10.1000%2Fb" in url:
            return _work("W3", "B Work", references=("W2",), topics=({"display_name": "Topic", "score": 1},))
        return _work("W2", "Referenced Work")

    works, report = fetch_diverse_openalex_records(
        ["10.1000/a", "10.1000/b"], request_delay_sec=0, user_agent="test", get_json=fetch,
    )
    assert len(works) == 2
    assert report["seed_query_count"] == 2
    assert report["reference_lookup_count"] == 1
    assert report["total_queries"] == 3
    assert len(calls) == 3


def test_openalex_precision_gate_keeps_the_pair_as_proposal_only():
    rows = extract_topic_reference_rows([
        _work("W1", "A Diverse Work", references=("W2",), topics=({"display_name": "Graph Theory", "score": 0.9},))
    ], {"W2": _work("W2", "A Referenced Work")})

    report = validate_openalex_topic_reference_proposals(rows)

    assert report["proposal_only"] is True
    assert report["accepted_memory_modified"] is False
    assert report["serving_overlay_modified"] is False
    assert report["passed_source_gate"] == 2
    assert report["passed_precision_gate"] == 2
    assert report["entities_with_second_relation_group_after_gate"] == 1
    assert report["predicate_group_compositions"] == {"has_topic+references_work": 1}
    assert report["accepted_multi_predicate_entities"] == [{
        "canonical_openalex_id": "https://openalex.org/W1",
        "title": "A Diverse Work",
        "predicate_groups": ["has_topic", "references_work"],
    }]
