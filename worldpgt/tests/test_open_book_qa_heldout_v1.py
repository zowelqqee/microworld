from worldpgt.benchmarks.open_book_qa.heldout_v1 import build_heldout_cases, heldout_pool_diagnostics
from worldpgt.benchmarks.open_book_qa.dataset import relation_id
from worldpgt.knowledge_pump.heldout_density_frontier import (
    attach_wikipedia_resolution_layer,
    build_density_frontier,
)
from worldpgt.knowledge_pump.wikidata_relation_layer import extract_relation_rows


def _relation(subject: str, predicate: str, obj: str) -> dict:
    return {
        "overlay_type": "overlay_relation", "subject": subject, "predicate": predicate,
        "object": obj, "evidence_text": f"{subject} {predicate.replace('_', ' ')} {obj}.",
        "source_url": "https://example.test/source",
    }


def test_heldout_builder_rejects_any_group_touching_main_ids():
    relations = []
    for index in range(10):
        subject = f"Implicit {index}"
        relations.extend([_relation(subject, "uses", f"resource {index}"), _relation(subject, "supports", f"outcome {index}")])
    for index in range(10):
        subject = f"Explicit {index}"
        relations.extend([_relation(subject, "uses", f"resource e{index}"), _relation(subject, "enables", f"outcome e{index}")])
    for index in range(20):
        relations.append(_relation(f"Paraphrase {index}", "uses", f"resource p{index}"))
    blocked = relation_id(_relation("Paraphrase 0", "uses", "resource p0"))
    # Replace the blocked group with an extra clean one so the requested size is retained.
    relations.append(_relation("Paraphrase extra", "uses", "resource extra"))
    cases, summary = build_heldout_cases(relations, {blocked})
    assert summary["overlap_count"] == 0
    assert summary["cases_per_category"] == {
        "multi_evidence_explicit": 10, "multi_evidence_implicit": 10, "paraphrase": 20,
    }
    assert all(blocked not in case["relation_ids"] for case in cases)
    implicit = [case for case in cases if case["category"] == "multi_evidence_implicit"]
    assert all(len(case["expected_objects"]) == 2 for case in implicit)
    assert summary["relation_density_distribution"]["subjects_with_2_predicate_groups"] == 20


def test_heldout_diagnostics_measure_density_after_id_exclusion():
    rows = [
        _relation("Entity A", "uses", "a resource"),
        _relation("Entity A", "supports", "an outcome"),
        _relation("Entity B", "uses", "another resource"),
    ]
    blocked = relation_id(rows[1])
    _, _, summary = heldout_pool_diagnostics(rows, {blocked})
    assert summary["relation_density_distribution"] == {
        "predicate_groups_per_subject": {1: 2},
        "subjects_with_1_predicate_group": 2,
        "subjects_with_2_predicate_groups": 0,
        "subjects_with_3_or_more_predicate_groups": 0,
    }


def test_density_frontier_targets_only_single_group_subjects():
    rows = [
        _relation("Single subject", "uses", "a resource"),
        _relation("Dense subject", "uses", "another resource"),
        _relation("Dense subject", "supports", "an outcome"),
    ]
    frontier, summary = build_density_frontier(rows, set())
    assert [row["subject"] for row in frontier] == ["Single subject"]
    assert frontier[0]["existing_clean_predicates"] == ["uses"]
    assert frontier[0]["predicates_touched_by_main"] == []
    assert summary["target_subject_count"] == 1
    assert "question" not in frontier[0]


def test_resolution_layer_keeps_surface_when_canonical_title_is_available_or_missing():
    frontier = [
        {"subject": "Aging research", "surface_subject": "Aging research"},
        {"subject": "AI-driven", "surface_subject": "AI-driven"},
    ]
    manifest = [
        {"title": "Aging research", "normalized_title": "Gerontology", "source_url": "https://example.test/gerontology", "fetch_status": "success"},
        {"title": "AI-driven", "normalized_title": "AI-driven", "fetch_status": "missing"},
    ]
    rows = attach_wikipedia_resolution_layer(frontier, manifest)
    assert rows[0]["surface_subject"] == "Aging research"
    assert rows[0]["canonical_entity"] == "Gerontology"
    assert rows[0]["surface_retained"] is True
    assert rows[1]["surface_subject"] == "AI-driven"
    assert rows[1]["canonical_entity"] is None
    assert rows[1]["canonical_resolution_status"] == "unresolved_wikipedia_title"


def test_wikidata_relation_layer_keeps_surface_and_uses_explicit_claims_only():
    rows = extract_relation_rows(
        surface_subject="Aging research", canonical_entity="Gerontology", canonical_qid="Q1",
        claims={
            "P178": [{"mainsnak": {"datavalue": {"value": {"id": "Q2"}}}}],
            "P366": [{"rank": "deprecated", "mainsnak": {"datavalue": {"value": {"id": "Q3"}}}}],
        }, labels={"Q2": "Example developer", "Q3": "Example use"},
        blocked_predicates={"used_for"},
    )
    assert len(rows) == 1
    assert rows[0]["subject"] == "Aging research"
    assert rows[0]["canonical_entity"] == "Gerontology"
    assert rows[0]["predicate"] == "developed_by"
    assert rows[0]["object"] == "Example developer"
    assert rows[0]["wikidata_property"] == "P178"
