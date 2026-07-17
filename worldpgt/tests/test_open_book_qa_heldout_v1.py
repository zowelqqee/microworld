from worldpgt.benchmarks.open_book_qa.heldout_v1 import (
    build_direct_negative_heldout_cases,
    build_heldout_cases,
    heldout_pool_diagnostics,
)
from worldpgt.benchmarks.open_book_qa.dataset import relation_id
from worldpgt.knowledge_pump.heldout_density_frontier import (
    attach_wikipedia_resolution_layer,
    attach_wikidata_exact_resolution,
    attach_wikidata_alias_disambiguated_resolution,
    build_density_frontier,
    require_wikipedia_anchor,
)
from worldpgt.knowledge_pump.wikidata_relation_layer import extract_relation_rows
from worldpgt.experiments.run_structured_entity_seed_v1 import (
    _select_bounded_entities,
    _selector_query,
)


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


def test_direct_negative_heldout_is_disjoint_and_negative_is_absent_from_full_subject_bundle():
    relations = []
    for index in range(40):
        subject = f"Entity {index:02d}"
        relations.extend([
            _relation(subject, "uses", f"resource {index}"),
            _relation(subject, "supports", f"outcome {index}"),
        ])

    cases, summary = build_direct_negative_heldout_cases(relations, set())
    assert summary["cases_per_category"] == {"direct": 20, "negative": 20}
    assert summary["zero_overlap_relation_ids"] is True
    assert summary["direct_and_negative_subjects_disjoint"] is True
    direct_subjects = {case["expected_subject"] for case in cases if case["category"] == "direct"}
    negative_cases = [case for case in cases if case["category"] == "negative"]
    assert not direct_subjects & {case["expected_subject"] for case in negative_cases}
    for case in negative_cases:
        visible_predicates = {row.split("|")[1] for row in case["relation_ids"]}
        assert set(case["expected_predicate"]).isdisjoint(visible_predicates)
        assert case["expected_decision"] == "unknown"


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


def test_wikidata_resolution_keeps_all_cohorts_and_requires_one_exact_item():
    frontier = [
        {"subject": "Canonical entity", "surface_subject": "Canonical entity"},
        {"subject": "A fragment", "surface_subject": "A fragment"},
        {"subject": "Ambiguous", "surface_subject": "Ambiguous"},
    ]
    rows = attach_wikidata_exact_resolution(frontier, {
        "canonical entity": [{"id": "Q1", "label": "Canonical entity"}],
        "a fragment": [{"id": "Q2", "label": "Different entity"}],
        "ambiguous": [{"id": "Q3", "label": "Ambiguous"}, {"id": "Q4", "label": "Ambiguous"}],
    })

    assert [row["surface_subject"] for row in rows] == ["Canonical entity", "A fragment", "Ambiguous"]
    assert rows[0]["canonical_qid"] == "Q1"
    assert rows[0]["canonical_resolution_status"] == "resolved_wikidata_exact"
    assert rows[1]["canonical_resolution_status"] == "unresolved_wikidata_exact"
    assert rows[2]["canonical_resolution_status"] == "ambiguous_wikidata_exact"


def test_alias_and_p31_disambiguation_are_explicit_opt_in():
    rows = attach_wikidata_alias_disambiguated_resolution(
        [{"subject": "XMPP", "surface_subject": "XMPP"}, {"subject": "neural network", "surface_subject": "neural network"}],
        {
            "xmpp": [{"id": "Q1", "label": "Extensible Messaging and Presence Protocol", "display": {"label": {"language": "en"}}, "match": {"type": "alias", "text": "XMPP"}}],
            "neural network": [{"id": "Q2", "label": "neural network", "display": {"label": {"language": "en"}}}, {"id": "Q3", "label": "neural network", "display": {"label": {"language": "en"}}}],
        },
        entities={
            "Q1": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q7397"}}}}]}},
            "Q2": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q7397"}}}}]}},
            "Q3": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q13442814"}}}}]}},
        },
    )
    assert [row["canonical_qid"] for row in rows] == ["Q1", "Q2"]
    assert rows[0]["canonical_resolution_method"] == "unique_wikidata_alias"
    assert rows[1]["canonical_resolution_method"] == "exact_label_disambiguated_by_p31"


def test_alias_disambiguation_fixtures_cover_diagnostic_acronyms_and_cnn():
    """Wikidata's alias fields, rather than a local acronym list, drive fallback."""
    rows = attach_wikidata_alias_disambiguated_resolution(
        [
            {"subject": "CISM"}, {"subject": "COFs"}, {"subject": "XMPP"},
            {"subject": "convolutional neural network"},
        ],
        {
            "cism": [{"id": "Q1308825", "label": "Common Information Sharing Environment", "display": {"label": {"language": "en"}}, "match": {"type": "alias", "text": "CISM"}}],
            "cofs": [{"id": "Q38028371", "label": "covalent organic framework", "display": {"label": {"language": "en"}}, "match": {"type": "alias", "text": "COFs"}}],
            "xmpp": [{"id": "Q188951", "label": "Extensible Messaging and Presence Protocol", "display": {"label": {"language": "en"}}, "match": {"type": "alias", "text": "XMPP"}}],
            "convolutional neural network": [
                {"id": "Q17084460", "label": "convolutional neural network", "display": {"label": {"language": "en"}}},
                {"id": "Q999", "label": "convolutional neural network", "display": {"label": {"language": "en"}}},
            ],
        },
        entities={
            "Q1308825": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q7397"}}}}]}},
            "Q38028371": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q7397"}}}}]}},
            "Q188951": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q7397"}}}}]}},
            "Q17084460": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q1936384"}}}}]}},
            "Q999": {"claims": {"P31": [{"mainsnak": {"datavalue": {"value": {"id": "Q13442814"}}}}]}},
        },
    )
    assert [row["canonical_qid"] for row in rows] == ["Q1308825", "Q38028371", "Q188951", "Q17084460"]


def test_wikidata_auto_proposals_require_an_independent_wikipedia_anchor():
    resolved = [
        {
            "surface_subject": "Anchored", "canonical_qid": "Q1",
            "canonical_resolution_status": "resolved_wikidata_exact",
        },
        {
            "surface_subject": "Unanchored", "canonical_qid": "Q2",
            "canonical_resolution_status": "resolved_wikidata_exact",
        },
    ]
    rows = require_wikipedia_anchor(resolved, {
        "Q1": {"sitelinks": {"enwiki": {"title": "Anchored"}}},
        "Q2": {"sitelinks": {}},
    })

    assert rows[0]["canonical_resolution_status"] == "resolved_wikidata_exact_enwiki"
    assert rows[0]["canonical_source_url"].endswith("/Anchored")
    assert rows[1]["canonical_resolution_status"] == "exact_wikidata_without_enwiki_anchor"
    assert rows[1]["canonical_qid"] == "Q2"


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


def test_wikidata_relation_layer_maps_structured_corporate_claims():
    rows = extract_relation_rows(
        surface_subject="Example company", canonical_entity="Example company", canonical_qid="Q1",
        claims={
            "P112": [{"mainsnak": {"datavalue": {"value": {"id": "Q2"}}}}],
            "P127": [{"mainsnak": {"datavalue": {"value": {"id": "Q3"}}}}],
            "P159": [{"mainsnak": {"datavalue": {"value": {"id": "Q4"}}}}],
            "P1056": [{"mainsnak": {"datavalue": {"value": {"id": "Q5"}}}}],
        },
        labels={"Q2": "Founder", "Q3": "Owner", "Q4": "City", "Q5": "Product"},
    )

    assert {(row["wikidata_property"], row["predicate"], row["object"]) for row in rows} == {
        ("P112", "founded_by", "Founder"),
        ("P127", "owned_by", "Owner"),
        ("P159", "headquartered_in", "City"),
        ("P1056", "produces", "Product"),
    }


def test_structured_seed_selector_requires_two_relation_groups_and_is_bounded():
    rows = [
        {"canonical_qid": "Q1", "canonical_entity": "One", "predicate": "uses", "object_qid": "Q10"},
        {"canonical_qid": "Q1", "canonical_entity": "One", "predicate": "used_for", "object_qid": "Q11"},
        {"canonical_qid": "Q2", "canonical_entity": "Two", "predicate": "uses", "object_qid": "Q12"},
        {"canonical_qid": "Q3", "canonical_entity": "Three", "predicate": "uses", "object_qid": "Q13"},
        {"canonical_qid": "Q3", "canonical_entity": "Three", "predicate": "used_for", "object_qid": "Q14"},
        {"canonical_qid": "Q3", "canonical_entity": "Three", "predicate": "developed_by", "object_qid": "Q15"},
        {"canonical_qid": "Q3", "canonical_entity": "Three", "predicate": "runs_on", "object_qid": "Q16"},
        {"canonical_qid": "Q3", "canonical_entity": "Three", "predicate": "produces", "object_qid": "Q17"},
        {"canonical_qid": "Q3", "canonical_entity": "Three", "predicate": "owned_by", "object_qid": "Q18"},
        {"canonical_qid": "Q3", "canonical_entity": "Three", "predicate": "founded_by", "object_qid": "Q19"},
    ]
    selected, manifest = _select_bounded_entities(rows, max_entities=5, max_per_family_object=1)

    assert {row["canonical_qid"] for row in selected} == {"Q1"}
    assert manifest[0]["predicate_groups"] == ["used_for", "uses"]
    query = _selector_query(10)
    assert "wdt:P178" in query and "wdt:P112" in query and "wdt:P127" in query
