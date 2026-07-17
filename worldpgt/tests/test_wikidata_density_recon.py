from worldpgt.knowledge_pump.wikidata_density_recon import content_property_ids, summarize_property_density


def _claim(*, datatype="wikibase-item", rank="normal", snaktype="value"):
    return {"rank": rank, "mainsnak": {"snaktype": snaktype, "datatype": datatype}}


def test_density_excludes_structural_identifier_and_deprecated_claims():
    entity = {"claims": {
        "P31": [_claim()],
        "P646": [_claim(datatype="external-id")],
        "P856": [_claim(datatype="url")],
        "P178": [_claim()],
        "P5008": [_claim()],
        "P999": [_claim()],
        "P998": [_claim(rank="deprecated")],
    }}

    assert content_property_ids(entity) == {"P178", "P999"}


def test_density_marks_only_unmapped_content_properties_as_new_potential():
    summary, rows = summarize_property_density(
        [
            {"subject": "One", "canonical_qid": "Q1", "cohorts": ["original"]},
            {"subject": "Two", "canonical_qid": "Q2", "cohorts": ["crossref"]},
        ],
        {
            "Q1": {"claims": {"P178": [_claim()], "P999": [_claim()], "P998": [_claim()]}},
            "Q2": {"claims": {"P50": [_claim()]}},
        },
        {"P999": "New relation", "P998": "Another relation", "P178": "developer"},
    )

    assert rows[0]["new_schema_property_ids"] == ["P998", "P999"]
    assert rows[1]["new_schema_property_ids"] == []
    assert summary["subjects_with_two_or_more_new_schema_property_groups"] == 1
    assert summary["top_15_content_bearing_properties"][0]["property_id"] == "P178"
