from worldpgt.knowledge_pump.wikidata_resolution_gap import classify_manual_sample, seeded_original_failure_sample


def test_seeded_sample_is_reproducible_and_excludes_resolved_subjects():
    rows = [
        {"subject": f"Subject {index}", "cohorts": ["original_331"], "canonical_qid": None}
        for index in range(40)
    ] + [{"subject": "Resolved", "cohorts": ["original_331"], "canonical_qid": "Q1"}]

    first = seeded_original_failure_sample(rows, size=30, seed=42)
    second = seeded_original_failure_sample(rows, size=30, seed=42)

    assert [row["subject"] for row in first] == [row["subject"] for row in second]
    assert "Resolved" not in {row["subject"] for row in first}


def test_manual_classification_requires_a_complete_rationale():
    sample = [{"subject": "Alias", "canonical_resolution_status": "unresolved_wikidata_exact"}]
    rows = classify_manual_sample(sample, {
        "Alias": {
            "verdict": "matching_gap", "failure_type": "alias_mismatch",
            "correct_wikidata_qid": "Q1", "correct_wikidata_label": "Canonical Alias",
            "rationale": "The official search returns the canonical item through an alias.",
        }
    })

    assert rows[0]["verdict"] == "matching_gap"
    assert rows[0]["correct_wikidata_qid"] == "Q1"
