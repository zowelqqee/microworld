from worldpgt.experiments.run_llm_manual_review_batch_v1 import (
    _sentence_rows,
    group_for_manual_review,
    node_quality_triage,
)


def test_sentence_selection_keeps_one_literal_cue_sentence_per_source() -> None:
    records = [
        {
            "title": "First record",
            "source_url": "https://arxiv.org/abs/one",
            "text": "First System uses a bounded artifact for testing. Another sentence supports a second object.",
        },
        {
            "title": "Second record",
            "source_url": "https://arxiv.org/abs/two",
            "text": "Second System enables a bounded workflow for researchers.",
        },
    ]

    rows = _sentence_rows(records)

    assert len(rows) == 2
    assert {row["source_title"] for row in rows} == {"First record", "Second record"}
    assert all("source_fingerprint" in row for row in rows)


def test_triage_keeps_literal_clean_named_relation_only_for_manual_review() -> None:
    rows = [
        {
            "id": "arxiv-000",
            "model": "gemini-3.1-flash-lite",
            "source_title": "Stored arXiv record",
            "source_url": "https://arxiv.org/abs/example",
            "source_fingerprint": "test",
            "source_text": "SciServer uses SkyServer.",
            "triples": [
                {
                    "subject": "SciServer",
                    "predicate": "uses",
                    "object": "SkyServer",
                    "evidence_span": "SciServer uses SkyServer",
                }
            ],
        }
    ]

    review, rejected = node_quality_triage(rows)

    assert rejected == []
    assert len(review) == 1
    assert review[0]["candidate_id"] == "arxiv-000:0"
    assert review[0]["manual_review"]["verdict"] == ""
    assert review[0]["literal_subject"] is True
    assert review[0]["literal_object"] is True


def test_grouping_labels_candidates_without_assigning_a_verdict() -> None:
    candidates = [
        {
            "candidate_id": "one",
            "subject": "This system",
            "predicate": "uses",
            "object": "SkyServer",
            "source_text": "This system uses SkyServer.",
            "manual_review": {"verdict": ""},
        },
        {
            "candidate_id": "two",
            "subject": "result",
            "predicate": "uses",
            "object": "a method",
            "source_text": "In 2020, a result uses a method.",
            "manual_review": {"verdict": ""},
        },
        {
            "candidate_id": "three",
            "subject": "MetaChem",
            "predicate": "has",
            "object": "a formal description",
            "source_text": "MetaChem has a formal description.",
            "manual_review": {"verdict": ""},
        },
        {
            "candidate_id": "four",
            "subject": "SciServer",
            "predicate": "extends",
            "object": "SkyServer",
            "source_text": "Tools: SciServer extends SkyServer, with support.",
            "manual_review": {"verdict": ""},
        },
        {
            "candidate_id": "five",
            "subject": "SciServer",
            "predicate": "uses",
            "object": "SkyServer",
            "source_text": "SciServer uses SkyServer.",
            "manual_review": {"verdict": ""},
        },
        {
            "candidate_id": "six",
            "subject": "Result",
            "predicate": "uses",
            "object": "a method",
            "source_text": "Result may use a method in a future experiment.",
            "manual_review": {"verdict": ""},
        },
    ]

    grouped, primary_counts, flag_counts = group_for_manual_review(candidates)
    primary = {candidate["candidate_id"]: candidate["primary_review_group"] for candidate in grouped}

    assert primary == {
        "one": "anaphora_likely",
        "two": "temporal_referent_likely",
        "three": "generic_property_likely",
        "four": "attachment_shape_flag",
        "five": "clean_no_flag",
        "six": "clean_no_flag",
    }
    assert primary_counts == {
        "anaphora_likely": 1,
        "attachment_shape_flag": 1,
        "clean_no_flag": 2,
        "generic_property_likely": 1,
        "temporal_referent_likely": 1,
    }
    assert flag_counts["anaphora_likely"] == 1
    assert all(candidate["manual_review"]["verdict"] == "" for candidate in grouped)
