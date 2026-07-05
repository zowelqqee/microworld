"""Tests for targeted broad-question gap refetch planning."""

from __future__ import annotations

import json

from worldpgt.experiments.run_question_gap_refetch_v1 import (
    _add_redirect_definition_aliases,
    build_question_gap_plan,
    title_from_question,
)
from worldpgt.relation_extraction_v2.sentence_splitter import extract_full_body


def test_title_from_question_strips_leading_article() -> None:
    assert title_from_question("What is the Industrial Revolution?") == "Industrial Revolution"
    assert title_from_question("What is ancient history?") == "Ancient history"
    assert title_from_question("Where is France located?") is None


def test_extract_full_body_skips_empty_metadata_values() -> None:
    text = "\n".join(
        [
            "# Brazil",
            "",
            "Source: https://en.wikipedia.org/wiki/Brazil",
            "Retrieved at: 2026-07-04T19:37:36Z",
            "Revision ID: ",
            "Raw text SHA256: ",
            "Status: LOCAL_WIKIPEDIA_SNAPSHOT",
            "Safe for accepted memory: false",
            "Requires ingestion/quarantine/promotion/regression: true",
        ]
    )

    assert extract_full_body(text) == ""


def test_question_gap_plan_distinguishes_empty_and_body_docs(tmp_path) -> None:
    rows_path = tmp_path / "rows.json"
    rows_path.write_text(
        json.dumps(
            [
                {
                    "question": "What is Brazil?",
                    "category": "geography",
                    "label": "missing_desired",
                },
                {
                    "question": "What is climate change?",
                    "category": "science",
                    "label": "missing_desired",
                },
                {
                    "question": "What is France?",
                    "category": "geography",
                    "label": "good_answer",
                },
            ]
        ),
        encoding="utf-8",
    )
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "Brazil.md").write_text(
        "\n".join(
            [
                "# Brazil",
                "",
                "Source: https://en.wikipedia.org/wiki/Brazil",
                "Revision ID: ",
                "Raw text SHA256: ",
                "Status: LOCAL_WIKIPEDIA_SNAPSHOT",
            ]
        ),
        encoding="utf-8",
    )
    (docs_dir / "Climate_change.md").write_text(
        "\n".join(
            [
                "# Climate change",
                "",
                "Source: https://en.wikipedia.org/wiki/Climate_change",
                "Revision ID: 1",
                "Raw text SHA256: abc",
                "Status: LOCAL_WIKIPEDIA_SNAPSHOT",
                (
                    "Present-day climate change includes both global warming and its wider "
                    "effects on Earth's climate system."
                ),
            ]
        ),
        encoding="utf-8",
    )

    plan = build_question_gap_plan(audit_rows_json=rows_path, docs_dir=docs_dir)

    assert plan["gap_count"] == 2
    by_title = {row["title"]: row for row in plan["rows"]}
    assert by_title["Brazil"]["action"] == "network_refetch_needed"
    assert by_title["Climate change"]["action"] == "try_local_backfill"


def test_add_redirect_definition_aliases_adds_requested_subject(tmp_path) -> None:
    overlay_json = tmp_path / "overlay.json"
    overlay_json.write_text(
        json.dumps(
            [
                {
                    "overlay_type": "overlay_definition",
                    "subject": "Transport",
                    "definition": "intentional movement of humans",
                    "source_page": "Transport",
                    "trust": "overlay_candidate",
                    "risk": "low",
                    "stability": "stable",
                }
            ]
        ),
        encoding="utf-8",
    )

    report = _add_redirect_definition_aliases(
        overlay_json,
        [{"title": "Transportation", "normalized_title": "Transport"}],
    )

    items = json.loads(overlay_json.read_text(encoding="utf-8"))
    alias = [item for item in items if item.get("subject") == "Transportation"]
    assert report["redirect_alias_added_count"] == 1
    assert len(alias) == 1
    assert alias[0]["definition"] == "intentional movement of humans"
    assert alias[0]["redirect_target"] == "Transport"
