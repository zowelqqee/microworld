"""Tests for proposal-only lead definition backfill."""

from __future__ import annotations

import json

from worldpgt.knowledge_pump.lead_definition_backfill import (
    backfill_overlay_definitions,
    extract_lead_definition,
)


def test_extracts_definition_after_alias_clause() -> None:
    sentence = (
        "World War II, or the Second World War (1 September 1939 - 2 September 1945), "
        "was a global conflict between two coalitions: the Allies and the Axis powers."
    )

    assert (
        extract_lead_definition("World War II", sentence)
        == "global conflict between two coalitions: the Allies and the Axis powers"
    )


def test_strips_renderer_article_for_country_definition() -> None:
    sentence = (
        "The United Kingdom of Great Britain and Northern Ireland, commonly known as the "
        "United Kingdom (UK) or Britain, is a country in northwestern Europe, off the "
        "coast of the continental mainland."
    )

    assert extract_lead_definition("United Kingdom", sentence) == "country in northwestern Europe"


def test_extracts_longer_definition_with_with_clause() -> None:
    sentence = (
        "Mathematics is a field of knowledge concerned with abstract concepts such as "
        "numbers, geometric shapes, sets, functions, and probabilities."
    )

    assert (
        extract_lead_definition("Mathematics", sentence)
        == "field of knowledge concerned with abstract concepts such as numbers"
    )


def test_keeps_comma_list_when_prefix_is_incomplete() -> None:
    economics = (
        "Economics is a social science that studies the production, distribution, "
        "and consumption of goods and services."
    )
    industrial = (
        "The Industrial Revolution was a transitional period of the global economy "
        "toward more widespread, efficient and stable manufacturing processes, "
        "succeeding the Second Agricultural Revolution."
    )

    assert (
        extract_lead_definition("Economics", economics)
        == "social science that studies the production, distribution, and consumption of goods and services"
    )
    assert (
        extract_lead_definition("Industrial Revolution", industrial)
        == "transitional period of the global economy toward more widespread, efficient and stable manufacturing processes"
    )


def test_extracts_any_collection_definition() -> None:
    assert (
        extract_lead_definition("Literature", "Literature is any collection of written work.")
        == "collection of written work"
    )


def test_extracts_lasted_approximately_period() -> None:
    sentence = (
        "In the history of Europe, the Middle Ages or medieval period lasted "
        "approximately from the 5th to late 15th centuries, comparable with the "
        "post-classical period of global history."
    )

    assert (
        extract_lead_definition("Middle Ages", sentence)
        == "period in the history of Europe lasting approximately from the 5th to late 15th centuries"
    )


def test_extracts_includes_both_definition() -> None:
    sentence = (
        "Present-day climate change includes both global warming—the ongoing increase "
        "in global average temperature—and its wider effects on Earth's climate system."
    )

    assert (
        extract_lead_definition("Climate change", sentence)
        == "present-day change that includes both global warming—the ongoing increase in global average temperature—and its wider effects on Earth's climate system"
    )


def test_backfill_overlay_definitions_is_idempotent(tmp_path) -> None:
    docs_dir = tmp_path / "normalized_docs"
    docs_dir.mkdir()
    (docs_dir / "World_War_II.md").write_text(
        "\n".join(
            [
                "# World War II",
                "",
                "Source: https://en.wikipedia.org/wiki/World_War_II",
                "Retrieved at: 2026-07-04T19:36:52Z",
                "Raw text SHA256: abc",
                "Status: LOCAL_WIKIPEDIA_SNAPSHOT",
                "Safe for accepted memory: false",
                "Requires ingestion/quarantine/promotion/regression: true",
                "World War II, or the Second World War, was a global conflict between two coalitions.",
            ]
        ),
        encoding="utf-8",
    )
    overlay_json = tmp_path / "pump_dry_run_overlay.json"
    overlay_json.write_text(
        json.dumps([
            {
                "overlay_type": "overlay_entity",
                "label": "World War II",
                "source_page": "World War II",
                "risk": "low",
            }
        ]),
        encoding="utf-8",
    )

    first = backfill_overlay_definitions(overlay_json=overlay_json, docs_dir=docs_dir)
    second = backfill_overlay_definitions(overlay_json=overlay_json, docs_dir=docs_dir)

    items = json.loads(overlay_json.read_text(encoding="utf-8"))
    definitions = [item for item in items if item.get("overlay_type") == "overlay_definition"]
    assert first["added_count"] == 1
    assert second["added_count"] == 1
    assert second["previous_backfill_removed_count"] == 1
    assert len(definitions) == 1
    assert definitions[0]["subject"] == "World War II"
