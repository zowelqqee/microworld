from __future__ import annotations

import json

from worldpgt.knowledge_pump.dynamic_frontier import (
    extract_internal_link_titles,
    load_dynamic_frontier,
    merge_frontiers,
    update_dynamic_frontier_from_fetch_rows,
)
from worldpgt.knowledge_pump.expanded_allowlist_builder import build_expanded_allowlist
from worldpgt.knowledge_pump.types import ExpandedAllowlistEntry, FrontierTitle


def _allow_entry(title: str) -> ExpandedAllowlistEntry:
    return ExpandedAllowlistEntry(
        title=title,
        normalized_title=title,
        priority=1,
        reason="seed",
        source="test",
        risk_hint="low",
        already_fetched=False,
        selected_for_batch=False,
        batch_index=0,
    )


def test_extracts_explicit_internal_wikipedia_links_from_text():
    text = "See [[New Linked Page|surface]], /wiki/Another_Page, and https://en.wikipedia.org/wiki/Third_Page."
    assert extract_internal_link_titles(text) == ["New Linked Page", "Another Page", "Third Page"]


def test_dynamic_frontier_updates_from_successful_fetch_rows(tmp_path):
    raw_path = tmp_path / "seed.json"
    raw_path.write_text(
        json.dumps(
            {
                "title": "Seed Page",
                "normalized_title": "Seed Page",
                "fetch_status": "success",
                "raw_text": "Also links to [[Markdown Linked Page]] and /wiki/Url_Linked_Page.",
                "links": [
                    "Fresh Linked Page",
                    "Existing Allowlist Page",
                    "Seed Page",
                    "Fresh Page (qualifier)",
                    "File:Logo.svg",
                    "2024",
                ],
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "title": "Seed Page",
            "normalized_title": "Seed Page",
            "fetch_status": "success",
            "ready_for_self_ingestion": False,
            "raw_snapshot_path": str(raw_path),
        }
    ]

    summary = update_dynamic_frontier_from_fetch_rows(
        rows,
        dynamic_frontier_path=tmp_path / "dynamic_frontier_titles.json",
        dynamic_frontier_csv_path=tmp_path / "dynamic_frontier_titles.csv",
        already_fetched_titles={"Seed Page"},
        current_allowlist=[_allow_entry("Existing Allowlist Page")],
    )

    titles = [item.title for item in load_dynamic_frontier(tmp_path / "dynamic_frontier_titles.json")]
    assert titles == ["Fresh Linked Page", "Fresh Page (qualifier)", "Markdown Linked Page", "Url Linked Page"]
    assert summary["dynamic_frontier_added_this_run"] == 4
    assert summary["dynamic_frontier_rejected_already_fetched"] == 1
    assert summary["dynamic_frontier_rejected_current_allowlist"] == 1
    assert summary["dynamic_frontier_rejected_hygiene"] == 2


def test_dynamic_frontier_merges_into_allowlist_planning():
    static = [FrontierTitle("Static Seed", "overlay_entity", "seed", 6)]
    dynamic = [FrontierTitle("Dynamic Graph Page", "dynamic_wiki_link", "internal link", 6)]

    allowlist = build_expanded_allowlist(
        merge_frontiers([static, dynamic]),
        target_total=10,
        batch_size=5,
        already_fetched=set(),
    )

    titles = {entry.normalized_title for entry in allowlist}
    assert "Static Seed" in titles
    assert "Dynamic Graph Page" in titles
