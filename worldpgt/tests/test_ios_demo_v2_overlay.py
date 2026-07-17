"""Regression coverage for the offline iPhone v2 serving graph."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "ios_demo" / "scripts" / "build_ios_serving_overlay.py"


def _module():
    spec = importlib.util.spec_from_file_location("ios_demo_v2_overlay_builder", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_ios_v2_overlay_contains_all_approved_new_lanes(tmp_path):
    builder = _module()
    overlay, summary = builder.build(_ROOT)

    assert summary["offline_only"] is True
    assert summary["accepted_memory_modified"] is False
    assert summary["promoted_wiki_overlay_modified"] is False
    assert summary["validated_multi_evidence_subject_cohorts"] == {
        "original_331": 331,
        "crossref_multi_evidence": 46,
        "wikidata_multi_evidence": 12,
        "openalex_multi_evidence": 2,
    }
    assert summary["added_relation_counts_by_lane"] == {
        "crossref": 113,
        "openalex": 4,
        "original_campaign": 713,
        "wikidata": 73,
    }
    assert any(
        row.get("subject") == "The Economics of Superstars"
        and row.get("predicate") == "has_topic"
        for row in overlay
    )
    output = tmp_path / "extended_serving_overlay.json"
    output.write_text(json.dumps(overlay, ensure_ascii=False), encoding="utf-8")
    assert output.is_file()
