"""Focused tests for Relation Extraction v2 founded/founded_by directionality.

Offline only. No network, no overlay mutation, no neural/GPT/training code.
"""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_candidate_extractor import extract_candidates_from_doc
from worldpgt.relation_extraction_v2.relation_candidate_validator import validate_candidates
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _index(tmp_path: Path) -> EntitySurfaceIndex:
    entities = [
        {"overlay_type": "overlay_entity", "label": "Elon Musk", "aliases": ["Musk"], "entity_type": "person"},
        {"overlay_type": "overlay_entity", "label": "SpaceX", "aliases": [], "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "Neuralink", "aliases": [], "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "The Boring Company", "aliases": [], "entity_type": "organization"},
    ]
    accepted = tmp_path / "accepted.json"
    promoted = tmp_path / "promoted.json"
    snapshot = tmp_path / "snapshot.json"
    manifest = tmp_path / "manifest.json"
    _write_json(accepted, entities)
    _write_json(promoted, entities)
    _write_json(snapshot, entities)
    _write_json(
        manifest,
        [
            {"title": "Elon Musk", "normalized_title": "Elon Musk"},
            {"title": "SpaceX", "normalized_title": "SpaceX"},
            {"title": "Neuralink", "normalized_title": "Neuralink"},
            {"title": "The Boring Company", "normalized_title": "The Boring Company"},
        ],
    )
    return EntitySurfaceIndex(accepted, promoted, snapshot, manifest)


def _doc(tmp_path: Path, title: str, sentence: str) -> ReadySnapshotDoc:
    path = tmp_path / f"{title.replace(' ', '_')}.md"
    path.write_text(
        f"# {title}\n\n"
        "Source: https://en.wikipedia.org/wiki/Test\n"
        "Retrieved at: 2026-06-15T00:00:00Z\n"
        "Revision ID: 1\n"
        "Raw text SHA256: hash\n"
        "Status: LOCAL_WIKIPEDIA_SNAPSHOT\n"
        "Safe for accepted memory: false\n"
        "Requires ingestion/quarantine/promotion/regression: true\n\n"
        f"{sentence}\n",
        encoding="utf-8",
    )
    return ReadySnapshotDoc(
        title=title,
        normalized_title=title,
        source_url="https://en.wikipedia.org/wiki/Test",
        retrieved_at="2026-06-15T00:00:00Z",
        revision_id=1,
        raw_text_sha256=f"hash-{title}",
        normalized_doc_path=str(path),
    )


def _validated(tmp_path: Path, title: str, sentence: str):
    index = _index(tmp_path)
    raw, _scanned = extract_candidates_from_doc(_doc(tmp_path, title, sentence), index)
    safe, quarantine, _dups, _conflicts = validate_candidates(raw, index, [])
    return safe, quarantine


def test_spacex_was_founded_by_elon_musk_is_accepted(tmp_path):
    safe, quarantine = _validated(tmp_path, "SpaceX", "SpaceX was founded by Elon Musk.")
    assert any(c.subject == "SpaceX" and c.relation == "founded_by" and c.object == "Elon Musk" for c in safe)
    assert not quarantine


def test_neuralink_was_founded_by_elon_musk_is_accepted(tmp_path):
    safe, quarantine = _validated(tmp_path, "Neuralink", "Neuralink was founded by Elon Musk.")
    assert any(c.subject == "Neuralink" and c.relation == "founded_by" and c.object == "Elon Musk" for c in safe)
    assert not quarantine


def test_the_boring_company_reduced_passive_founded_by_is_accepted(tmp_path):
    safe, quarantine = _validated(
        tmp_path,
        "The Boring Company",
        "The Boring Company is an American infrastructure company founded by Elon Musk.",
    )
    assert any(
        c.subject == "The Boring Company"
        and c.relation == "founded_by"
        and c.object == "Elon Musk"
        for c in safe
    )
    assert not quarantine


def test_elon_musk_founded_spacex_is_accepted(tmp_path):
    safe, quarantine = _validated(tmp_path, "Elon Musk", "Elon Musk founded SpaceX.")
    assert any(c.subject == "Elon Musk" and c.relation == "founded" and c.object == "SpaceX" for c in safe)
    assert not quarantine


def test_spacex_founded_elon_musk_is_quarantined(tmp_path):
    safe, quarantine = _validated(tmp_path, "SpaceX", "SpaceX founded Elon Musk.")
    assert not any(c.subject == "SpaceX" and c.relation == "founded" and c.object == "Elon Musk" for c in safe)
    assert any("entity_type_conflict" in q.reason for q in quarantine)


def test_relation_extraction_v2_runtime_safety_static_checks():
    root = Path(__file__).resolve().parent.parent
    files = list((root / "relation_extraction_v2").glob("*.py"))
    text = "\n".join(path.read_text(encoding="utf-8").lower() for path in files)
    assert "urllib.request" not in text
    assert "import requests" not in text
    assert "import torch" not in text
    assert "import openai" not in text
    assert "backprop" not in text
    assert not (root / "nanogpt").exists()
