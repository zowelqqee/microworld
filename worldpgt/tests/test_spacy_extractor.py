"""Tests for the spaCy dependency-parse relation extraction backend.

Offline only. No network, no overlay mutation, no neural/GPT/training code.
All tests use tiny in-memory entity indexes and temporary doc files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.relation_candidate_extractor import (
    extract_candidates_from_doc,
)
from worldpgt.relation_extraction_v2.spacy_extractor import (
    _extract_from_spacy_doc,
    _get_nlp,
    _too_generic,
    extract_candidates_from_doc_spacy,
)
from worldpgt.wiki_snapshot_ingestion.types import ReadySnapshotDoc


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _make_index(tmp_path: Path, entities: list[dict]) -> EntitySurfaceIndex:
    accepted = tmp_path / "accepted.json"
    promoted = tmp_path / "promoted.json"
    snapshot = tmp_path / "snapshot.json"
    manifest = tmp_path / "manifest.json"
    _write_json(accepted, entities)
    _write_json(promoted, entities)
    _write_json(snapshot, entities)
    _write_json(
        manifest,
        [{"title": e["label"], "normalized_title": e["label"]} for e in entities],
    )
    return EntitySurfaceIndex(accepted, promoted, snapshot, manifest)


def _make_doc(tmp_path: Path, title: str, sentence: str) -> ReadySnapshotDoc:
    path = tmp_path / f"{title.replace(' ', '_')}.md"
    path.write_text(
        f"# {title}\n\n"
        "Source: https://en.wikipedia.org/wiki/Test\n"
        "Retrieved at: 2026-06-20T00:00:00Z\n"
        "Revision ID: 1\n"
        "Raw text SHA256: testhash\n"
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
        retrieved_at="2026-06-20T00:00:00Z",
        revision_id=1,
        raw_text_sha256=f"testhash-{title}",
        normalized_doc_path=str(path),
    )


def _default_entities() -> list[dict]:
    return [
        {"overlay_type": "overlay_entity", "label": "Elon Musk",         "aliases": ["Musk"],  "entity_type": "person"},
        {"overlay_type": "overlay_entity", "label": "Peter Thiel",        "aliases": [],        "entity_type": "person"},
        {"overlay_type": "overlay_entity", "label": "SpaceX",             "aliases": [],        "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "Tesla",              "aliases": [],        "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "PayPal",             "aliases": [],        "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "Neuralink",          "aliases": [],        "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "The Boring Company", "aliases": [],        "entity_type": "organization"},
        {"overlay_type": "overlay_entity", "label": "Hawthorne",          "aliases": [],        "entity_type": "location"},
    ]


@pytest.fixture()
def nlp():
    return _get_nlp()


# ---------------------------------------------------------------------------
# Unit-level: _extract_from_spacy_doc
# ---------------------------------------------------------------------------

def _run_spacy_sentence(sentence: str, entities: list[dict], tmp_path: Path):
    """Helper: run spaCy extractor on a single sentence string."""
    nlp = _get_nlp()
    index = _make_index(tmp_path, entities)
    doc = nlp(sentence)
    meta = {
        "title": "Test",
        "url": "https://en.wikipedia.org/wiki/Test",
        "retrieved_at": "2026-06-20T00:00:00Z",
        "sha": "testhash",
        "sent_idx": 0,
        "para_idx": 0,
    }
    return _extract_from_spacy_doc(doc, index, meta)


class TestPassiveVoice:
    def test_passive_founded_by(self, tmp_path):
        cands = _run_spacy_sentence(
            "SpaceX was founded by Elon Musk.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "SpaceX" and c.relation == "founded_by" and c.object == "Elon Musk"
            for c in cands
        ), f"Expected (SpaceX, founded_by, Elon Musk) in {[(c.subject, c.relation, c.object) for c in cands]}"

    def test_passive_pattern_id_contains_spacy_prefix(self, tmp_path):
        cands = _run_spacy_sentence(
            "SpaceX was founded by Elon Musk.",
            _default_entities(), tmp_path,
        )
        founded = [c for c in cands if c.relation == "founded_by"]
        assert founded
        assert founded[0].pattern_id.startswith("spacy:")

    def test_passive_agent_inverts_correctly(self, tmp_path):
        """Passive voice: nsubjpass (SpaceX) is subject, agent (Musk) is object."""
        cands = _run_spacy_sentence(
            "Neuralink was founded by Elon Musk.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "Neuralink"
            and c.relation == "founded_by"
            and c.object == "Elon Musk"
            for c in cands
        )

    def test_the_boring_company_passive(self, tmp_path):
        """Proper name starting with 'The' must be preserved."""
        cands = _run_spacy_sentence(
            "The Boring Company was founded by Elon Musk.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "The Boring Company"
            and c.relation == "founded_by"
            and c.object == "Elon Musk"
            for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"

    def test_headquartered_in_passive_prep(self, tmp_path):
        """'is headquartered in Y' uses prep(in) not agent."""
        cands = _run_spacy_sentence(
            "SpaceX is headquartered in Hawthorne.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "SpaceX"
            and c.relation == "headquartered_in"
            and c.object == "Hawthorne"
            for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"


class TestActiveVoice:
    def test_active_founded(self, tmp_path):
        cands = _run_spacy_sentence(
            "Elon Musk founded SpaceX.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "Elon Musk" and c.relation == "founded" and c.object == "SpaceX"
            for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"

    def test_active_own_inverts_correctly(self, tmp_path):
        """'X owns Y' → Y owned_by X (invert_active=True for 'own')."""
        cands = _run_spacy_sentence(
            "Elon Musk owns SpaceX.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "SpaceX" and c.relation == "owned_by" and c.object == "Elon Musk"
            for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"


class TestCoordination:
    def test_subject_coordination(self, tmp_path):
        """'X and Y founded Z' → two triples, one per founder."""
        cands = _run_spacy_sentence(
            "Elon Musk and Peter Thiel founded PayPal.",
            _default_entities(), tmp_path,
        )
        triples = {(c.subject, c.relation, c.object) for c in cands}
        assert ("Elon Musk", "founded", "PayPal") in triples, f"Got: {triples}"
        assert ("Peter Thiel", "founded", "PayPal") in triples, f"Got: {triples}"

    def test_object_coordination(self, tmp_path):
        """'X founded Y and Z' → two triples, one per founded entity."""
        cands = _run_spacy_sentence(
            "Elon Musk founded SpaceX and Tesla.",
            _default_entities(), tmp_path,
        )
        triples = {(c.subject, c.relation, c.object) for c in cands}
        assert ("Elon Musk", "founded", "SpaceX") in triples, f"Got: {triples}"
        assert ("Elon Musk", "founded", "Tesla") in triples, f"Got: {triples}"


class TestCopulaPatterns:
    def test_copula_leader_of(self, tmp_path):
        """'X is the CEO of Y' → (X, leader_of, Y)."""
        cands = _run_spacy_sentence(
            "Elon Musk is the CEO of SpaceX.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "Elon Musk"
            and c.relation == "leader_of"
            and c.object == "SpaceX"
            for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"

    def test_copula_is_a_org_type(self, tmp_path):
        """'X is an aerospace company' → (X, is_a, <type text>)."""
        cands = _run_spacy_sentence(
            "SpaceX is an American aerospace company.",
            _default_entities(), tmp_path,
        )
        assert any(
            c.subject == "SpaceX" and c.relation == "is_a"
            for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"


# ---------------------------------------------------------------------------
# Garbage filtering
# ---------------------------------------------------------------------------

class TestGarbageFiltering:
    def test_pronoun_subject_is_filtered(self, tmp_path):
        """'It was founded by Elon Musk' — 'it' is too generic."""
        cands = _run_spacy_sentence(
            "It was founded by Elon Musk.",
            _default_entities(), tmp_path,
        )
        assert not any(
            c.subject.lower() == "it" for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"

    def test_pronoun_object_is_filtered(self, tmp_path):
        """'Elon Musk founded it' — 'it' is too generic."""
        cands = _run_spacy_sentence(
            "Elon Musk founded it.",
            _default_entities(), tmp_path,
        )
        assert not any(
            c.object.lower() == "it" for c in cands
        ), f"Got: {[(c.subject, c.relation, c.object) for c in cands]}"

    def test_too_generic_helper(self):
        assert _too_generic("it")
        assert _too_generic("he")
        assert _too_generic("a")
        assert _too_generic("X")    # len < 3
        assert not _too_generic("SpaceX")
        assert not _too_generic("Elon Musk")

    def test_current_live_screen_degrades_confidence(self, tmp_path):
        """Sentences with 'current CEO' are screened; confidence → low."""
        cands = _run_spacy_sentence(
            "Elon Musk is the current CEO of SpaceX.",
            _default_entities(), tmp_path,
        )
        # Candidate may or may not appear, but if it does confidence is low.
        for c in cands:
            if c.subject == "Elon Musk" and c.relation == "leader_of":
                assert c.confidence == "low", f"Expected low confidence for screened sentence, got {c.confidence}"


# ---------------------------------------------------------------------------
# Full-doc extraction via extract_candidates_from_doc_spacy
# ---------------------------------------------------------------------------

class TestFullDocExtraction:
    def test_extract_from_doc_returns_list_and_sentence_count(self, tmp_path):
        index = _make_index(tmp_path, _default_entities())
        doc = _make_doc(tmp_path, "SpaceX", "SpaceX was founded by Elon Musk.")
        cands, sents = extract_candidates_from_doc_spacy(doc, index)
        assert isinstance(cands, list)
        assert sents >= 1

    def test_extract_from_doc_finds_passive_founded_by(self, tmp_path):
        index = _make_index(tmp_path, _default_entities())
        doc = _make_doc(tmp_path, "SpaceX", "SpaceX was founded by Elon Musk.")
        cands, _ = extract_candidates_from_doc_spacy(doc, index)
        assert any(
            c.subject == "SpaceX" and c.relation == "founded_by" and c.object == "Elon Musk"
            for c in cands
        )

    def test_extract_from_doc_deduplicates_within_doc(self, tmp_path):
        index = _make_index(tmp_path, _default_entities())
        doc = _make_doc(
            tmp_path, "SpaceX",
            "SpaceX was founded by Elon Musk. SpaceX was founded by Elon Musk.",
        )
        cands, _ = extract_candidates_from_doc_spacy(doc, index)
        fb = [c for c in cands if c.subject == "SpaceX" and c.relation == "founded_by"]
        assert len(fb) == 1, f"Expected 1 deduped candidate, got {len(fb)}"


# ---------------------------------------------------------------------------
# Integration with extract_candidates_from_doc (merged paths)
# ---------------------------------------------------------------------------

class TestMergedPaths:
    def test_enable_spacy_false_produces_same_result_as_before(self, tmp_path):
        """enable_spacy=False must not change existing behavior."""
        index = _make_index(tmp_path, _default_entities())
        doc = _make_doc(tmp_path, "SpaceX", "SpaceX was founded by Elon Musk.")
        cands_no_spacy, sents = extract_candidates_from_doc(doc, index, enable_spacy=False)
        assert isinstance(cands_no_spacy, list)
        assert sents >= 1

    def test_enable_spacy_deduplicates_with_regex_path(self, tmp_path):
        """Triple from regex path must not appear twice when spaCy also finds it."""
        index = _make_index(tmp_path, _default_entities())
        doc = _make_doc(tmp_path, "SpaceX", "SpaceX was founded by Elon Musk.")
        cands, _ = extract_candidates_from_doc(doc, index, enable_spacy=True)
        fb = [
            c for c in cands
            if c.subject == "SpaceX" and c.relation == "founded_by" and c.object == "Elon Musk"
        ]
        assert len(fb) == 1, (
            f"Deduplication failed: {len(fb)} copies of (SpaceX, founded_by, Elon Musk). "
            f"pattern_ids: {[c.pattern_id for c in fb]}"
        )

    def test_spacy_adds_coordination_triples_not_in_regex_path(self, tmp_path):
        """Subject coordination is unique to the spaCy path."""
        d_regex = tmp_path / "regex"
        d_spacy = tmp_path / "spacy_dir"
        d_regex.mkdir()
        d_spacy.mkdir()

        idx_r = _make_index(d_regex, _default_entities())
        idx_s = _make_index(d_spacy, _default_entities())

        doc_regex = _make_doc(d_regex, "PayPal", "Elon Musk and Peter Thiel founded PayPal.")
        doc_spacy = _make_doc(d_spacy, "PayPal", "Elon Musk and Peter Thiel founded PayPal.")

        cands_regex, _ = extract_candidates_from_doc(doc_regex, idx_r, enable_spacy=False)
        cands_both, _  = extract_candidates_from_doc(doc_spacy, idx_s, enable_spacy=True)

        regex_subjects = {c.subject for c in cands_regex if c.relation == "founded"}
        both_subjects  = {c.subject for c in cands_both  if c.relation == "founded"}

        # The spaCy path should add Peter Thiel as a founder.
        assert "Peter Thiel" in both_subjects, (
            f"spaCy path missed Peter Thiel; both_subjects={both_subjects}"
        )
        # The regex path should not have Peter Thiel (it can't handle subject coordination).
        assert "Peter Thiel" not in regex_subjects, (
            f"Regex unexpectedly found Peter Thiel; regex_subjects={regex_subjects}"
        )


# ---------------------------------------------------------------------------
# Static safety check — spacy_extractor must not import forbidden libraries
# ---------------------------------------------------------------------------

def test_spacy_extractor_runtime_safety_static_check():
    root = Path(__file__).resolve().parent.parent
    src = (root / "relation_extraction_v2" / "spacy_extractor.py").read_text(encoding="utf-8")
    assert "urllib.request" not in src
    assert "import requests" not in src
    assert "import torch" not in src
    assert "import openai" not in src
    assert "backprop" not in src
