"""Short Entity Safety v1 — regression tests for _MIN_SURFACE_LEN=2.

Knowledge Pump v1.8 lowered _MIN_SURFACE_LEN from 3 to 2 so 2-char acronyms
like "XM" (which has pump overlay founded_by facts) can be matched. These tests
lock in two guarantees:

  (a) "XM" is matchable and correctly answered for its supported founded_by facts.
  (b) Unsupported 2-char tokens (AI, US, UK, HP, EU) and single-char tokens (X)
      that are not in the pump overlay never produce a false "answer" decision.

No network. No overlay writes. No memory modifications. Pump overlay = proposal/dry-run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.context_pack.entity_matcher import build_surface_index, match_entities
from worldpgt.experiments import ask_microworld_v1
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_REPO = Path(__file__).resolve().parent.parent.parent
_EXP = _REPO / "worldpgt" / "experiments"
_PUMP_OVERLAY = _EXP / "knowledge_pump_v1" / "pump_dry_run_overlay.json"
_PUMP_QA_SUMMARY = _EXP / "knowledge_pump_v1" / "pump_fact_qa_v1" / "pump_fact_qa_summary.json"

_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _OverlayAccess:
    """Thin adapter so entity_matcher can read the pump overlay."""

    def __init__(self, provider: WikiMemoryOverlayProvider) -> None:
        self._p = provider

    def entities(self):
        return self._p.all_entities()

    def definitions(self):
        return self._p.all_definitions()

    def relations(self):
        return self._p.all_relations()

    def source_facts(self):
        return self._p.all_source_facts()


class _MockOverlay:
    """Stub overlay that only holds what the test injects."""

    def __init__(self, entities=None, definitions=None, relations=None, source_facts=None):
        self._entities = entities or []
        self._definitions = definitions or []
        self._relations = relations or []
        self._source_facts = source_facts or []

    def entities(self):
        return self._entities

    def definitions(self):
        return self._definitions

    def relations(self):
        return self._relations

    def source_facts(self):
        return self._source_facts


@pytest.fixture(scope="module")
def pump_overlay_access():
    provider = WikiMemoryOverlayProvider(_PUMP_OVERLAY)
    return _OverlayAccess(provider)


def _ask(question: str):
    return ask_microworld_v1.ask(question, "pump-dry-run")


# ===========================================================================
# Category 1: XM is matchable and answerable for supported founded_by facts
# ===========================================================================


def test_xm_who_founded_answers():
    a = _ask("Who founded XM?")
    assert a.decision == "answer", f"Expected answer, got: {a.answer_text}"


def test_xm_who_founded_contains_lon_levin():
    a = _ask("Who founded XM?")
    assert "Lon Levin" in a.answer_text


def test_xm_who_founded_contains_gary_parsons():
    a = _ask("Who founded XM?")
    assert "Gary Parsons" in a.answer_text


def test_xm_was_founded_by_answers():
    a = _ask("Who was XM founded by?")
    assert a.decision == "answer", f"Expected answer, got: {a.answer_text}"


def test_xm_was_founded_by_contains_lon_levin():
    a = _ask("Who was XM founded by?")
    assert "Lon Levin" in a.answer_text


def test_xm_was_founded_by_contains_gary_parsons():
    a = _ask("Who was XM founded by?")
    assert "Gary Parsons" in a.answer_text


def test_xm_founded_supported_by_context():
    a = _ask("Who founded XM?")
    assert a.supported_by_context is True


# ===========================================================================
# Category 2: "What is XM?" audits — no stable definition/is_a for XM
# ===========================================================================


def test_what_is_xm_audits():
    a = _ask("What is XM?")
    assert a.decision == "audit", f"Expected audit, got answer: {a.answer_text}"


def test_what_is_xm_not_supported_by_context():
    a = _ask("What is XM?")
    assert not a.supported_by_context


# ===========================================================================
# Category 3: Unsupported short tokens audit
# ===========================================================================


@pytest.mark.parametrize(
    "question",
    [
        "Who founded X?",
        "What is X?",
        "What is AI?",
        "What is US?",
        "What is UK?",
        "What is HP?",
        "Who founded HP?",
        "What does HP own?",
        "Who founded EU?",
        "What is developed by US?",
        "What is developed by UK?",
        "What is developed by AI?",
    ],
)
def test_unsupported_short_token_audits(question):
    a = _ask(question)
    assert a.decision == "audit", (
        f"Expected audit for {question!r}, got decision={a.decision!r} "
        f"answer_text={a.answer_text!r}"
    )


# ===========================================================================
# Category 4: No unsupported short-token query returns Decision: answer
# ===========================================================================


_UNSUPPORTED_SHORT_TOKEN_QUESTIONS = [
    "Who founded X?",
    "What is X?",
    "What is AI?",
    "What is US?",
    "What is UK?",
    "What is HP?",
    "Who founded HP?",
    "What does HP own?",
    "Who founded EU?",
    "What is developed by US?",
    "What is developed by UK?",
    "What is developed by AI?",
]


def test_no_unsupported_short_token_returns_answer():
    failures = []
    for q in _UNSUPPORTED_SHORT_TOKEN_QUESTIONS:
        a = _ask(q)
        if a.decision == "answer":
            failures.append(f"{q!r} -> answer: {a.answer_text!r}")
    assert not failures, "Unsupported short tokens returned 'answer':\n" + "\n".join(failures)


# ===========================================================================
# Category 5a: Entity matcher — unsupported 2-char tokens absent from pump overlay
# ===========================================================================


def test_entity_matcher_ai_not_in_pump_overlay(pump_overlay_access):
    matched = match_entities("What is AI?", pump_overlay_access)
    names_norm = {m.name.lower() for m in matched}
    assert "ai" not in names_norm
    surfaces_norm = {m.surface.lower() for m in matched}
    assert "ai" not in surfaces_norm


def test_entity_matcher_us_not_in_pump_overlay(pump_overlay_access):
    matched = match_entities("What is US?", pump_overlay_access)
    names_norm = {m.name.lower() for m in matched}
    assert "us" not in names_norm


def test_entity_matcher_uk_not_in_pump_overlay(pump_overlay_access):
    matched = match_entities("What is UK?", pump_overlay_access)
    names_norm = {m.name.lower() for m in matched}
    assert "uk" not in names_norm


def test_entity_matcher_hp_not_in_pump_overlay(pump_overlay_access):
    matched = match_entities("What is HP?", pump_overlay_access)
    names_norm = {m.name.lower() for m in matched}
    assert "hp" not in names_norm


def test_entity_matcher_eu_not_in_pump_overlay(pump_overlay_access):
    matched = match_entities("What is EU?", pump_overlay_access)
    names_norm = {m.name.lower() for m in matched}
    assert "eu" not in names_norm


# ===========================================================================
# Category 5b: Entity matcher — no substring/fuzzy matching (mock overlay)
# ===========================================================================


def test_ai_does_not_match_arianespace_by_substring():
    overlay = _MockOverlay(
        entities=[
            {
                "label": "Arianespace",
                "entity_id": "arianespace",
                "entity_type": "organization",
                "aliases": [],
            }
        ]
    )
    matched = match_entities("What is AI?", overlay)
    names = {m.name for m in matched}
    assert "Arianespace" not in names, "AI matched Arianespace by substring"


def test_us_does_not_match_united_states_by_substring():
    overlay = _MockOverlay(
        entities=[
            {
                "label": "United States",
                "entity_id": "united-states",
                "entity_type": "country",
                "aliases": [],
            }
        ]
    )
    matched = match_entities("What is US?", overlay)
    names = {m.name for m in matched}
    assert "United States" not in names, "US matched United States by substring"


def test_hp_does_not_match_hewlett_packard_without_explicit_alias():
    overlay = _MockOverlay(
        entities=[
            {
                "label": "Hewlett-Packard",
                "entity_id": "hp-inc",
                "entity_type": "organization",
                "aliases": [],
            }
        ]
    )
    matched = match_entities("What is HP?", overlay)
    names = {m.name for m in matched}
    assert "Hewlett-Packard" not in names, "HP matched Hewlett-Packard without alias"


def test_x_does_not_match_everything(pump_overlay_access):
    # "X" is a 1-char token — filtered by _MIN_SURFACE_LEN=2 (len<2 is False for len=1)
    # Nothing in the pump overlay should match bare "X" as a word boundary entity.
    matched = match_entities("Who founded X?", pump_overlay_access)
    # Tesla Model X has alias "X" (1 char) — still filtered. No entity named "X" exists
    # as a standalone surface.
    names_that_are_just_x = [m for m in matched if m.surface.strip() == "X"]
    assert not names_that_are_just_x, (
        f"'X' matched an entity as standalone surface: {names_that_are_just_x}"
    )


def test_ai_does_not_match_artificial_intelligence_label():
    overlay = _MockOverlay(
        entities=[
            {
                "label": "Artificial Intelligence",
                "entity_id": "ai-concept",
                "entity_type": "concept",
                "aliases": [],
            }
        ]
    )
    matched = match_entities("What is AI?", overlay)
    names = {m.name for m in matched}
    assert "Artificial Intelligence" not in names


def test_two_char_explicit_alias_matches_when_present():
    """Only an explicit 2-char alias creates a match — not substring."""
    overlay = _MockOverlay(
        entities=[
            {
                "label": "XM Satellite Radio",
                "entity_id": "xm",
                "entity_type": "organization",
                "aliases": ["XM"],
            }
        ]
    )
    matched = match_entities("Who founded XM?", overlay)
    names = {m.name for m in matched}
    assert "XM Satellite Radio" in names, "Explicit 2-char alias 'XM' was not matched"


# ===========================================================================
# Category 6: Existing pump fact QA summary remains green
# ===========================================================================


@pytest.fixture(scope="module")
def pump_qa_summary():
    assert _PUMP_QA_SUMMARY.exists(), f"Pump QA summary not found at {_PUMP_QA_SUMMARY}"
    return json.loads(_PUMP_QA_SUMMARY.read_text(encoding="utf-8"))


def test_pump_qa_wrong_count_is_zero(pump_qa_summary):
    assert pump_qa_summary.get("pump_fact_qa_positive_wrong_count", 0) == 0


def test_pump_qa_unsupported_count_is_zero(pump_qa_summary):
    assert pump_qa_summary.get("pump_fact_qa_positive_unsupported_count", 0) == 0


def test_pump_qa_adversarial_fail_count_is_zero(pump_qa_summary):
    assert pump_qa_summary.get("pump_fact_qa_adversarial_fail_count", 0) == 0


def test_pump_qa_current_safety_fail_count_is_zero(pump_qa_summary):
    assert pump_qa_summary.get("pump_fact_qa_current_safety_fail_count", 0) == 0


def test_pump_qa_supported_answer_rate_is_one(pump_qa_summary):
    assert pump_qa_summary.get("pump_fact_qa_supported_answer_rate") == 1.0


def test_pump_qa_all_critical_passed(pump_qa_summary):
    assert pump_qa_summary.get("pump_fact_qa_all_critical_passed") is True


# ===========================================================================
# Safety contract: protected files unchanged
# ===========================================================================


def test_protected_files_not_modified():
    for path in _PROTECTED:
        if path.exists():
            assert path.is_file(), f"Protected path {path} is not a regular file"
