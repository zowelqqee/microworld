"""Tests for Wiki Candidate Memory Overlay v1.

All tests are deterministic and offline. No network access. No modification
of sense_memory.py, accepted_knowledge_memory_v1.json, or any planner/validator.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worldpgt.knowledge.wiki_candidate_overlay_builder import (
    WikiCandidateOverlayBuilder,
    as_dict,
)
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.knowledge.wiki_memory_overlay_types import (
    SAFE_FOR_GENERAL_RUNTIME,
    OverlayContextLink,
    OverlayDefinition,
    OverlayEntity,
    OverlayRelation,
    OverlaySourceFact,
)

_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_CANDIDATES_JSON = _EXPERIMENTS / "wiki_ingestion_v2_candidates.json"
_OVERLAY_JSON = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_REPO = Path(__file__).parent.parent.parent
_SENSE_MEMORY = _REPO / "worldpgt" / "continuation" / "sense_memory.py"
_ACCEPTED = _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY_SHA = "66980abacf371edcefcfc3e2254de10f3582f04b"
_ACCEPTED_SHA = "0979a4e7ff25598a56c5dfe05be621112159fca4"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def candidates() -> list[dict]:
    return json.loads(_CANDIDATES_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def build_result(candidates):
    return WikiCandidateOverlayBuilder().build(candidates)


@pytest.fixture(scope="module")
def overlay_items(build_result):
    return build_result.items


@pytest.fixture(scope="module")
def provider():
    return WikiMemoryOverlayProvider(_OVERLAY_JSON)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _by_type(items, overlay_type):
    return [i for i in items if getattr(i, "overlay_type", None) == overlay_type]


# ---------------------------------------------------------------------------
# 1. entity_card → overlay_entity
# ---------------------------------------------------------------------------


def test_converts_entity_card_to_overlay_entity(overlay_items):
    entities = _by_type(overlay_items, "overlay_entity")
    assert len(entities) >= 45
    labels = {e.label for e in entities}
    assert "Elon Musk" in labels
    assert "Tesla" in labels
    assert "SpaceX" in labels
    assert "Forbes" in labels


def test_overlay_entity_fields(overlay_items):
    entities = _by_type(overlay_items, "overlay_entity")
    musk = next(e for e in entities if e.label == "Elon Musk")
    assert musk.entity_id == "wikidata:Q317521"
    assert musk.entity_type == "person"
    assert musk.trust == "overlay_candidate"
    assert musk.risk == "low"
    assert musk.source_candidate_type == "entity_card"


# ---------------------------------------------------------------------------
# 2. definition_claim → overlay_definition
# ---------------------------------------------------------------------------


def test_converts_definition_claim_to_overlay_definition(overlay_items):
    defs = _by_type(overlay_items, "overlay_definition")
    assert len(defs) >= 45
    subjects = {d.subject for d in defs}
    assert "Tesla" in subjects
    assert "SpaceX" in subjects


def test_overlay_definition_tesla(overlay_items):
    defs = _by_type(overlay_items, "overlay_definition")
    tesla_def = next(d for d in defs if d.subject == "Tesla")
    assert "electric vehicle" in tesla_def.definition.lower()
    assert tesla_def.stability == "stable"
    assert tesla_def.risk == "low"
    assert tesla_def.trust == "overlay_candidate"


# ---------------------------------------------------------------------------
# 3. relation_claim → overlay_relation
# ---------------------------------------------------------------------------


def test_converts_relation_claim_to_overlay_relation(overlay_items):
    rels = _by_type(overlay_items, "overlay_relation")
    assert len(rels) >= 19
    predicates = {r.predicate for r in rels}
    assert "leader_of" in predicates


def test_overlay_relation_leader_of(overlay_items):
    rels = _by_type(overlay_items, "overlay_relation")
    leader = [r for r in rels if r.predicate == "leader_of" and r.subject == "Elon Musk"]
    objects = {r.object for r in leader}
    assert "Tesla" in objects
    assert "SpaceX" in objects
    for r in leader:
        assert r.trust == "overlay_candidate"
        assert r.stability == "semi_stable"


# ---------------------------------------------------------------------------
# 4. context_link → overlay_context_link with weak trust
# ---------------------------------------------------------------------------


def test_converts_context_link_to_overlay_context_link(overlay_items):
    links = _by_type(overlay_items, "overlay_context_link")
    assert len(links) >= 27
    for lnk in links:
        assert lnk.strength == "weak"
        assert lnk.trust == "weak_context_only"


def test_context_link_is_not_treated_as_fact(overlay_items):
    links = _by_type(overlay_items, "overlay_context_link")
    for lnk in links:
        assert lnk.trust != "overlay_candidate"
        assert lnk.trust != "source_qualified_overlay_candidate"


# ---------------------------------------------------------------------------
# 5. source_qualified_fact → overlay_source_fact
# ---------------------------------------------------------------------------


def test_converts_source_qualified_fact_to_overlay_source_fact(overlay_items):
    facts = _by_type(overlay_items, "overlay_source_fact")
    assert len(facts) >= 1
    fact = next(f for f in facts if f.subject == "Elon Musk")
    assert fact.subject == "Elon Musk"
    assert fact.predicate == "estimated_net_worth"
    assert fact.source_name == "Forbes"
    assert fact.as_of == "2026-06"
    assert fact.requires_recheck is True
    assert fact.stability == "volatile"
    assert fact.risk == "high"
    assert fact.trust == "source_qualified_overlay_candidate"


# ---------------------------------------------------------------------------
# 6. Volatile source fact is NOT converted to overlay_relation
# ---------------------------------------------------------------------------


def test_volatile_source_fact_not_converted_to_relation(overlay_items):
    rels = _by_type(overlay_items, "overlay_relation")
    for r in rels:
        assert r.stability != "volatile"
        assert r.risk != "high"
    # The SQF is only in overlay_source_fact.
    facts = _by_type(overlay_items, "overlay_source_fact")
    assert any(f.predicate == "estimated_net_worth" for f in facts)


# ---------------------------------------------------------------------------
# 7. Summary: safe_for_general_runtime is False
# ---------------------------------------------------------------------------


def test_summary_safe_for_general_runtime_false(build_result):
    assert build_result.summary["safe_for_general_runtime"] is False
    assert SAFE_FOR_GENERAL_RUNTIME is False


# ---------------------------------------------------------------------------
# 8. Summary: safe_for_entity_qa_overlay is True (clean fixture)
# ---------------------------------------------------------------------------


def test_summary_safe_for_entity_qa_overlay_true(build_result):
    assert build_result.summary["safe_for_entity_qa_overlay"] is True


def test_summary_has_required_keys(build_result):
    s = build_result.summary
    for key in (
        "source_candidates_total",
        "overlay_items_total",
        "skipped_candidates_total",
        "by_overlay_type",
        "by_risk",
        "by_stability",
        "source_facts_count",
        "weak_context_links_count",
        "safe_for_general_runtime",
        "safe_for_entity_qa_overlay",
    ):
        assert key in s, f"Missing key in summary: {key}"


# ---------------------------------------------------------------------------
# 9. Accepted memory v1 unchanged
# ---------------------------------------------------------------------------


def test_accepted_memory_unchanged():
    assert _sha1(_ACCEPTED) == _ACCEPTED_SHA


# ---------------------------------------------------------------------------
# 10. sense_memory.py unchanged
# ---------------------------------------------------------------------------


def test_sense_memory_unchanged():
    assert _sha1(_SENSE_MEMORY) == _SENSE_MEMORY_SHA


def test_builder_does_not_modify_protected_files(candidates):
    before = (_sha1(_SENSE_MEMORY), _sha1(_ACCEPTED))
    WikiCandidateOverlayBuilder().build(candidates)
    after = (_sha1(_SENSE_MEMORY), _sha1(_ACCEPTED))
    assert before == after


# ---------------------------------------------------------------------------
# Overlay provider tests
# ---------------------------------------------------------------------------


def test_provider_resolves_entity_by_label(provider):
    entity = provider.get_entity("Elon Musk")
    assert entity is not None
    assert entity["label"] == "Elon Musk"
    assert entity["entity_id"] == "wikidata:Q317521"


def test_provider_resolves_entity_by_alias(provider):
    entity = provider.get_entity("Musk")
    assert entity is not None
    assert entity["label"] == "Elon Musk"


def test_provider_resolves_entity_case_insensitive(provider):
    entity = provider.get_entity("elon musk")
    assert entity is not None
    assert entity["label"] == "Elon Musk"


def test_provider_finds_tesla_definition(provider):
    defn = provider.get_definition("Tesla")
    assert defn is not None
    assert "electric vehicle" in defn["definition"].lower()


def test_provider_finds_musk_leader_of_tesla(provider):
    rels = provider.get_relations("Elon Musk", predicate="leader_of")
    objects = {r["object"] for r in rels}
    assert "Tesla" in objects
    assert "SpaceX" in objects


def test_provider_finds_forbes_context_link_from_musk(provider):
    links = provider.get_context_links(source_page="Elon Musk", target="Forbes")
    assert links
    for lnk in links:
        assert lnk["strength"] == "weak"
        assert lnk["trust"] == "weak_context_only"


def test_provider_finds_musk_net_worth_source_fact(provider):
    facts = provider.get_source_facts(subject="Elon Musk", predicate="estimated_net_worth")
    assert facts
    f = facts[0]
    assert f["source_name"] == "Forbes"
    assert f["as_of"] == "2026-06"
    assert f["requires_recheck"] is True
    assert f["stability"] == "volatile"


def test_provider_skipped_filter_returns_empty(provider):
    entity = provider.get_entity("NonExistentEntityXYZ")
    assert entity is None


def test_provider_explain_link(provider):
    links = provider.explain_link("Elon Musk", "Forbes")
    assert links
    assert all(lnk["strength"] == "weak" for lnk in links)


def test_provider_total_items(provider):
    total = provider.total_items()
    assert total >= 67
