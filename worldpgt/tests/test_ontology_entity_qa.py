"""Entity QA tests for explicit ontology ``is_a`` traversal."""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.assistant_surface import answer_orchestrator
from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.entity_qa.entity_answer_planner import EntityAnswerPlanner
from worldpgt.entity_qa.entity_answer_renderer import render
from worldpgt.entity_qa.entity_question_analyzer import analyze
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider


def _entity(label: str) -> dict:
    return {
        "overlay_type": "overlay_entity",
        "entity_id": f"test:{label}",
        "label": label,
        "aliases": [],
        "entity_type": "other",
    }


def _typed_entity(label: str, entity_type: str) -> dict:
    item = _entity(label)
    item["entity_type"] = entity_type
    return item


def _rel(subject: str, obj: str, *, stability: str = "stable", risk: str = "low") -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": "is_a",
        "object": obj,
        "stability": stability,
        "risk": risk,
        "trust": "overlay_candidate",
    }


def _relation(
    subject: str,
    predicate: str,
    obj: str,
    *,
    stability: str = "stable",
    risk: str = "low",
) -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "stability": stability,
        "risk": risk,
        "trust": "overlay_candidate",
    }


def _planner(tmp_path: Path, items: list[dict]) -> EntityAnswerPlanner:
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return EntityAnswerPlanner(WikiMemoryOverlayProvider(path))


def _answer(planner: EntityAnswerPlanner, question: str):
    analyzed = analyze(question)
    plan = planner.plan(analyzed)
    return plan, render(plan)


def test_is_a_direct_one_hop_answers(tmp_path):
    planner = _planner(tmp_path, [
        _entity("SpaceX"),
        _rel("SpaceX", "aerospace manufacturer"),
    ])

    plan, answer = _answer(planner, "Is SpaceX an aerospace manufacturer?")

    assert plan.decision == "answer"
    assert answer == "Yes. SpaceX is an aerospace manufacturer."
    assert "explicit is_a chain: 1 hops" in plan.evidence.overlay_items_used


def test_is_a_transitive_chain_answers_with_full_chain(tmp_path):
    planner = _planner(tmp_path, [
        _entity("SpaceX"),
        _rel("SpaceX", "aerospace manufacturer"),
        _rel("aerospace manufacturer", "manufacturer"),
        _rel("manufacturer", "organization"),
    ])

    plan, answer = _answer(planner, "Is SpaceX an organization?")

    assert plan.decision == "answer"
    assert answer == (
        "Yes. SpaceX is an aerospace manufacturer, "
        "which is a manufacturer, which is an organization."
    )
    assert "explicit is_a chain: 3 hops" in plan.evidence.overlay_items_used


def test_is_a_explicit_type_contradiction_returns_no(tmp_path):
    planner = _planner(tmp_path, [
        _entity("SpaceX"),
        _rel("SpaceX", "organization"),
    ])

    plan, answer = _answer(planner, "Is SpaceX a person?")

    assert plan.decision == "no"
    assert plan.render_args["support_kind"] == "explicit_type_contradiction"
    assert answer == "No. SpaceX is known to be an organization, which contradicts a person."


def test_is_a_entity_type_mismatch_returns_no(tmp_path):
    planner = _planner(tmp_path, [
        _typed_entity("SpaceX", "organization"),
    ])

    plan, answer = _answer(planner, "Is SpaceX a person?")

    assert plan.decision == "no"
    assert plan.render_args["support_kind"] == "entity_type_mismatch"
    assert answer == (
        "No. SpaceX is an organization, not a person. The question's premise "
        "is incompatible with SpaceX's known type."
    )


def test_is_a_weak_coverage_audits_not_no(tmp_path):
    planner = _planner(tmp_path, [
        _entity("Acme"),
        _relation("Acme", "develops", "Widget"),
        _relation("Acme", "produces", "Gadget"),
        _relation("Acme", "owned_by", "ParentCo"),
    ])

    plan, answer = _answer(planner, "Is Acme a contractor?")

    assert plan.decision == "audit"
    assert "not found in well-covered entity, verify externally" in (plan.audit_reason or "")
    assert any(kw in answer.lower() for kw in ("cannot answer", "don't have", "i don't", "no verified"))


def test_is_a_volatile_contradiction_basis_audits_not_no(tmp_path):
    planner = _planner(tmp_path, [
        _entity("Acme"),
        _rel("Acme", "organization", stability="volatile"),
    ])

    plan, answer = _answer(planner, "Is Acme a person?")

    assert plan.decision == "audit"
    assert any(kw in answer.lower() for kw in ("cannot answer", "don't have", "i don't", "no verified"))


def test_is_a_absent_fact_audits_not_no(tmp_path):
    planner = _planner(tmp_path, [
        _entity("UnknownCo"),
    ])

    plan, answer = _answer(planner, "Is UnknownCo a company?")

    assert plan.decision == "audit"
    assert any(kw in answer.lower() for kw in ("cannot answer", "don't have", "i don't", "no verified"))


def test_is_a_reverse_inference_audits(tmp_path):
    planner = _planner(tmp_path, [
        _entity("SpaceX"),
        _entity("Organization"),
        _rel("SpaceX", "organization"),
    ])

    plan, _answer_text = _answer(planner, "Is organization a SpaceX?")

    assert plan.decision == "audit"


def test_is_a_volatile_chain_audits(tmp_path):
    planner = _planner(tmp_path, [
        _entity("SpaceX"),
        _rel("SpaceX", "aerospace manufacturer"),
        _rel("aerospace manufacturer", "organization", stability="volatile"),
    ])

    plan, _answer_text = _answer(planner, "Is SpaceX an organization?")

    assert plan.decision == "audit"


def test_assistant_surface_routes_is_a_class_question_to_ontology_chain(tmp_path):
    path = tmp_path / "overlay.json"
    path.write_text(json.dumps([
        _entity("SpaceX"),
        _rel("SpaceX", "aerospace manufacturer"),
        _rel("aerospace manufacturer", "manufacturer"),
        _rel("manufacturer", "organization"),
    ], indent=2), encoding="utf-8")

    answer = AnswerOrchestrator(overlay_path=str(path)).answer("Is SpaceX an organization?")

    assert answer.decision == "answer"
    assert answer.support_kind == "explicit_is_a_chain"
    assert "SpaceX is an aerospace manufacturer" in answer.answer_text
    assert "which is an organization" in answer.answer_text


def test_assistant_surface_uses_explicit_ontology_layer_path(tmp_path):
    overlay_path = tmp_path / "overlay.json"
    ontology_path = tmp_path / "ontology_layer.json"
    overlay_path.write_text(json.dumps([
        _entity("Elon Musk"),
        _rel("Elon Musk", "businessman"),
    ], indent=2), encoding="utf-8")
    ontology_path.write_text(json.dumps([
        _rel("businessman", "worker"),
        _rel("worker", "person with an activity"),
    ], indent=2), encoding="utf-8")

    answer = AnswerOrchestrator(
        overlay_path=str(overlay_path),
        ontology_layer_path=str(ontology_path),
    ).answer("Is Elon Musk a person with an activity?")

    assert answer.decision == "answer"
    assert answer.support_kind == "explicit_is_a_chain"
    assert answer.answer_text == (
        "Yes. Elon Musk is a businessman, which is a worker, "
        "which is a person with an activity."
    )


def test_assistant_surface_auto_loads_default_ontology_layer_when_present(tmp_path, monkeypatch):
    overlay_path = tmp_path / "overlay.json"
    ontology_path = tmp_path / "wikidata_p279_ontology_layer.json"
    overlay_path.write_text(json.dumps([
        _entity("Elon Musk"),
        _rel("Elon Musk", "businessman"),
    ], indent=2), encoding="utf-8")
    ontology_path.write_text(json.dumps([
        _rel("businessman", "worker"),
    ], indent=2), encoding="utf-8")
    monkeypatch.setattr(
        answer_orchestrator,
        "DEFAULT_WIKIDATA_P279_ONTOLOGY_LAYER_PATH",
        ontology_path,
    )

    answer = AnswerOrchestrator(overlay_path=str(overlay_path)).answer("Is Elon Musk a worker?")

    assert answer.decision == "answer"
    assert answer.answer_text == "Yes. Elon Musk is a businessman, which is a worker."
