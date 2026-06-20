"""Execution tests for semantic query types beyond plain lookup."""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.surface_validator import validate_answer


def _entity(label: str) -> dict:
    return {
        "overlay_type": "overlay_entity",
        "entity_id": f"test:{label}",
        "label": label,
        "aliases": [],
        "entity_type": "organization",
    }


def _definition(subject: str, text: str) -> dict:
    return {
        "overlay_type": "overlay_definition",
        "subject": subject,
        "definition": text,
        "stability": "stable",
        "trust": "overlay_candidate",
    }


def _relation(
    subject: str,
    predicate: str,
    obj: str,
    *,
    stability: str = "stable",
) -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "stability": stability,
        "risk": "low",
        "trust": "overlay_candidate",
    }


def _overlay(tmp_path: Path, items: list[dict]) -> str:
    path = tmp_path / "semantic_query_overlay.json"
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return str(path)


def test_comparative_intersection_answers_common_relation_and_class(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("SpaceX"),
            _entity("Blue Origin"),
            _relation("SpaceX", "is_a", "aerospace company"),
            _relation("Blue Origin", "is_a", "aerospace company"),
            _relation("SpaceX", "develops", "rockets", stability="semi_stable"),
            _relation("Blue Origin", "develops", "rockets", stability="semi_stable"),
            _relation("SpaceX", "develops", "spacecraft", stability="semi_stable"),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer(
        "What do SpaceX and Blue Origin have in common?"
    )

    assert answer.decision == "answer"
    assert answer.support_kind == "semi_stable_relation"
    assert "both are an aerospace company" in answer.answer_text
    assert "both develop rockets" in answer.answer_text
    assert validate_answer(answer) == []


def test_comparative_intersection_audits_when_empty(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("SpaceX"),
            _entity("Blue Origin"),
            _definition("SpaceX", "space company"),
            _definition("Blue Origin", "rocket engine company"),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer(
        "What do SpaceX and Blue Origin have in common?"
    )

    assert answer.decision == "audit"
    assert answer.answer_text == "no common facts in current overlay"


def test_inverse_aggregation_finds_subjects_by_object_and_relation(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("Tesla"),
            _entity("SolarCity"),
            _relation("SolarCity", "owned_by", "Tesla", stability="semi_stable"),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer(
        "Which companies does Tesla own?"
    )

    assert answer.decision == "answer"
    assert answer.support_kind == "semi_stable_relation"
    assert answer.answer_text == "Tesla owns SolarCity."
    assert validate_answer(answer) == []


def test_passive_inverse_owned_by_finds_entities_owned_by_known_object(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("SpaceX"),
            _entity("Starlink"),
            _relation("Starlink", "owned_by", "SpaceX", stability="semi_stable"),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer(
        "Which entities are owned by SpaceX?"
    )

    assert answer.decision == "answer"
    assert answer.answer_text == "SpaceX owns Starlink."
    assert validate_answer(answer) == []


def test_inverse_develops_finds_developer_of_known_object(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("SpaceX"),
            _entity("Falcon 9"),
            _relation("SpaceX", "develops", "Falcon 9", stability="semi_stable"),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer("Who develops Falcon 9?")

    assert answer.decision == "answer"
    assert answer.answer_text == "SpaceX develops Falcon 9."
    assert validate_answer(answer) == []
