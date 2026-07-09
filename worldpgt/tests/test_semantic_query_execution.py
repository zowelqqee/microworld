"""Execution tests for semantic query types beyond plain lookup."""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.surface_validator import validate_answer


def _entity(label: str, aliases: list[str] | None = None) -> dict:
    return {
        "overlay_type": "overlay_entity",
        "entity_id": f"test:{label}",
        "label": label,
        "aliases": aliases or [],
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


def test_definition_uses_entity_alias_subject(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("Sam Walton", aliases=["Samuel Moore Walton", "Walton"]),
            _definition("Samuel Moore Walton", "American philanthropist"),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer("What is Sam Walton?")

    assert answer.decision == "answer"
    assert answer.support_kind == "stable_definition"
    assert answer.answer_text == "Samuel Moore Walton is an American philanthropist."
    assert validate_answer(answer) == []


def test_definition_uses_source_page_lead_subject_without_alias(tmp_path):
    entity = _entity("John D. Rockefeller")
    entity["source_page"] = "John D. Rockefeller"
    overlay_path = _overlay(
        tmp_path,
        [
            entity,
            {
                **_definition("John Davison Rockefeller Sr.", "American businessman"),
                "source_page": "John D. Rockefeller",
            },
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer(
        "What is John D. Rockefeller?"
    )

    assert answer.decision == "answer"
    assert answer.support_kind == "stable_definition"
    assert answer.answer_text == "John Davison Rockefeller Sr. is an American businessman."
    assert validate_answer(answer) == []


def test_generic_tail_alias_does_not_define_narrow_compound(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("Space medicine", aliases=["Medicine"]),
            _definition("Space medicine", "subspecialty of emergency medicine"),
            _entity("Transport economics", aliases=["Economics"]),
            _definition("Transport economics", "branch of economics"),
        ],
    )
    orchestrator = AnswerOrchestrator(overlay_path=overlay_path)

    medicine = orchestrator.answer("What is medicine?")
    economics = orchestrator.answer("What is economics?")

    assert medicine.decision == "audit"
    assert economics.decision == "audit"


def test_broad_topic_alias_does_not_define_unrelated_entity(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("School district", aliases=["North America", "district"]),
            _definition(
                "School district",
                "administrative body for education institutions",
            ),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer(
        "What is North America?"
    )

    assert answer.decision == "audit"
    assert "School district" not in answer.answer_text


def test_where_located_uses_exact_raw_subject_not_longer_global_match(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("France"),
            _relation("France", "located_in", "Western Europe", stability="semi_stable"),
            {
                **_relation("France", "located_in", "French", stability="semi_stable"),
                "evidence_text": "Provincia Nostra evolved into Provence in French.",
            },
            _entity("African Americans in France"),
            _relation("African Americans in France", "located_in", "Paris", stability="semi_stable"),
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer("Where is France located?")

    assert answer.decision == "answer"
    assert "France is located in Western Europe" in answer.answer_text
    assert "French" not in answer.answer_text
    assert "African Americans in France" not in answer.answer_text


def test_definition_filters_abstract_located_in_relation(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _entity("Islam"),
            _definition(
                "Islam",
                "Abrahamic religion based on the Quran and the teachings of Muhammad",
            ),
            {
                **_relation(
                    "Islam",
                    "located_in",
                    "Islamic mystical teachings",
                    stability="semi_stable",
                ),
                "evidence_text": (
                    "It is usually thought of as a precise monotheism, but is "
                    "also panentheistic in Islamic mystical teachings."
                ),
            },
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer("What is Islam?")

    assert answer.decision == "answer"
    assert answer.answer_text.startswith("Islam is an Abrahamic religion")
    assert "located in Islamic mystical teachings" not in answer.answer_text


def test_definition_filters_located_in_from_unrelated_source_page(tmp_path):
    overlay_path = _overlay(
        tmp_path,
        [
            _definition("Germany", "country in Western and Central Europe"),
            {
                **_relation(
                    "Germany",
                    "located_in",
                    "Bremen",
                    stability="semi_stable",
                ),
                "source_page": "EADS Astrium",
                "evidence_text": (
                    "The company has facilities in France and in Germany; "
                    "the main facility in Germany is located in Bremen."
                ),
            },
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer("What is Germany?")

    assert answer.decision == "answer"
    assert answer.answer_text == "Germany is a country in Western and Central Europe."
    assert "Bremen" not in answer.answer_text


def test_definition_source_page_fallback_skips_copula_fragments(tmp_path):
    entity = _entity("Bill Gates")
    entity["source_page"] = "Bill Gates"
    overlay_path = _overlay(
        tmp_path,
        [
            entity,
            {
                **_definition(
                    "Born and raised in Seattle, Washington, Gates",
                    "privately educated at Lakeside School",
                ),
                "source_page": "Bill Gates",
            },
            {
                **_definition("William Henry Gates III", "American businessman"),
                "source_page": "Bill Gates",
            },
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer("What is Bill Gates?")

    assert answer.decision == "answer"
    assert answer.support_kind == "stable_definition"
    assert answer.answer_text == "William Henry Gates III is an American businessman."
    assert "Born and raised" not in answer.answer_text
    assert validate_answer(answer) == []


def test_definition_renderer_handles_one_of_without_article(tmp_path):
    entity = _entity("Apple Inc.")
    entity["source_page"] = "Apple Inc."
    overlay_path = _overlay(
        tmp_path,
        [
            entity,
            {
                **_definition("Apple", "one of the Big Tech companies"),
                "source_page": "Apple Inc.",
            },
        ],
    )

    answer = AnswerOrchestrator(overlay_path=overlay_path).answer("What is Apple Inc.?")

    assert answer.decision == "answer"
    assert answer.answer_text == "Apple is one of the Big Tech companies."
    assert "an one" not in answer.answer_text
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
