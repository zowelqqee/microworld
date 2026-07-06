"""Tests for semantic SPO-style question parsing."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from worldpgt.assistant_surface.question_router import route
from worldpgt.entity_qa.entity_question_analyzer import analyze
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.relation_extraction_v2.relation_policy import relation_intent_from_text

_ROOT = Path(__file__).resolve().parent.parent.parent
_CLI = _ROOT / "worldpgt" / "experiments" / "ask_microworld_v1.py"


def test_relation_keyword_mapping_covers_system_predicates():
    examples = {
        "Who founded SpaceX?": "founded_by",
        "SpaceX was started by whom?": "founded_by",
        "Which companies does Tesla own?": "owned_by",
        "What does SpaceX build?": "develops",
        "Who leads Starlink?": "leader_of",
        "Where is Tesla headquartered in?": "headquartered_in",
        "What is Starlink a service of?": "service_of",
        "What is Falcon 9 part of?": "part_of",
        "What is Elon Musk's estimated net worth?": "estimated_net_worth",
        "Where is Gigafactory Texas located?": "located_in",
        "What funded SpaceX Dragon?": "funded_by",
        "What is BYD M3 marketed as?": "marketed_as",
        "What does Oracle Database support?": "supports",
        "What does Oracle Database run on?": "runs_on",
        "When was Oracle Database first released?": "first_released",
        "What did Euronext Amsterdam merge with?": "merged_with",
        "When did XCOR Aerospace cease operations?": "ceased_operations",
    }
    for text, expected in examples.items():
        assert relation_intent_from_text(text) == expected


def test_demo_questions_parse_to_semantic_queries():
    expectations = {
        "Who founded SpaceX?": ("SpaceX", None, "founded_by", "object", "lookup"),
        "SpaceX was started by whom?": ("SpaceX", None, "founded_by", "object", "lookup"),
        "What does SpaceX and Blue Origin have in common?": (
            "SpaceX",
            "Blue Origin",
            None,
            "intersection",
            "comparative",
        ),
        "Which companies does Tesla own?": ("Tesla", None, "owned_by", "subject", "inverse"),
        "Which entities are owned by SpaceX?": ("SpaceX", None, "owned_by", "subject", "inverse"),
        "Who develops Falcon 9?": ("Falcon 9", None, "develops", "subject", "inverse"),
        "Where was SpaceX located?": ("SpaceX", None, "located_in", "object", "lookup"),
        "How is Starlink connected to Falcon 9?": (
            "Starlink",
            "Falcon 9",
            None,
            "path",
            "lookup",
        ),
        "Is Elon Musk a worker?": ("Elon Musk", "worker", "is_a", "relation", "is_a"),
        "What is Falcon 9?": ("Falcon 9", None, None, "relation", "definition"),
        "Who leads Starlink?": ("Starlink", None, "leader_of", "subject", "inverse"),
    }

    for question, expected in expectations.items():
        sq = parse_semantic_query(question)
        assert (
            sq.entity_a,
            sq.entity_b,
            sq.relation_intent,
            sq.unknown_position,
            sq.query_type,
        ) == expected
        assert sq.confidence >= 0.75


def test_semantic_parser_drives_analyzer_and_router():
    analyzed = analyze("SpaceX was started by whom?")
    assert analyzed.intent == "relation_lookup"
    assert analyzed.subject == "SpaceX"
    assert analyzed.predicate_hint == "founded_by"
    assert analyzed.semantic_query is not None

    inverse = analyze("Which companies does Tesla own?")
    assert inverse.predicate_hint == "owned_by"
    assert inverse.semantic_query is not None
    assert inverse.semantic_query.unknown_position == "subject"

    founded_followup = analyze("What else did Elon Musk found?")
    assert founded_followup.subject == "Elon Musk"
    assert founded_followup.predicate_hint == "founded"
    assert founded_followup.semantic_query is not None

    routed = route("How is Starlink connected to Falcon 9?")
    assert routed.intent == "connection_path"
    assert routed.subject == "Starlink"
    assert routed.secondary == "Falcon 9"

    comparative = route("What do SpaceX and Blue Origin have in common?")
    assert comparative.intent == "entity_relation"
    assert comparative.subject == "SpaceX"
    assert comparative.secondary == "Blue Origin"


def test_russian_controlled_patterns_parse_to_semantic_queries():
    expectations = {
        "что такое SpaceX": ("SpaceX", None, None, "relation", "definition"),
        "кто основал Blue Origin": ("Blue Origin", None, "founded_by", "object", "lookup"),
        "расскажи про Elon Musk": ("Elon Musk", None, None, "relation", "open_synthesis"),
        "чем занимается Tesla": ("Tesla", None, "produces", "object", "lookup"),
        "кому принадлежит Starlink": ("Starlink", None, "owned_by", "object", "lookup"),
    }

    for question, expected in expectations.items():
        sq = parse_semantic_query(question)
        assert (
            sq.entity_a,
            sq.entity_b,
            sq.relation_intent,
            sq.unknown_position,
            sq.query_type,
        ) == expected
        assert sq.confidence >= 0.75


@pytest.mark.parametrize(
    "question,expected_entity",
    [
        ("Explain SpaceX in simple terms.", "SpaceX"),
        ("Explain Blue Origin for beginners.", "Blue Origin"),
        ("Describe Tesla in plain English.", "Tesla"),
        ("Summarize Starlink briefly.", "Starlink"),
    ],
)
def test_open_synthesis_style_tail_routes_to_known_entity(question, expected_entity):
    sq = parse_semantic_query(question)

    assert sq.entity_a == expected_entity
    assert sq.entity_b is None
    assert sq.relation_intent is None
    assert sq.unknown_position == "relation"
    assert sq.query_type == "open_synthesis"
    assert sq.confidence >= 0.75


def test_low_confidence_question_routes_to_question_not_understood():
    routed = route("flarble norp zzz?")
    assert routed.intent == "unknown_or_unsupported"
    assert routed.notes == "question not understood"


def test_partial_title_entity_match_does_not_override_full_subject_fallback():
    sq = parse_semantic_query("What is Rocket Science Games?")
    assert sq.confidence < 0.75
    assert sq.entity_a is None


def test_cli_can_show_semantic_query():
    proc = subprocess.run(
        [
            sys.executable,
            str(_CLI),
            "Who founded SpaceX?",
            "--show-semantic-query",
        ],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    assert "SemanticQuery:" in proc.stdout
    assert '"relation_intent": "founded_by"' in proc.stdout
    assert "Decision: answer." in proc.stdout

    json_text = proc.stdout.split("SemanticQuery:", 1)[1].split("\n\n", 1)[0].strip()
    assert json.loads(json_text)["entity_a"] == "SpaceX"
