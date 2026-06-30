"""End-to-end: a cold-start domain overlay answers via the existing QA stack.

Proves the system answers domain questions (visas) with NO prior entity list and
NO hand-written domain predicates — the overlay was bootstrapped from raw text.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.schema_induction.domain_overlay_builder import build_domain_overlay

_DOCS = [
    {"doc_id": "d1", "title": "O-1A", "url": "", "text":
        "The O-1A visa is a nonimmigrant visa for individuals with extraordinary "
        "ability in the sciences, education, business, or athletics. The O-1A visa "
        "requires evidence of sustained national or international acclaim. The O-1A "
        "visa allows employment with the petitioning employer. The O-1A visa "
        "prohibits self-petition without a sponsor."},
    {"doc_id": "d2", "title": "O-1B", "url": "", "text":
        "The O-1B visa is a nonimmigrant visa for individuals with extraordinary "
        "ability in the arts. The O-1B visa requires evidence of distinction in the arts."},
]


@pytest.fixture(scope="module")
def overlay_path(tmp_path_factory):
    built = build_domain_overlay(_DOCS, domain="o1a", min_evidence=1, min_sources=1)
    p = tmp_path_factory.mktemp("domain") / "o1a.json"
    p.write_text(json.dumps(built["overlay"], ensure_ascii=False), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def orch(overlay_path):
    return AnswerOrchestrator("pump-dry-run", overlay_path=overlay_path)


def _text(ans) -> str:
    return (getattr(ans, "answer_text", None) or getattr(ans, "text", "") or "")


def test_definition_question(orch):
    ans = orch.answer("What is O-1A visa?")
    assert ans.decision == "answer"
    assert "visa" in _text(ans).lower()


def test_requires_question(orch):
    ans = orch.answer("What does O-1A require?")
    assert ans.decision == "answer"
    assert "acclaim" in _text(ans).lower()


def test_allows_question(orch):
    ans = orch.answer("What does O-1A allow?")
    assert ans.decision == "answer"
    assert "employ" in _text(ans).lower()


def test_eligibility_question_routes_to_definition(orch):
    ans = orch.answer("Who qualifies for O-1A?")
    assert ans.decision == "answer"
    assert "extraordinary ability" in _text(ans).lower()


def test_synthesis_question(orch):
    ans = orch.answer("Tell me about O-1A.")
    assert ans.decision == "answer"
    assert "O-1A" in _text(ans)


def test_unsupported_relation_audits(orch):
    # The corpus says nothing about who FOUNDED a visa — must audit, not invent.
    ans = orch.answer("Who founded O-1A?")
    assert ans.decision == "audit"
