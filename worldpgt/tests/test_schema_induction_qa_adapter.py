"""QA adapter answers over generated/promoted schema, audits when unsupported,
and always carries a source trace."""

from __future__ import annotations

import pytest

from worldpgt.schema_induction.promotion_gates import GateConfig
from worldpgt.schema_induction.run_schema_induction import run_induction
from worldpgt.schema_induction.schema_qa_adapter import SchemaQAAdapter

_VISA_DOCS = [
    {"doc_id": "d1", "title": "D7", "url": "",
     "text": "Portugal D7 visa requires proof of passive income. "
             "Portugal D7 visa requires accommodation."},
    {"doc_id": "d2", "title": "NLV", "url": "",
     "text": "Spain non-lucrative visa prohibits work. "
             "Spain non-lucrative visa requires proof of financial means. "
             "Spain non-lucrative visa prohibits any local employment."},
    {"doc_id": "d3", "title": "DNV", "url": "",
     "text": "Digital nomad visa allows remote work under conditions."},
]

_ANIMAL_DOCS = [
    {"doc_id": "d4", "title": "Giraffe", "url": "",
     "text": "Giraffes move seasonally in search of food and water. "
             "Some giraffe populations shift ranges during dry seasons."},
    {"doc_id": "d5", "title": "Wildebeest", "url": "",
     "text": "Wildebeest migrate toward areas with fresh grass. "
             "Seasonal movement depends on rainfall and forage availability."},
]


@pytest.fixture(scope="module")
def visa_adapter():
    result = run_induction(_VISA_DOCS, GateConfig(min_evidence=2, min_sources=1))
    return SchemaQAAdapter.from_result(result, allow_generated=True)


@pytest.fixture(scope="module")
def animal_adapter():
    result = run_induction(_ANIMAL_DOCS, GateConfig(min_evidence=2, min_sources=1))
    return SchemaQAAdapter.from_result(result, allow_generated=True)


def test_visa_requirement_lookup(visa_adapter):
    ans = visa_adapter.answer("What does Portugal D7 visa require?")
    assert ans.decision == "answer"
    assert "proof of passive income" in ans.text
    assert "accommodation" in ans.text
    assert ans.sources  # source trace present


def test_visa_prohibition_lookup(visa_adapter):
    ans = visa_adapter.answer("What does Spain non-lucrative visa prohibit?")
    assert ans.decision == "answer"
    assert "work" in ans.text


def test_animal_cause_lookup(animal_adapter):
    ans = animal_adapter.answer("Why do giraffes move seasonally?")
    assert ans.decision == "answer"
    assert "food" in ans.text and "water" in ans.text
    assert ans.sources


def test_animal_destination_lookup(animal_adapter):
    ans = animal_adapter.answer("Куда мигрируют wildebeest?")
    assert ans.decision == "answer"
    assert "fresh grass" in ans.text


def test_russian_requirement_query(visa_adapter):
    ans = visa_adapter.answer("Что нужно для Portugal D7 visa?")
    assert ans.decision == "answer"
    assert "proof of passive income" in ans.text


def test_open_synthesis(visa_adapter):
    ans = visa_adapter.answer("Tell me about Digital nomad visa.")
    assert ans.decision == "answer"
    assert "Digital nomad visa" in ans.text
    assert "remote work" in ans.text
    assert ans.sources


def test_missing_relation_audits(visa_adapter):
    # No "founded by" family exists for visas -> must audit, not hallucinate.
    ans = visa_adapter.answer("Who founded Portugal D7 visa?")
    assert ans.decision == "audit"
    assert ans.tier == "UNKNOWN"


def test_default_prefers_promoted_over_generated():
    # Without allow_generated, a generated-only family (allows: 1 evidence)
    # must audit rather than answer.
    result = run_induction(_VISA_DOCS, GateConfig(min_evidence=2, min_sources=1))
    strict = SchemaQAAdapter.from_result(result, allow_generated=False)
    ans = strict.answer("What does Digital nomad visa allow?")
    assert ans.decision == "audit"


def test_every_answer_has_source_trace(visa_adapter):
    ans = visa_adapter.answer("What does Portugal D7 visa require?")
    assert ans.evidence
    for ev in ans.evidence:
        assert ev["sentence_id"]
        assert ev["sentence"]
