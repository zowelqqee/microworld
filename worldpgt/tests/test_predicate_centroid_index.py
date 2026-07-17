"""Calibration guardrails for the phrase-centroid predicate fallback.

These tests pin the conservative decision behavior of
``worldpgt.knowledge.predicate_centroid_index``: structural paraphrase forms
the index exists to catch must match, and non-relational / out-of-schema
question shapes must abstain.  A failure here means the threshold/margin or
the example-phrase table changed the false-positive surface — recalibrate
before shipping.

Marked slow: the first build loads GloVe (cached to worldpgt/artifacts
afterwards).
"""

from __future__ import annotations

import pytest

from worldpgt.knowledge.predicate_centroid_index import (
    AGENT_PREDICATES,
    get_default_centroid_index,
)


@pytest.fixture(scope="module")
def index():
    return get_default_centroid_index()


# Structural paraphrase forms (subject already stripped by the caller).
_MUST_MATCH = [
    ("Which manufacturer made X?", "X", None, "product_of"),
    ("Where does X maintain its headquarters?", "X", False, "headquartered_in"),
]

# Non-relational or out-of-schema shapes: a confident match here would be a
# false positive that could silently answer the wrong relation.
_MUST_ABSTAIN = [
    "What is X?",
    "Is X a company?",
    "Tell me about X",
    "Why does X develop Y?",
    "What does X have in common with Y?",
    "When was X founded?",
    "Who acquired X?",
    "What products does X sell?",
    "How many employees does X have?",
    "When did X release Y?",
    "Is X better than Y?",
    "What happened to X?",
    "Where can I download X?",
    "Who is the CEO of X?",
    "What year did X start?",
]


@pytest.mark.parametrize("question, subject, agent_shape, expected", _MUST_MATCH)
def test_structural_paraphrase_forms_match(index, question, subject, agent_shape, expected):
    allowed = None
    if agent_shape is True:
        allowed = AGENT_PREDICATES
    predicate, similarity = index.find_predicate(
        question, subject_span=subject, allowed=allowed,
    )
    assert predicate == expected, (
        f"{question!r}: expected {expected}, got {predicate} (sim={similarity:.3f})"
    )


@pytest.mark.parametrize("question", _MUST_ABSTAIN)
def test_non_relational_shapes_abstain(index, question):
    predicate, similarity = index.find_predicate(question, subject_span="X")
    assert predicate is None, (
        f"{question!r}: false positive {predicate} (sim={similarity:.3f}) — "
        "the fallback must abstain (audit) on non-relational shapes"
    )


def test_agent_restriction_rejects_out_of_set_winner(index):
    # "When was X founded?" ranks founded_by first but below the margin gate;
    # with an agent restriction the decision must still abstain, never promote
    # a runner-up past the ambiguity check.
    predicate, _ = index.find_predicate(
        "When was X founded?", subject_span="X", allowed=AGENT_PREDICATES,
    )
    assert predicate is None
