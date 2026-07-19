"""Tests for the isolated constrained-creative experiment.

Covers the fact selector, the template generator, and — most importantly — the
shared post-hoc verifier, which must actually detect a missing fact, a
mis-attached (unfaithful) fact, and hallucinated extra content, so it is a real
measurement instrument rather than a rubber stamp.
"""

from __future__ import annotations

from worldpgt.reasoning.constrained_creative_v1 import (
    ConstraintSpec,
    Fact,
    generate_constrained,
    proxy_fluency,
    select_facts,
    verify,
)


def _rel(subject, predicate, obj):
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "evidence_id": f"e:{subject}|{predicate}|{obj}".lower(),
    }


OVERLAY = [
    _rel("SpaceX", "founded", "2002"),
    _rel("SpaceX", "develops", "rockets"),
    _rel("SpaceX", "located_in", "Hawthorne"),
    _rel("SpaceX", "develops", "rockets"),  # duplicate, must be deduped
    _rel("Tesla", "produces", "electric cars"),
]


# --- selector --------------------------------------------------------------- #

def test_select_facts_picks_subject_and_dedupes():
    spec = select_facts(OVERLAY, "SpaceX", n=5)
    assert spec.subject == "SpaceX"
    pairs = {(f.predicate, f.object) for f in spec.facts}
    assert ("develops", "rockets") in pairs
    assert len(spec.facts) == 3  # duplicate collapsed


def test_select_facts_respects_n():
    spec = select_facts(OVERLAY, "SpaceX", n=2)
    assert spec.n == 2


# --- generator -------------------------------------------------------------- #

def test_generator_includes_every_fact():
    spec = select_facts(OVERLAY, "SpaceX", n=3)
    text = generate_constrained(spec)
    report = verify(text, spec)
    assert report.inclusion_rate == 1.0            # all facts present
    assert report.fidelity_rate == 1.0             # all correctly attached
    assert report.hallucination_token_rate == 0.0  # nothing extra by construction


def test_generator_empty_spec():
    assert generate_constrained(ConstraintSpec("X", ())) == ""


# --- verifier detects a MISSING fact ---------------------------------------- #

def test_verify_flags_missing_fact():
    spec = ConstraintSpec("SpaceX", (Fact("develops", "rockets"), Fact("located_in", "Hawthorne")))
    text = "SpaceX develops rockets."  # omits the location fact
    report = verify(text, spec)
    assert report.inclusion_rate == 0.5
    assert any("Hawthorne" in m for m in report.missing)


# --- verifier detects HALLUCINATED content ---------------------------------- #

def test_verify_flags_hallucinated_content():
    spec = ConstraintSpec("SpaceX", (Fact("develops", "rockets"),))
    # An LLM-style embellishment adding facts not in the spec.
    text = "SpaceX develops rockets and was famously founded by Elon Musk in California."
    report = verify(text, spec)
    assert report.inclusion_rate == 1.0
    # 'elon', 'musk', 'california', 'famously' are extra content not in the spec.
    assert report.hallucination_token_rate > 0.0
    assert "musk" in report.extra_content_tokens
    assert "california" in report.extra_content_tokens


# --- verifier detects an UNFAITHFUL (mis-attached) fact ---------------------- #

def test_verify_flags_unfaithful_attachment():
    spec = ConstraintSpec("SpaceX", (Fact("develops", "rockets"),))
    # The object token appears, but in a sentence about a different subject with no
    # link back to SpaceX -> included but not faithful.
    text = "Aerospace is a broad field. Many firms build rockets worldwide."
    report = verify(text, spec)
    assert report.inclusion_rate == 1.0     # 'rockets' present
    assert report.fidelity_rate == 0.0      # not attached to SpaceX
    assert report.unfaithful


# --- symmetry: same verifier scores arbitrary (LLM-style) text -------------- #

def test_verifier_is_text_agnostic():
    spec = select_facts(OVERLAY, "SpaceX", n=3)
    llm_style = (
        "Founded in 2002, SpaceX develops rockets from its base in Hawthorne, "
        "and it has revolutionized spaceflight forever."
    )
    report = verify(llm_style, spec)
    assert report.inclusion_rate == 1.0
    # 'revolutionized', 'spaceflight', 'forever', 'base' are extra -> hallucination.
    assert report.hallucination_token_rate > 0.0


# --- proxy fluency ---------------------------------------------------------- #

def test_proxy_fluency_none_without_corpus():
    assert proxy_fluency("SpaceX develops rockets.", set()) is None


def test_proxy_fluency_counts_attested_windows():
    attested = {("spacex", "develops", "rockets")}
    val = proxy_fluency("SpaceX develops rockets today.", attested)
    assert val is not None and 0.0 < val <= 1.0
