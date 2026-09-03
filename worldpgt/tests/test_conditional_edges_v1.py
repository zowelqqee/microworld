"""Tests for the conditional-edge / class-subject / citation-normalizer stack.

Covers the five integration pieces validated retrospectively in the legal
pilots (artifacts/legal_domain_pilot_v1, _v2, conditional_edge_v1):
  #3 dynamic citation normalization,
  #1 conditional-edge schema + structured validation,
  #2 class-subject nodes,
  #4 structure-driven conditional rendering + mandatory-guard invariant,
  #5 disjunctive consequences.
"""

from pathlib import Path

import pytest

from worldpgt.relation_extraction_v2.citation_normalizer import (
    is_bare_reference,
    normalize_citation_surface,
)
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.relation_extraction_v2.node_quality_filter import classify_subject_node
from worldpgt.relation_extraction_v2.relation_candidate_validator import validate_candidates
from worldpgt.relation_extraction_v2.types import ConditionClause, ExtractedRelationCandidate
from worldpgt.reasoning.answer_behavior import _edge_from_item
from worldpgt.reasoning.answer_plan_renderer import _render_conditional_claim


def _index(tmp_path: Path, *labels: str) -> EntitySurfaceIndex:
    overlay = tmp_path / "overlay.json"
    if labels:
        items = ",".join(f'{{"overlay_type":"overlay_entity","label":"{l}"}}' for l in labels)
        overlay.write_text(f"[{items}]", encoding="utf-8")
    absent = tmp_path / "absent.json"
    return EntitySurfaceIndex(overlay if labels else absent, absent, tmp_path / "a2.json")


def _cand(**kw) -> ExtractedRelationCandidate:
    base = dict(
        confidence="high", stability="semi_stable", risk="medium", source_title="",
        source_url="", retrieved_at="", raw_text_sha256="", pattern_id="p",
        directionality="forward", requires_review=True, safe_for_overlay_delta=False,
    )
    base.update(kw)
    return ExtractedRelationCandidate(**base)


# --- #3 citation normalizer -------------------------------------------------

def test_citation_normalizer_recovers_elided_governing_word():
    ev = "a right of priority under section 119, 365(a), 365(b), or 386(b)"
    assert normalize_citation_surface("365(a)", ev) == ("section 365(a)", True)
    assert normalize_citation_surface("386(b)", ev) == ("section 386(b)", True)


def test_citation_normalizer_is_noop_off_target():
    ev = "under section 119, 365(a)"
    # already governed
    assert normalize_citation_surface("section 119", ev) == ("section 119", False)
    # not a bare reference in any governed list
    assert normalize_citation_surface("a claimed invention", ev)[1] is False
    assert normalize_citation_surface("subsection (a)(1)", ev)[1] is False
    assert not is_bare_reference("Whoever")


def test_citation_normalizer_dynamic_governing_word():
    # The governing word is read from the text, not assumed to be "section".
    ev = "as provided in paragraph 2, 3, or 4 of this subsection"
    assert normalize_citation_surface("3", ev) == ("paragraph 3", True)


# --- #2 class-subject classification ---------------------------------------

@pytest.mark.parametrize("surface,expected", [
    ("Whoever knowingly and willfully deposits any threat upon the President", "class_subject"),
    ("any invention made, used or sold in outer space on a space object", "class_subject"),
    ("a disclosure made 1 year or less before the effective filing date", "class_subject"),
    ("the individual", "entity"),
    ("a novel architecture", "entity"),
    ("SpaceX", "entity"),
    ("this paper", "entity"),
    # A multi-word proper name is not misread as a class description merely
    # for being long, even without a relative clause.
    ("the United States Court of Appeals for the Federal Circuit", "entity"),
])
def test_classify_subject_node(surface, expected):
    assert classify_subject_node(surface) == expected


# --- #1 structured validation ----------------------------------------------

def test_structured_conditional_edge_admitted_as_proposal(tmp_path):
    ev = ("A disclosure shall not be prior art to a claimed invention under subsection (a)(2) "
          "if the subject matter disclosed was obtained directly or indirectly from the inventor")
    cand = _cand(
        id="c", subject="a disclosure", relation="is_prior_art_to", object="a claimed invention",
        evidence_sentence=ev, polarity="negate", subject_kind="class_subject",
        conditions=[ConditionClause("the determination is under subsection (a)(2)",
                                    "under subsection (a)(2)", "scope"),
                    ConditionClause("the subject matter disclosed was obtained from the inventor",
                                    "the subject matter disclosed was obtained directly or indirectly from the inventor")],
    )
    safe, quar, _, _ = validate_candidates([cand], _index(tmp_path), [])
    assert [c.id for c in safe] == ["c"]
    # proposal-only: never auto-admitted
    assert safe[0].safe_for_overlay_delta is False
    assert safe[0].requires_review is True
    assert quar == []


def test_welded_predicate_quarantined(tmp_path):
    cand = _cand(
        id="w", subject="a person", object="a patent",
        relation="states that if the invention was patented it shall not be entitled",
        evidence_sentence="a person shall be entitled unless the invention was patented",
        polarity="negate", conditions=[ConditionClause("x", "a person shall be entitled")],
    )
    safe, quar, _, _ = validate_candidates([cand], _index(tmp_path), [])
    assert safe == []
    assert "welded_predicate" in quar[0].reason


def test_ungrounded_clause_quarantined(tmp_path):
    cand = _cand(
        id="u", subject="a person", relation="entitled_to", object="a patent",
        evidence_sentence="a person shall be entitled to a patent unless the invention was patented",
        polarity="negate",
        conditions=[ConditionClause("made-up", "THIS SPAN IS NOT IN THE EVIDENCE")],
    )
    safe, quar, _, _ = validate_candidates([cand], _index(tmp_path), [])
    assert safe == []
    assert "unverified_clause_span" in quar[0].reason


def test_class_subject_must_be_grounded(tmp_path):
    cand = _cand(
        id="cs", subject="a fabricated class not present", relation="is_punishable_by",
        object="a fine", evidence_sentence="Whoever does X shall be fined.",
        subject_kind="class_subject", conditions=[],
    )
    safe, quar, _, _ = validate_candidates([cand], _index(tmp_path), [])
    assert safe == []
    assert "class_subject_span_not_literal" in quar[0].reason


def test_simple_edge_path_unchanged(tmp_path):
    # A plain edge with an allowed relation and clean entities still auto-admits.
    cand = _cand(
        id="s", subject="SpaceX", relation="develops", object="Rockets",
        evidence_sentence="SpaceX develops Rockets.",
    )
    safe, quar, _, _ = validate_candidates([cand], _index(tmp_path, "SpaceX", "Rockets"), [])
    assert [c.id for c in safe] == ["s"]
    assert safe[0].safe_for_overlay_delta is True


# --- #5 disjunctive consequence --------------------------------------------

def test_disjunctive_object_validated_and_rendered(tmp_path):
    ev = ("subject matter disclosed and a claimed invention shall be deemed to have been owned "
          "by the same person or subject to an obligation of assignment to the same person")
    cand = _cand(
        id="d", subject="subject matter disclosed", relation="deemed_commonly_owned_with",
        object="a claimed invention", evidence_sentence=ev,
        object_alternatives=[ConditionClause(
            "subject to an obligation of assignment to the same person",
            "subject to an obligation of assignment to the same person")],
    )
    safe, quar, _, _ = validate_candidates([cand], _index(tmp_path), [])
    assert [c.id for c in safe] == ["d"]


# --- #4 rendering + mandatory-guard invariant ------------------------------

def test_conditional_render_contains_every_guard():
    item = {
        "overlay_type": "overlay_relation",
        "subject": "any invention made in outer space under US control",
        "predicate": "considered_made_used_or_sold_in", "object": "the United States",
        "evidence_text": "Any invention ... within the United States ...", "source_url": "u",
        "subject_kind": "class_subject",
        "conditions": [{"text": "this title governs the determination",
                        "evidence_span": "for the purposes of this title", "kind": "scope"}],
        "exceptions": [
            {"text": "the object is covered by an international agreement", "evidence_span": "except ..."},
            {"text": "the object is on a foreign registry", "evidence_span": "or with respect ..."},
        ],
    }
    edge = _edge_from_item(item)
    assert edge.is_conditional()
    text = _render_conditional_claim(edge)
    assert "except where" in text
    assert "international agreement" in text and "foreign registry" in text
    assert "For purposes of" in text


def test_negate_polarity_surfaces_as_negation():
    item = {
        "overlay_type": "overlay_relation", "subject": "a person", "predicate": "entitled_to",
        "object": "a patent", "evidence_text": "...", "source_url": "u", "polarity": "negate",
        "conditions": [{"text": "the invention was patented before filing", "evidence_span": "x"}],
    }
    text = _render_conditional_claim(_edge_from_item(item))
    assert "is not entitled" in text
    assert "provided that" in text


def test_mandatory_guard_invariant_raises_if_guard_dropped(monkeypatch):
    import worldpgt.reasoning.answer_plan_renderer as R

    class _Edge:
        subject = "a person"; predicate = "entitled_to"; object = "a patent"
        polarity = "affirm"; object_alternatives = (); exceptions = ()
        conditions = (("UNRENDERABLE_GUARD_TOKEN", "span", "factual"),)
        def is_conditional(self):  # noqa: D401
            return True

    monkeypatch.setattr(R, "_join_list", lambda items: "")  # simulate a renderer that drops it
    with pytest.raises(AssertionError):
        R._render_conditional_claim(_Edge())


# --- #4b guard-unaware surfaces must not assert a conditional rule ---------

def test_entity_renderer_suppresses_guarded_relations():
    """A conditional rule must never surface as a bare "linked via" claim.

    Discovered in the legal QA study: the entity-QA renderer surfaced a
    polarity-negated edge ("a disclosure shall NOT be prior art ... if C") as
    "a disclosure is linked to a claimed invention via is_prior_art_to",
    inverting the legal meaning. That surface cannot express a guard, so a
    guard-bearing relation must be dropped from it entirely.
    """
    from worldpgt.entity_qa.entity_answer_renderer import _carries_guards, _drop_guarded

    negated = {"subject": "a disclosure", "predicate": "is_prior_art_to",
               "object": "a claimed invention", "polarity": "negate"}
    conditioned = {"subject": "x", "predicate": "p", "object": "y",
                   "conditions": [{"text": "c", "evidence_span": "c"}]}
    excepted = {"subject": "x", "predicate": "p", "object": "y",
                "exceptions": [{"text": "e", "evidence_span": "e"}]}
    plain = {"subject": "SpaceX", "predicate": "develops", "object": "rockets"}

    assert _carries_guards(negated)
    assert _carries_guards(conditioned)
    assert _carries_guards(excepted)
    assert not _carries_guards(plain)
    assert _drop_guarded([negated, conditioned, excepted, plain]) == [plain]
