"""Tests for the generic SVO fallback (2026-07-07).

Motivation: a fresh-domain probe (Fields Medal test corpus, not part of this
repo) showed the fixed ``_TRIGGERS`` list in raw_claim_extractor.py silently
skips any sentence whose verb isn't on that list -- "X won Y", "X worked at
Y", "X died in Y" produced zero claims even though they're simple SVO
sentences. This closes that gap on both sides of the pipeline:
extraction (raw_claim_extractor.py) and question understanding
(query_compiler.py) -- see both modules' docstrings for the full story.
"""

from __future__ import annotations

from worldpgt.schema_induction.query_compiler import compile_query
from worldpgt.schema_induction.raw_claim_extractor import extract_claims_from_sentence
from worldpgt.schema_induction.types import RelationFamily, SentenceRecord


def _claim(text: str):
    sent = SentenceRecord(sentence_id="d1:s0", doc_id="d1", index=0, text=text)
    claims = extract_claims_from_sentence(sent)
    assert claims, f"no claim extracted from: {text}"
    return claims[0]


def test_unlisted_verb_is_extracted_via_spacy_svo_fallback():
    claim = _claim("Maryam Mirzakhani won the Fields Medal.")
    assert claim.subject == "Maryam Mirzakhani"
    assert claim.relation_surface == "win"
    assert claim.object == "Fields Medal"
    assert claim.extraction_method == "spacy_svo"


def test_prepositional_object_is_captured():
    claim = _claim("Maryam Mirzakhani worked at Stanford University.")
    assert claim.relation_surface == "work"
    assert "Stanford University" in claim.object


def test_fixed_trigger_still_wins_when_it_matches():
    """The regex trigger path must be unaffected -- no double-claim, no
    regression for sentences the existing list already handles."""
    claim = _claim("Portugal D7 visa requires proof of passive income.")
    assert claim.relation_surface == "requires"
    assert claim.extraction_method == "regex_trigger"


def test_no_claim_when_sentence_has_no_svo_shape():
    sent = SentenceRecord(sentence_id="d1:s1", doc_id="d1", index=1, text="Hello there.")
    assert extract_claims_from_sentence(sent) == []


def test_query_compiler_matches_question_verb_lemma_to_existing_family():
    families = [
        RelationFamily(
            family_id="fam1",
            canonical_label="win",
            surface_forms=("won",),
            roles=("subject", "object"),
            source_doc_count=1,
            evidence_count=1,
            confidence=0.5,
            promotion_status="generated",
            rejection_reason=None,
            frame_ids=(),
            example_claim_ids=(),
            role_type_profile={},
        )
    ]
    plan = compile_query("What did Maryam Mirzakhani win?", families, ["Maryam Mirzakhani"])
    assert plan.operation == "find_role"
    assert plan.family_label == "win"


def test_query_compiler_audits_when_no_family_matches_the_verb():
    plan = compile_query("What did Maryam Mirzakhani win?", [], ["Maryam Mirzakhani"])
    assert plan.operation == "audit"
    assert plan.reason == "no_relation_cue_matched"


def test_fixed_cue_still_takes_priority_over_verb_lemma_fallback():
    """founded/founded-by must keep working exactly as before -- the fallback
    only kicks in when the fixed cue table has nothing to say."""
    families = [
        RelationFamily(
            family_id="fam2",
            canonical_label="founded by",
            surface_forms=("was founded by",),
            roles=("subject", "agent"),
            source_doc_count=1,
            evidence_count=1,
            confidence=0.9,
            promotion_status="promoted",
            rejection_reason=None,
            frame_ids=(),
            example_claim_ids=(),
            role_type_profile={},
        )
    ]
    plan = compile_query("Who founded Tesla?", families, ["Tesla"])
    assert plan.family_label == "founded by"


_WIN_FAMILY = RelationFamily(
    family_id="fam3",
    canonical_label="win",
    surface_forms=("won",),
    roles=("subject", "object"),
    source_doc_count=1,
    evidence_count=3,
    confidence=0.5,
    promotion_status="generated",
    rejection_reason=None,
    frame_ids=(),
    example_claim_ids=(),
    role_type_profile={},
)


def test_reverse_direction_question_matches_object_and_returns_subject():
    """"Who won ENTITY?" -- entity is the grammatical object in the raw
    claims (subject=winner, object=prize) -- the plan must search by object
    and return the subject, not the other way around."""
    plan = compile_query(
        "Who won the Fields Medal?", [_WIN_FAMILY], ["Fields Medal", "Maryam Mirzakhani"]
    )
    assert plan.operation == "find_role"
    assert plan.match_role == "object"
    assert plan.target_role == "subject"


def test_forward_direction_question_still_matches_subject():
    """"What did ENTITY win?" -- entity is given as the subject; unaffected
    by the reverse-direction fallback."""
    plan = compile_query(
        "What did Maryam Mirzakhani win?", [_WIN_FAMILY], ["Fields Medal", "Maryam Mirzakhani"]
    )
    assert plan.match_role == "subject"
    assert plan.target_role == "object"


def test_irregular_verb_past_tense_still_reverses_correctly():
    """"won" doesn't contain its lemma "win" as a substring (unlike regular
    "-ed" verbs) -- this used to silently defeat the position-based reversal
    check because it searched for the lemma instead of the surface form."""
    plan = compile_query("Who won the Fields Medal?", [_WIN_FAMILY], ["Fields Medal"])
    assert plan.match_role == "object"


def test_russian_fixed_cue_direction_is_unaffected_by_reversal_heuristic():
    """Regression guard: Russian question word order ("Куда мигрируют X?")
    puts the subject after the verb, which the (English-parse-driven)
    direction-reversal heuristic must NOT apply to -- it's gated to the
    verb-lemma fallback only, never the pre-existing fixed cue table."""
    families = [
        RelationFamily(
            family_id="fam4",
            canonical_label="move/migrate",
            surface_forms=("migrate",),
            roles=("subject", "destination"),
            source_doc_count=1,
            evidence_count=1,
            confidence=0.9,
            promotion_status="promoted",
            rejection_reason=None,
            frame_ids=(),
            example_claim_ids=(),
            role_type_profile={},
        )
    ]
    plan = compile_query("Куда мигрируют wildebeest?", families, ["wildebeest"])
    assert plan.match_role == "subject"
    assert plan.target_role == "destination"
