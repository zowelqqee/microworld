"""Tests for GeneralizedQuestionAnalyzer v1 — new question phrasings.

Covers the three new intent families introduced in v1:
  - classify_context / define_sense  via "SENTENCE. What does TERM mean?"
  - classify_context / define_sense  via "SENTENCE. What kind of TERM is it?"
  - classify_context / define_sense  via "Is SUBJECT [prep CUE] an A or B?"
  - explain_cue                       via "Why does/do CUE point to TERM as SENSE"
  - distinguish_senses                via "Compare X with Y."

All previously passing tests in test_answer_planner_v1.py are unaffected
(confirmed by running the full test suite after every change).
"""

from __future__ import annotations

import pytest

from worldpgt.qa.question_analyzer import analyze


# ---------------------------------------------------------------------------
# Task 1 — classify_context: "Is TERM with/near CUE an A or B?" phrasings
# ---------------------------------------------------------------------------

class TestIsAOrBClassifyContext:

    def test_bat_with_wings_routes_define_sense_animal(self):
        r = analyze("Is a bat with wings an animal or sports equipment?")
        # Unambiguous context (wings → animal) → routed as define_sense
        assert r.intent == "define_sense"
        assert r.term == "bat"
        assert r.target_sense == "animal"
        assert not r.underconstrained

    def test_bat_near_pitcher_routes_define_sense_sports(self):
        r = analyze("Is the bat near the pitcher an animal or sports equipment?")
        assert r.intent == "define_sense"
        assert r.term == "bat"
        assert r.target_sense == "sports_equipment"
        assert not r.underconstrained


# ---------------------------------------------------------------------------
# Task 1 — classify_context: "SENTENCE. What kind of TERM is it?"
# ---------------------------------------------------------------------------

class TestSentenceWhatKind:

    def test_seal_swimming_coast_animal(self):
        r = analyze("The seal was swimming near the coast. What kind of seal is it?")
        assert r.intent == "define_sense"
        assert r.term == "seal"
        assert r.target_sense == "animal"
        assert not r.underconstrained

    def test_seal_wax_envelope_closure_stamp(self):
        r = analyze("The seal was pressed into wax on the envelope. What kind of seal is it?")
        assert r.intent == "define_sense"
        assert r.term == "seal"
        assert r.target_sense == "closure_stamp"
        assert not r.underconstrained


# ---------------------------------------------------------------------------
# Task 1 — classify_context: "SENTENCE. What does TERM mean?"
# ---------------------------------------------------------------------------

class TestSentenceWhatDoes:

    def test_crane_hook_load_machine(self):
        r = analyze("The crane had a hook and lifted a load. What does crane mean?")
        assert r.intent == "define_sense"
        assert r.term == "crane"
        assert r.target_sense == "machine"
        assert not r.underconstrained

    def test_crane_wetland_reeds_bird(self):
        r = analyze("The crane stood in the wetland near reeds. What does crane mean?")
        assert r.intent == "define_sense"
        assert r.term == "crane"
        assert r.target_sense == "bird"
        assert not r.underconstrained

    def test_rock_band_stage_music(self):
        r = analyze("The band played rock on stage. What does rock mean?")
        assert r.intent == "define_sense"
        assert r.term == "rock"
        assert r.target_sense == "music"
        assert not r.underconstrained

    def test_rock_cliff_stone(self):
        r = analyze("The rock rolled down near the cliff. What does rock mean?")
        assert r.intent == "define_sense"
        assert r.term == "rock"
        assert r.target_sense == "stone"
        assert not r.underconstrained

    def test_spring_flowers_thaw_season(self):
        r = analyze("The spring came after winter with flowers and thaw. What does spring mean?")
        assert r.intent == "define_sense"
        assert r.term == "spring"
        assert r.target_sense == "season"
        assert not r.underconstrained

    def test_spring_device_compressed_coil(self):
        r = analyze("The spring inside the device compressed and snapped back. What does spring mean?")
        assert r.intent == "define_sense"
        assert r.term == "spring"
        assert r.target_sense == "coil"
        assert not r.underconstrained


# ---------------------------------------------------------------------------
# Task 2 — distinguish_senses: "Compare X with Y." phrasings
# ---------------------------------------------------------------------------

class TestCompareSensesWithSeparator:

    def test_compare_crane_bird_with_construction(self):
        r = analyze("Compare a crane bird with a construction crane.")
        assert r.intent == "distinguish_senses"
        assert r.term == "crane"
        assert r.target_sense == "bird"
        assert r.second_sense == "machine"
        assert not r.underconstrained

    def test_compare_rock_music_with_rock_stone(self):
        r = analyze("Compare rock music with a rock stone.")
        assert r.intent == "distinguish_senses"
        assert r.term == "rock"
        assert r.target_sense == "music"
        assert r.second_sense == "stone"
        assert not r.underconstrained

    def test_compare_spring_as_season_and_coil(self):
        r = analyze("Compare spring as a season and spring as a coil.")
        assert r.intent == "distinguish_senses"
        assert r.term == "spring"
        assert r.target_sense == "season"
        assert r.second_sense == "coil"
        assert not r.underconstrained


# ---------------------------------------------------------------------------
# Task 3 — explain_cue: "Why does/do CUE point to TERM as SENSE"
# ---------------------------------------------------------------------------

class TestPointToExplainCue:

    def test_account_points_to_bank_financial(self):
        r = analyze("Why does an account point to bank as a financial institution?")
        assert r.intent == "explain_cue"
        assert r.term == "bank"
        assert r.target_sense == "financial_institution"
        assert r.cues_in_question == ["account"]
        assert not r.underconstrained

    def test_current_points_to_bank_river(self):
        r = analyze("Why does current point to bank as a river edge?")
        assert r.intent == "explain_cue"
        assert r.term == "bank"
        assert r.target_sense == "river_edge"
        assert r.cues_in_question == ["current"]
        assert not r.underconstrained

    def test_wings_point_to_bat_animal(self):
        r = analyze("Why do wings point to bat as an animal?")
        assert r.intent == "explain_cue"
        assert r.term == "bat"
        assert r.target_sense == "animal"
        assert r.cues_in_question == ["wings"]
        assert not r.underconstrained

    def test_batter_points_to_bat_sports(self):
        r = analyze("Why does batter point to bat as sports equipment?")
        assert r.intent == "explain_cue"
        assert r.term == "bat"
        assert r.target_sense == "sports_equipment"
        assert r.cues_in_question == ["batter"]
        assert not r.underconstrained


# ---------------------------------------------------------------------------
# Task 4 — conflict safety: sentences with conflicting cues must remain audit
# ---------------------------------------------------------------------------

class TestConflictSafety:

    def test_bat_flew_near_pitcher_is_ambiguous(self):
        """Flying cue (animal) conflicts with pitcher cue (sports_equipment)."""
        r = analyze("The bat flew near the pitcher. What does bat mean?")
        # Must route as classify_context so planner can detect the conflict
        assert r.intent == "classify_context"
        assert r.term == "bat"
        assert r.inline_context == "The bat flew near the pitcher"
        # underconstrained=False because the term is known; planner decides to audit

    def test_crane_wing_lifted_load_is_ambiguous(self):
        """Wing cue (bird) conflicts with lifted/load cues (machine)."""
        r = analyze("The crane wing lifted the load. What does crane mean?")
        assert r.intent == "classify_context"
        assert r.term == "crane"
        assert r.inline_context == "The crane wing lifted the load"

    def test_standalone_what_does_remains_unknown(self):
        """Bare 'What does X mean?' without context stays unknown_or_ambiguous."""
        for q in [
            "What does seal mean?",
            "What does crane mean?",
            "What does spring mean?",
        ]:
            r = analyze(q)
            assert r.intent == "unknown_or_ambiguous", f"expected audit for: {q!r}"
            assert r.underconstrained


# ---------------------------------------------------------------------------
# Task 1 extra — inline_context is correctly extracted
# ---------------------------------------------------------------------------

class TestInlineContextExtraction:

    def test_sentence_what_does_sets_inline_context(self):
        r = analyze("The crane had a hook and lifted a load. What does crane mean?")
        assert r.inline_context == "The crane had a hook and lifted a load"

    def test_sentence_what_kind_sets_inline_context(self):
        r = analyze("The seal was swimming near the coast. What kind of seal is it?")
        assert r.inline_context == "The seal was swimming near the coast"

    def test_is_a_or_b_sets_inline_context(self):
        r = analyze("Is a bat with wings an animal or sports equipment?")
        assert r.inline_context == "a bat with wings"

    def test_is_a_or_b_near_sets_inline_context(self):
        r = analyze("Is the bat near the pitcher an animal or sports equipment?")
        assert r.inline_context == "the bat near the pitcher"
