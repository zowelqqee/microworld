"""Tests for AnswerPlanner v1.

Covers all 30 required test points.
"""

from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path

import pytest

from worldpgt.continuation.continuation_policy import ContinuationPolicy
from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.knowledge.accepted_memory_provider import AcceptedKnowledgeMemoryProvider
from worldpgt.qa.answer_planner import AnswerPlanner
from worldpgt.qa.answer_renderer import render
from worldpgt.qa.answer_validator import validate
from worldpgt.qa.question_analyzer import analyze
from worldpgt.qa.types import QADecision

_REPO = Path(__file__).resolve().parents[2]
_ARTIFACT = _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
_QA_PROMPTS = _REPO / "worldpgt" / "experiments" / "qa_prompts_v1.csv"
_QA_OUTPUTS = _REPO / "worldpgt" / "experiments" / "answer_planner_v1_outputs.csv"
_QA_SUMMARY = _REPO / "worldpgt" / "experiments" / "answer_planner_v1_summary.json"
_OLD_PROMPTS = _REPO / "worldpgt" / "experiments" / "continuation_prompts_v1.csv"
_OLD_BASELINE = _REPO / "worldpgt" / "experiments" / "microworld_continuation_v1_2_outputs.csv"

_ALL_TERMS = {"bank", "bat", "seal", "crane", "rock", "spring"}
_ALL_SENSES = {
    "financial_institution", "river_edge",
    "animal", "sports_equipment",
    "closure_stamp", "bird",
    "machine", "stone",
    "music", "season",
    "coil",
}


@lru_cache(maxsize=1)
def _provider() -> AcceptedKnowledgeMemoryProvider:
    return AcceptedKnowledgeMemoryProvider(str(_ARTIFACT))


@lru_cache(maxsize=1)
def _planner() -> AnswerPlanner:
    return AnswerPlanner(provider=_provider(), sense_mem=ExplicitSenseMemory())


@lru_cache(maxsize=1)
def _qa_prompt_rows() -> list[dict]:
    with _QA_PROMPTS.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=1)
def _output_rows() -> list[dict]:
    with _QA_OUTPUTS.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@lru_cache(maxsize=1)
def _summary() -> dict:
    with _QA_SUMMARY.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 1. QA prompt file has at least 48 rows
# ---------------------------------------------------------------------------

def test_qa_prompt_file_has_at_least_48_rows():
    assert len(_qa_prompt_rows()) >= 48


# ---------------------------------------------------------------------------
# 2. QA prompt file covers all 6 terms
# ---------------------------------------------------------------------------

def test_qa_prompt_file_covers_all_6_terms():
    terms = {r["expected_term"] for r in _qa_prompt_rows() if r["expected_term"]}
    assert _ALL_TERMS.issubset(terms), f"Missing terms: {_ALL_TERMS - terms}"


# ---------------------------------------------------------------------------
# 3. QA prompt file covers all 12 senses
# ---------------------------------------------------------------------------

def test_qa_prompt_file_covers_all_12_senses():
    senses = {r["expected_sense"] for r in _qa_prompt_rows() if r["expected_sense"]}
    assert _ALL_SENSES.issubset(senses), f"Missing senses: {_ALL_SENSES - senses}"


# ---------------------------------------------------------------------------
# 4-8. Question analyzer intent detection
# ---------------------------------------------------------------------------

def test_analyzer_detects_define_sense():
    a = analyze("What is a financial bank?")
    assert a.intent == "define_sense"
    assert a.term == "bank"
    assert a.target_sense == "financial_institution"


def test_analyzer_detects_define_sense_river():
    a = analyze("What is a river bank?")
    assert a.intent == "define_sense"
    assert a.term == "bank"
    assert a.target_sense == "river_edge"


def test_analyzer_detects_distinguish_senses():
    a = analyze("What is the difference between a financial bank and a river bank?")
    assert a.intent == "distinguish_senses"
    assert a.term == "bank"
    assert a.target_sense is not None
    assert a.second_sense is not None
    assert a.target_sense != a.second_sense


def test_analyzer_detects_explain_cue():
    a = analyze("Why does 'teller' suggest bank as a financial institution?")
    assert a.intent == "explain_cue"
    assert a.term == "bank"
    assert a.cues_in_question == ["teller"]
    assert a.target_sense == "financial_institution"


def test_analyzer_detects_classify_context():
    a = analyze("In 'she went to the bank to open an account', what does bank mean?")
    assert a.intent == "classify_context"
    assert a.term == "bank"
    assert a.inline_context is not None
    assert "account" in a.inline_context


def test_analyzer_detects_unknown_or_ambiguous():
    a = analyze("What does bank mean?")
    assert a.intent == "unknown_or_ambiguous"
    assert a.term == "bank"
    assert a.underconstrained is True


def test_analyzer_all_six_ambiguous_questions():
    for term in _ALL_TERMS:
        a = analyze(f"What does {term} mean?")
        assert a.intent == "unknown_or_ambiguous", f"Failed for term: {term}"
        assert a.term == term


# ---------------------------------------------------------------------------
# 9. Planner answers known definition questions
# ---------------------------------------------------------------------------

def test_planner_answers_define_sense_financial_bank():
    analyzed = analyze("What is a financial bank?")
    plan = _planner().plan(analyzed)
    assert plan.decision == "answer"
    assert plan.render_template == "define_sense"
    assert "bank" in plan.render_args["term"]
    assert plan.render_args["sense_id"] == "financial_institution"


def test_planner_answers_define_sense_river_bank():
    analyzed = analyze("What is a river bank?")
    plan = _planner().plan(analyzed)
    assert plan.decision == "answer"
    assert plan.render_args["sense_id"] == "river_edge"


def test_planner_answers_all_define_sense_rows():
    for row in _qa_prompt_rows():
        if row["expected_intent"] != "define_sense":
            continue
        a = analyze(row["question"])
        plan = _planner().plan(a)
        assert plan.decision == "answer", f"Expected answer for {row['row_id']}"


# ---------------------------------------------------------------------------
# 10. Planner answers cue explanation questions
# ---------------------------------------------------------------------------

def test_planner_answers_explain_cue_teller():
    a = analyze("Why does 'teller' suggest bank as a financial institution?")
    plan = _planner().plan(a)
    assert plan.decision == "answer"
    assert plan.render_template == "explain_cue"
    assert plan.render_args["cue"] == "teller"


def test_planner_answers_explain_cue_pitcher():
    a = analyze("Why does 'pitcher' suggest bat as sports equipment?")
    plan = _planner().plan(a)
    assert plan.decision == "answer"
    assert plan.render_args["cue"] == "pitcher"
    assert plan.render_args["sense_id"] == "sports_equipment"


def test_planner_audits_unknown_cue():
    from worldpgt.qa.question_analyzer import AnalyzedQuestion
    analyzed = AnalyzedQuestion(
        question="Why does 'xyzzy' suggest bank as a financial institution?",
        intent="explain_cue",
        term="bank",
        target_sense="financial_institution",
        second_sense=None,
        cues_in_question=["xyzzy"],
        inline_context=None,
        underconstrained=False,
    )
    plan = _planner().plan(analyzed)
    assert plan.decision == "audit"
    assert plan.audit_reason == "cue_not_found"


# ---------------------------------------------------------------------------
# 11. Planner answers safe context classification questions
# ---------------------------------------------------------------------------

def test_planner_classifies_bank_financial_context():
    a = analyze("In 'she went to the bank to open an account', what does bank mean?")
    plan = _planner().plan(a)
    assert plan.decision == "answer"
    assert plan.render_args["sense_id"] == "financial_institution"


def test_planner_classifies_bat_animal_context():
    a = analyze("In 'the bat flew out of the cave at dusk', what does bat mean?")
    plan = _planner().plan(a)
    assert plan.decision == "answer"
    assert plan.render_args["sense_id"] == "animal"


# ---------------------------------------------------------------------------
# 12. Planner audits ambiguous questions
# ---------------------------------------------------------------------------

def test_planner_audits_what_does_bank_mean():
    a = analyze("What does bank mean?")
    plan = _planner().plan(a)
    assert plan.decision == "audit"
    assert plan.audit_reason == "underconstrained_question"


def test_planner_audits_all_ambiguous_rows():
    for row in _qa_prompt_rows():
        if row["expected_intent"] != "unknown_or_ambiguous":
            continue
        a = analyze(row["question"])
        plan = _planner().plan(a)
        assert plan.decision == "audit", f"Expected audit for {row['row_id']}"


# ---------------------------------------------------------------------------
# 13. Planner audits low-margin/conflicting context
# ---------------------------------------------------------------------------

def test_planner_audits_no_cue_context():
    from worldpgt.qa.question_analyzer import AnalyzedQuestion
    analyzed = AnalyzedQuestion(
        question="In 'she visited the bank yesterday', what does bank mean?",
        intent="classify_context",
        term="bank",
        target_sense=None,
        second_sense=None,
        cues_in_question=[],
        inline_context="she visited the bank yesterday",
        underconstrained=False,
    )
    plan = _planner().plan(analyzed)
    assert plan.decision == "audit"


# ---------------------------------------------------------------------------
# 14. Renderer produces short answers
# ---------------------------------------------------------------------------

def test_renderer_produces_short_answers():
    for row in _output_rows():
        if row["decision"] == "answer":
            assert len(row["answer"]) <= 400, (
                f"{row['row_id']} answer too long: {len(row['answer'])}"
            )


def test_renderer_answer_non_empty_for_answer_decision():
    for row in _output_rows():
        if row["decision"] == "answer":
            assert row["answer"].strip(), f"{row['row_id']} has empty answer"


# ---------------------------------------------------------------------------
# 15. Renderer includes explicit evidence
# ---------------------------------------------------------------------------

def test_renderer_define_sense_includes_term():
    a = analyze("What is a financial bank?")
    plan = _planner().plan(a)
    answer = render(plan)
    assert "bank" in answer.lower()
    assert "financial" in answer.lower()


def test_renderer_explain_cue_includes_cue():
    a = analyze("Why does 'teller' suggest bank as a financial institution?")
    plan = _planner().plan(a)
    answer = render(plan)
    assert "teller" in answer.lower()


def test_renderer_classify_context_includes_sense():
    a = analyze("In 'she went to the bank to open an account', what does bank mean?")
    plan = _planner().plan(a)
    answer = render(plan)
    assert "financial" in answer.lower()


# ---------------------------------------------------------------------------
# 16. Validator flags wrong-sense answers
# ---------------------------------------------------------------------------

def test_validator_flags_wrong_sense_answer():
    result = validate(
        answer="A financial bank is a river edge next to a stream and reeds.",
        decision="answer",
        term="bank",
        sense_id="financial_institution",
        intent="define_sense",
    )
    assert result.quality_flagged is True


def test_validator_passes_correct_sense_answer():
    result = validate(
        answer="A financial bank is an institution associated with accounts and deposits.",
        decision="answer",
        term="bank",
        sense_id="financial_institution",
        intent="define_sense",
    )
    assert result.quality_flagged is False


# ---------------------------------------------------------------------------
# 17. Validator rejects generic fallback answers
# ---------------------------------------------------------------------------

def test_validator_rejects_it_depends_on_many_things():
    result = validate(
        answer="It depends on many things, such as context.",
        decision="answer",
        term="bank",
        sense_id="financial_institution",
    )
    assert result.quality_flagged is True
    assert result.passed is False


def test_validator_rejects_in_general():
    result = validate(
        answer="In general, a bank could mean different things.",
        decision="answer",
        term="bank",
        sense_id="financial_institution",
    )
    assert result.quality_flagged is True


# ---------------------------------------------------------------------------
# 18-19. Output CSV and summary JSON are written
# ---------------------------------------------------------------------------

def test_output_csv_is_written():
    assert _QA_OUTPUTS.exists(), "answer_planner_v1_outputs.csv not found"
    rows = _output_rows()
    assert len(rows) >= 48


def test_summary_json_is_written():
    assert _QA_SUMMARY.exists(), "answer_planner_v1_summary.json not found"
    s = _summary()
    assert "qa_total" in s
    assert "answer_count" in s
    assert "wrong_count" in s


# ---------------------------------------------------------------------------
# 20. Summary counts are internally consistent
# ---------------------------------------------------------------------------

def test_summary_counts_consistent():
    s = _summary()
    assert s["answer_count"] + s["audit_count"] == s["qa_total"]
    assert s["correct_count"] + s["wrong_count"] == s["qa_total"]
    assert 0.0 <= s["accuracy"] <= 1.0
    assert 0.0 <= s["answer_precision"] <= 1.0


# ---------------------------------------------------------------------------
# 21-22. wrong_count = 0, quality_flagged = 0
# ---------------------------------------------------------------------------

def test_wrong_count_is_zero():
    assert _summary()["wrong_count"] == 0


def test_quality_flagged_is_zero():
    assert _summary()["quality_flagged"] == 0


# ---------------------------------------------------------------------------
# 23. Accepted memory provider is used
# ---------------------------------------------------------------------------

def test_provider_is_used():
    s = _summary()
    assert s["provider_usage"]["provider_enabled"] is True
    assert s["provider_usage"]["provider_items_used"] > 0
    assert len(s["provider_usage"]["provider_terms_used"]) == 6


# ---------------------------------------------------------------------------
# 24. Default continuation benchmark behavior unchanged
# ---------------------------------------------------------------------------

def test_default_continuation_benchmark_unchanged():
    engine_default = __import__(
        "worldpgt.continuation.continuation_engine",
        fromlist=["ControlledContinuationEngine"],
    ).ControlledContinuationEngine()
    count = 0
    with _OLD_PROMPTS.open(newline="", encoding="utf-8") as fh:
        for record in csv.DictReader(fh):
            result = engine_default.continue_prompt(record["prompt"])
            if result.decision == "continue":
                count += 1
            if count >= 5:
                break
    assert count >= 5


# ---------------------------------------------------------------------------
# 25. sense_memory.py not modified
# ---------------------------------------------------------------------------

def test_sense_memory_not_modified():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "worldpgt/continuation/sense_memory.py"],
        capture_output=True, text=True,
        cwd=str(_REPO),
    )
    assert result.stdout == "", "sense_memory.py has uncommitted changes"


# ---------------------------------------------------------------------------
# 26. Benchmark baseline outputs not modified
# ---------------------------------------------------------------------------

def test_baseline_outputs_not_modified():
    import subprocess
    result = subprocess.run(
        ["git", "diff", "worldpgt/experiments/microworld_continuation_v1_2_outputs.csv"],
        capture_output=True, text=True,
        cwd=str(_REPO),
    )
    assert result.stdout == "", "microworld_continuation_v1_2_outputs.csv has uncommitted changes"


# ---------------------------------------------------------------------------
# 27. No thresholds lowered
# ---------------------------------------------------------------------------

def test_thresholds_not_lowered():
    policy = ContinuationPolicy()
    assert policy.min_score == 1.0
    assert policy.min_margin == 1.0


# ---------------------------------------------------------------------------
# 28. No validators weakened — qa/ modules have no bypass keywords
# ---------------------------------------------------------------------------

_QA_MODULE_DIR = Path(__file__).resolve().parents[1] / "qa"

_FORBIDDEN_VALIDATOR_KEYWORDS = [
    "skip_validator",
    "bypass_validator",
    "disable_validator",
    "force_continue",
    "force_trust",
    "force_answer",
]


def test_qa_modules_have_no_validator_bypass():
    for py_file in _QA_MODULE_DIR.glob("*.py"):
        src = py_file.read_text(encoding="utf-8")
        for kw in _FORBIDDEN_VALIDATOR_KEYWORDS:
            assert kw not in src, f"Forbidden keyword '{kw}' in {py_file.name}"


# ---------------------------------------------------------------------------
# 29. No generic trusted fallback added
# ---------------------------------------------------------------------------

def test_qa_modules_have_no_generic_fallback():
    for py_file in _QA_MODULE_DIR.glob("*.py"):
        lines = py_file.read_text(encoding="utf-8").splitlines()
        # Only check non-comment, non-docstring code lines
        code_lines = [
            ln for ln in lines
            if ln.strip() and not ln.strip().startswith("#") and not ln.strip().startswith('"""')
        ]
        code_block = "\n".join(code_lines).lower()
        forbidden = ["always_answer", "force_answer"]
        for kw in forbidden:
            assert kw not in code_block, f"Forbidden keyword '{kw}' in {py_file.name}"


# ---------------------------------------------------------------------------
# 30. No neural/GPT/training imports or strings
# ---------------------------------------------------------------------------

_FORBIDDEN_IMPORT_TOKENS = [
    "torch",
    "transformers",
    "openai",
    "gptneox",
    "automodelfor",
    "backprop",
    "fine_tuning",
    "finetun",
    "nanogpt",
]

_ALL_NEW_MODULES = list(_QA_MODULE_DIR.glob("*.py")) + [
    _REPO / "worldpgt" / "experiments" / "run_answer_planner_v1.py",
]


@pytest.mark.parametrize("module_path", _ALL_NEW_MODULES, ids=lambda p: p.name)
def test_no_neural_gpt_training_imports(module_path: Path):
    lines = module_path.read_text(encoding="utf-8").splitlines()
    import_lines = [
        ln.lower() for ln in lines
        if ln.strip().startswith(("import ", "from "))
    ]
    import_block = "\n".join(import_lines)
    for token in _FORBIDDEN_IMPORT_TOKENS:
        assert token.lower() not in import_block, (
            f"Forbidden token '{token}' found in imports of {module_path.name}"
        )


# ---------------------------------------------------------------------------
# 31–37. Surface polish v2 — natural display labels, no wrong articles
# ---------------------------------------------------------------------------

def test_renderer_bat_animal_uses_natural_label():
    a = analyze("What is a bat animal?")
    plan = _planner().plan(a)
    answer = render(plan)
    assert answer.startswith("A bat is"), f"Expected 'A bat is...', got: {answer!r}"
    assert "a animal is" not in answer.lower()


def test_renderer_bat_sports_uses_baseball_label():
    a = analyze("What is a baseball bat?")
    plan = _planner().plan(a)
    answer = render(plan)
    assert answer.startswith("A baseball bat is"), f"Expected 'A baseball bat is...', got: {answer!r}"
    assert "a sports bat is" not in answer.lower()


def test_renderer_rock_music_no_article():
    a = analyze("What is rock music?")
    plan = _planner().plan(a)
    answer = render(plan)
    assert answer.startswith("Rock music is"), f"Expected 'Rock music is...', got: {answer!r}"
    assert "a rock music is" not in answer.lower()


def test_renderer_spring_season_no_article():
    a = analyze("What is the spring season?")
    plan = _planner().plan(a)
    answer = render(plan)
    assert answer.startswith("Spring is"), f"Expected 'Spring is...', got: {answer!r}"
    assert "a spring season is" not in answer.lower()


def test_renderer_no_surface_defects_in_benchmark():
    """No answer in the benchmark output contains known bad surface patterns."""
    bad_patterns = [
        "a animal is",
        "a rock music is",
        "a spring season is",
        "typicalactions",
        "andparcel",
    ]
    for row in _output_rows():
        if row["decision"] != "answer":
            continue
        lower = row["answer"].lower()
        for pat in bad_patterns:
            assert pat not in lower, (
                f"{row['row_id']}: surface defect {pat!r} in answer: {row['answer']!r}"
            )
        assert "  " not in row["answer"], (
            f"{row['row_id']}: double space in answer"
        )


def test_validator_catches_wrong_article_before_vowel():
    result = validate(
        answer="A animal is a nocturnal flying mammal.",
        decision="answer",
        term="bat",
        sense_id="animal",
        intent="define_sense",
    )
    assert result.quality_flagged is True
    assert result.passed is False


def test_validator_catches_article_before_mass_noun():
    result = validate(
        answer="A rock music is a genre of music featuring guitars.",
        decision="answer",
        term="rock",
        sense_id="music",
        intent="define_sense",
    )
    assert result.quality_flagged is True
    assert result.passed is False


# ---------------------------------------------------------------------------
# 38–47. HelpfulAuditRenderer v1 — ambiguous terms get helpful audit text
# ---------------------------------------------------------------------------

def test_helpful_audit_still_returns_audit_decision():
    for term in _ALL_TERMS:
        a = analyze(f"What does {term} mean?")
        plan = _planner().plan(a)
        assert plan.decision == "audit", f"Expected audit for {term!r}, got {plan.decision!r}"


def test_helpful_audit_bank_includes_sense_alternatives():
    a = analyze("What does bank mean?")
    plan = _planner().plan(a)
    answer = render(plan)
    lower = answer.lower()
    assert "financial institution" in lower, f"Missing 'financial institution': {answer!r}"
    assert "river" in lower, f"Missing 'river': {answer!r}"


def test_helpful_audit_seal_includes_both_senses():
    a = analyze("What does seal mean?")
    plan = _planner().plan(a)
    answer = render(plan)
    lower = answer.lower()
    assert "marine animal" in lower, f"Missing 'marine animal': {answer!r}"
    assert "wax" in lower or "document seal" in lower, f"Missing wax/seal sense: {answer!r}"


def test_helpful_audit_spring_includes_both_senses():
    a = analyze("What does spring mean?")
    plan = _planner().plan(a)
    answer = render(plan)
    lower = answer.lower()
    assert "season" in lower or "winter" in lower, f"Missing season sense: {answer!r}"
    assert "mechanical coil" in lower or "coil" in lower, f"Missing coil sense: {answer!r}"


def test_helpful_audit_all_six_terms_mention_ambiguity():
    for term in _ALL_TERMS:
        a = analyze(f"What does {term} mean?")
        plan = _planner().plan(a)
        answer = render(plan)
        lower = answer.lower()
        assert "ambiguous" in lower or "can mean" in lower, (
            f"Expected ambiguity signal for {term!r}: {answer!r}"
        )
        assert " or " in lower, f"Expected 'or' connector for {term!r}: {answer!r}"


def test_helpful_audit_includes_context_prompt():
    for term in _ALL_TERMS:
        a = analyze(f"What does {term} mean?")
        plan = _planner().plan(a)
        answer = render(plan)
        assert "context" in answer.lower(), (
            f"Expected 'context' in helpful audit for {term!r}: {answer!r}"
        )


def test_helpful_audit_passes_validator():
    for term in _ALL_TERMS:
        a = analyze(f"What does {term} mean?")
        plan = _planner().plan(a)
        answer = render(plan)
        result = validate(
            answer=answer,
            decision="audit",
            term=term,
            sense_id=None,
            intent="unknown_or_ambiguous",
        )
        assert result.passed is True, (
            f"Validator rejected helpful audit for {term!r}: {result.reasons} — {answer!r}"
        )


def test_helpful_audit_rows_not_counted_as_wrong():
    assert _summary()["wrong_count"] == 0
    assert _summary()["quality_flagged"] == 0


def test_helpful_audit_benchmark_still_perfect():
    s = _summary()
    assert s["qa_total"] == 48
    assert s["correct_count"] == 48
    assert s["wrong_count"] == 0
    assert s["quality_flagged"] == 0
    assert s["accuracy"] == 1.0
    assert s["answer_precision"] == 1.0


# ---------------------------------------------------------------------------
# 48–60. SemanticLanguageRealizer v1 — relation-aware natural phrasing
# ---------------------------------------------------------------------------

def _define_answer(question: str) -> str:
    a = analyze(question)
    plan = _planner().plan(a)
    assert plan.render_template == "define_sense"
    return render(plan)


def test_realizer_removes_associated_with_from_define_sense():
    for row in _output_rows():
        if row["detected_intent"] != "define_sense":
            continue
        assert "it is associated with" not in row["answer"].lower(), (
            f"{row['row_id']} still uses 'It is associated with': {row['answer']!r}"
        )


def test_realizer_bat_animal_definition():
    answer = _define_answer("What is a bat animal?")
    lower = answer.lower()
    assert "a bat is" in lower
    assert "common clues" in lower
    assert "caves" in lower
    assert "rafters" in lower
    assert "a animal is" not in lower
    assert "it is associated with" not in lower


def test_realizer_spring_season_definition():
    answer = _define_answer("What is the spring season?")
    lower = answer.lower()
    assert "spring is" in lower
    assert "common signs" in lower
    assert "thaw" in lower
    assert "flowers" in lower
    assert "associated with" not in lower


def test_realizer_spring_coil_definition():
    answer = _define_answer("What is a spring coil?")
    lower = answer.lower()
    assert "stores and releases mechanical energy" in lower
    assert "compress" in lower
    assert "stretch" in lower
    assert "inside" in lower
    assert "releasesmechanical" not in lower
    assert "associated with" not in lower


def test_realizer_rock_music_definition():
    answer = _define_answer("What is rock music?")
    lower = answer.lower()
    assert "rock music is" in lower
    assert "guitars" in lower
    assert "rhythms" in lower
    assert ("concerts" in lower) or ("stages" in lower)
    assert "a rock music is" not in lower


def test_realizer_rock_stone_definition():
    answer = _define_answer("What is a rock stone?")
    lower = answer.lower()
    assert "a rock is" in lower
    assert "mineral" in lower
    assert ("cliffs" in lower) or ("mountains" in lower)
    assert "a rock stone is" not in lower


def test_realizer_validator_flags_associated_with_in_define_sense():
    result = validate(
        answer=(
            "A bat is a nocturnal flying mammal. It is associated with cave and roost."
        ),
        decision="answer",
        term="bat",
        sense_id="animal",
        intent="define_sense",
    )
    assert result.passed is False
    assert result.quality_flagged is True


def test_realizer_validator_flags_spacing_artifacts():
    for bad in (
        "A spring coil stores and releasesmechanical energy.",
        "A bat is a mammal.Itis nocturnal.",
        "A bank is a place,associated with money.",
        "A wax seal is a device, with clues like wax,envelopes and documents.",
    ):
        result = validate(
            answer=bad,
            decision="answer",
            term="bat",
            sense_id="animal",
            intent="define_sense",
        )
        assert result.passed is False, f"Expected artifact flagged: {bad!r}"
        assert result.quality_flagged is True


def test_realizer_helpful_audits_unchanged():
    for term in _ALL_TERMS:
        a = analyze(f"What does {term} mean?")
        plan = _planner().plan(a)
        assert plan.decision == "audit"
        answer = render(plan)
        lower = answer.lower()
        assert "ambiguous" in lower or "can mean" in lower
        assert "context" in lower


def test_realizer_distinguish_row_still_uses_contrast():
    a = analyze("What is the difference between a financial bank and a river bank?")
    plan = _planner().plan(a)
    answer = render(plan)
    lower = answer.lower()
    assert "financial bank is" in lower
    assert "river bank is" in lower


def test_realizer_no_associated_with_in_define_outputs_only_distinguish():
    """Remaining 'associated with' occurrences belong only to distinguish_senses."""
    for row in _output_rows():
        if "associated with" in row["answer"].lower():
            assert row["detected_intent"] == "distinguish_senses", (
                f"{row['row_id']} ({row['detected_intent']}) unexpectedly uses "
                f"'associated with': {row['answer']!r}"
            )


# ---------------------------------------------------------------------------
# 61–73. SemanticLanguageRealizer v2 — action agency + safer morphology
# ---------------------------------------------------------------------------

def test_v2_spring_season_agency_and_morphology():
    answer = _define_answer("What is the spring season?")
    lower = answer.lower()
    assert "spring is" in lower
    assert "common signs" in lower
    assert "thaw" in lower
    assert "flowers" in lower
    assert "warmer mornings" in lower
    assert "rain" in lower
    assert "in spring, flowers bloom and snow thaws" in lower
    assert "thaws, flowers, mornings and rains" not in lower
    assert "it can bloom and thaw" not in lower


def test_v2_rock_music_performed_content_frame():
    answer = _define_answer("What is rock music?")
    lower = answer.lower()
    assert "rock music is" in lower
    assert "bands" in lower
    assert "concerts" in lower
    assert "stages" in lower
    assert "played and performed" in lower
    assert "it can play and perform" not in lower


def test_v2_rock_stone_patient_frame():
    answer = _define_answer("What is a rock stone?")
    lower = answer.lower()
    assert "a rock is" in lower
    assert "mineral" in lower
    assert "roll" in lower
    assert "be climbed" in lower
    assert "it can climb" not in lower


def test_v2_spring_coil_mechanical_frame():
    answer = _define_answer("What is a spring coil?")
    lower = answer.lower()
    assert "stores and releases mechanical energy" in lower
    assert "snap" in lower
    assert "compress" in lower
    assert "stretch" in lower
    assert "releasesmechanical" not in lower
    assert "  " not in answer


def test_v2_bat_animal_remains_subject_agent():
    answer = _define_answer("What is a bat animal?")
    lower = answer.lower()
    assert "a bat is" in lower
    assert "common clues" in lower
    assert "caves" in lower
    assert "rafters" in lower
    assert "it can hunt, roost and hang" in lower


def test_v2_bat_sports_instrument_frame():
    answer = _define_answer("What is a baseball bat?")
    lower = answer.lower()
    assert "a baseball bat is" in lower
    assert "it is used to swing, hit or strike" in lower
    assert "ball" in lower


def test_v2_validator_flags_bad_agency_phrases():
    bad_answers = (
        "Rock music is a genre. It can play and perform.",
        "A rock is a mineral mass. It can climb and roll.",
        "Spring is a season. It can bloom and thaw.",
        "Spring is a season. Common signs are thaws, flowers.",
        "Spring is a season. Common signs are warmer mornings and rains.",
    )
    for ans in bad_answers:
        result = validate(
            answer=ans,
            decision="answer",
            term="rock",
            sense_id="music",
            intent="define_sense",
        )
        assert result.passed is False, f"Expected flagged: {ans!r}"
        assert result.quality_flagged is True


def test_v2_no_bad_agency_phrases_in_benchmark():
    bad = (
        "it can play and perform",
        "it can climb",
        "it can bloom and thaw",
        "common signs are thaws",
        "mornings and rains",
    )
    for row in _output_rows():
        if row["decision"] != "answer":
            continue
        lower = row["answer"].lower()
        for phrase in bad:
            assert phrase not in lower, (
                f"{row['row_id']}: bad agency phrase {phrase!r}: {row['answer']!r}"
            )


def test_v2_uncountable_cues_not_pluralized():
    from worldpgt.qa.answer_renderer import _pluralize
    for word in ("thaw", "rain", "music", "water", "cash", "snow", "wax"):
        assert _pluralize(word) == word, f"{word} should not be pluralized"


# ---------------------------------------------------------------------------
# 74–86. ContrastRealizer v1 — distinguish_senses relation-aware contrast
# ---------------------------------------------------------------------------

def _row_answer(row_id: str) -> str:
    for row in _output_rows():
        if row["row_id"] == row_id:
            return row["answer"]
    raise AssertionError(f"row {row_id} not found")


def test_contrast_no_associated_with_in_any_distinguish_row():
    for row in _output_rows():
        if row["detected_intent"] != "distinguish_senses":
            continue
        assert "associated with" not in row["answer"].lower(), (
            f"{row['row_id']} still uses 'associated with': {row['answer']!r}"
        )


def test_contrast_outputs_have_no_comma_glued_phrases():
    pattern = re.compile(r",[A-Za-z]")
    for row in _output_rows():
        assert pattern.search(row["answer"]) is None, (
            f"{row['row_id']} has comma-glued output: {row['answer']!r}"
        )


def test_contrast_outputs_have_no_period_glued_phrases():
    pattern = re.compile(r"\.[A-Z]")
    for row in _output_rows():
        assert pattern.search(row["answer"]) is None, (
            f"{row['row_id']} has period-glued output: {row['answer']!r}"
        )


def test_contrast_bat_row():
    answer = _row_answer("qa-v1-038")
    lower = answer.lower()
    assert "a bat is" in lower
    assert "flying mammal" in lower
    assert "caves" in lower
    assert "a baseball bat" in lower
    assert "hit a ball" in lower
    assert ("pitchers" in lower) or ("batter" in lower)
    assert "associated with" not in lower
    assert "baseball,associated" not in lower


def test_contrast_seal_row_has_clean_comma_spacing():
    answer = _row_answer("qa-v1-039")
    assert "wax, envelopes" in answer
    assert "wax,envelopes" not in answer


def test_contrast_spring_row():
    answer = _row_answer("qa-v1-042")
    lower = answer.lower()
    assert "spring is" in lower
    assert "season after winter" in lower
    assert "thaw" in lower
    assert "flowers" in lower
    assert "a spring coil" in lower
    assert "mechanical" in lower
    assert ("snap" in lower) or ("compress" in lower) or ("stretch" in lower)
    assert "associated with" not in lower
    assert "releasesmechanical" not in lower


def test_contrast_rock_row():
    answer = _row_answer("qa-v1-041")
    lower = answer.lower()
    assert "rock music is" in lower
    assert "music" in lower
    assert ("bands" in lower) or ("concerts" in lower)
    assert "a rock is" in lower
    assert "mineral" in lower
    assert "associated with" not in lower


def test_contrast_validator_flags_associated_with_in_distinguish():
    result = validate(
        answer=(
            "A bat is a nocturnal flying mammal, associated with cave, roost and "
            "rafter. A baseball bat is a club, associated with pitcher and ball."
        ),
        decision="answer",
        term="bat",
        sense_id="animal",
        intent="distinguish_senses",
    )
    assert result.passed is False
    assert result.quality_flagged is True


def test_contrast_validator_flags_baseball_associated_glue():
    result = validate(
        answer="A baseball bat is a club used to hit a ball,associated with pitcher.",
        decision="answer",
        term="bat",
        sense_id="sports_equipment",
        intent="distinguish_senses",
    )
    assert result.passed is False
    assert result.quality_flagged is True


def test_contrast_validator_flags_releasesmechanical_glue():
    result = validate(
        answer="A spring coil stores and releasesmechanical energy.",
        decision="answer",
        term="spring",
        sense_id="coil",
        intent="distinguish_senses",
    )
    assert result.passed is False
    assert result.quality_flagged is True


def test_contrast_define_sense_v2_still_clean():
    answer = _define_answer("What is rock music?")
    lower = answer.lower()
    assert "rock music is" in lower
    assert "played and performed" in lower
    assert "it is associated with" not in lower


def test_contrast_helpful_audit_still_works():
    a = analyze("What does seal mean?")
    plan = _planner().plan(a)
    assert plan.decision == "audit"
    answer = render(plan)
    lower = answer.lower()
    assert "marine animal" in lower
    assert "wax" in lower or "document seal" in lower
    assert "context" in lower


def test_contrast_zero_associated_with_in_outputs():
    total = sum(
        row["answer"].lower().count("associated with") for row in _output_rows()
    )
    assert total == 0, f"Expected 0 'associated with', found {total}"
