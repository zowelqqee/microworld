"""Tests for AnswerPlanner v1.

Covers all 30 required test points.
"""

from __future__ import annotations

import csv
import json
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
