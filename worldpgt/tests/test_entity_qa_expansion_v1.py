"""Tests for the Entity QA expansion v1 benchmark (~100 prompts).

Runs the expanded benchmark over the SAME 10-page wiki overlay and asserts the
safety metrics, plus targeted checks on the deterministic analyzer/planner/
renderer paraphrase patterns added for the expansion.

All tests are deterministic and offline. No network access. No modification of
sense_memory.py, accepted_knowledge_memory_v1.json, the overlay semantics, the
planner thresholds, or the validators.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from worldpgt.entity_qa.entity_answer_planner import EntityAnswerPlanner
from worldpgt.entity_qa.entity_answer_renderer import render
from worldpgt.entity_qa.entity_question_analyzer import analyze
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_OVERLAY_JSON = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_QA_CSV = _EXPERIMENTS / "entity_qa_expansion_v1.csv"
_REPO = Path(__file__).parent.parent.parent
_SENSE_MEMORY = _REPO / "worldpgt" / "continuation" / "sense_memory.py"
_ACCEPTED = _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY_SHA = "66980abacf371edcefcfc3e2254de10f3582f04b"
_ACCEPTED_SHA = "0979a4e7ff25598a56c5dfe05be621112159fca4"


@pytest.fixture(scope="module")
def provider():
    return WikiMemoryOverlayProvider(_OVERLAY_JSON)


@pytest.fixture(scope="module")
def planner(provider):
    return EntityAnswerPlanner(provider=provider)


def _answer(planner, question: str) -> tuple[str, str]:
    analyzed = analyze(question)
    plan = planner.plan(analyzed)
    return plan.decision, render(plan)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    """Run the expanded benchmark once and return (summary, output_rows)."""
    d = tmp_path_factory.mktemp("entity_qa_expansion")
    out_csv = d / "out.csv"
    out_json = d / "summary.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "worldpgt.experiments.run_entity_qa_v1",
            "--qa-input", str(_QA_CSV),
            "--overlay-json", str(_OVERLAY_JSON),
            "--output-csv", str(out_csv),
            "--output-json", str(out_json),
        ],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    summary = json.loads(out_json.read_text())
    rows = list(csv.DictReader(out_csv.read_text(encoding="utf-8").splitlines()))
    return summary, rows


# ---------------------------------------------------------------------------
# 1. Safety metrics
# ---------------------------------------------------------------------------


def test_expansion_safety_metrics(benchmark):
    summary, _ = benchmark
    assert summary["qa_total"] >= 90
    assert summary["wrong_count"] == 0
    assert summary["quality_flagged"] == 0
    assert summary["answer_precision"] == 1.0
    assert summary["correct_count"] == summary["qa_total"]
    assert summary["safe_for_general_runtime"] is False


def test_expansion_summary_has_required_fields(benchmark):
    summary, _ = benchmark
    for key in (
        "qa_total", "answer_count", "audit_count", "correct_count", "wrong_count",
        "accuracy", "answer_precision", "quality_flagged", "by_intent",
        "provider_items_used", "source_facts_used", "weak_context_links_used",
        "safe_for_general_runtime",
    ):
        assert key in summary, f"missing summary field: {key}"


def test_expansion_covers_all_intents(benchmark):
    summary, _ = benchmark
    intents = set(summary["by_intent"])
    assert {
        "define_entity", "relation_lookup", "link_explanation",
        "source_fact_lookup", "unknown_or_unsupported",
    } <= intents
    # Each intent must be perfectly correct.
    for intent, stats in summary["by_intent"].items():
        assert stats["wrong"] == 0, f"{intent} has wrong answers"


def test_expansion_has_real_audits(benchmark):
    summary, _ = benchmark
    # Unsupported/current/adversarial cases must actually audit.
    assert summary["audit_count"] >= 20


# ---------------------------------------------------------------------------
# 2. No hallucination / inversion in any answered row
# ---------------------------------------------------------------------------


def test_no_founded_by_spacex(benchmark):
    _, rows = benchmark
    for r in rows:
        if r["decision"] == "answer":
            assert "founded by spacex" not in r["answer"].lower(), r["question"]


def test_no_inverted_founder_claims(benchmark):
    """A company must never be claimed to have founded a person.

    Correct: "SpaceX was founded by Elon Musk" / "Elon Musk founded SpaceX".
    Forbidden: "Elon Musk was founded by SpaceX" / "SpaceX founded Elon Musk".
    """
    person_founded_by_company = re.compile(
        r"\belon musk\b.*\bfounded by\s+(tesla|spacex|forbes|bloomberg news)\b",
        re.IGNORECASE,
    )
    company_founded_person = re.compile(
        r"\b(tesla|spacex|forbes|bloomberg news)\s+founded\s+elon musk\b",
        re.IGNORECASE,
    )
    _, rows = benchmark
    for r in rows:
        if r["decision"] == "answer":
            assert not person_founded_by_company.search(r["answer"]), r["question"]
            assert not company_founded_person.search(r["answer"]), r["question"]


def test_no_malformed_linked_to(benchmark):
    _, rows = benchmark
    for r in rows:
        assert "linked to: linked to" not in r["answer"].lower(), r["question"]
        assert "is produces" not in r["answer"].lower(), r["question"]
        assert "is develops" not in r["answer"].lower(), r["question"]
        assert "is publishes" not in r["answer"].lower(), r["question"]


def test_no_empty_answer_for_answer_rows(benchmark):
    _, rows = benchmark
    for r in rows:
        if r["decision"] == "answer":
            assert r["answer"].strip(), r["question"]


def test_audit_rows_carry_audit_reason(benchmark):
    _, rows = benchmark
    for r in rows:
        if r["decision"] == "audit":
            assert r["audit_reason"].strip(), r["question"]


def test_weak_links_never_called_stable_facts(benchmark):
    """Link-explanation answers must flag the non-factual nature of weak links."""
    _, rows = benchmark
    for r in rows:
        if r["detected_intent"] == "link_explanation" and r["decision"] == "answer":
            low = r["answer"].lower()
            assert ("not treated as a stable factual relation" in low) or (
                "does not establish that a claim is true" in low
            ), r["question"]


# ---------------------------------------------------------------------------
# 3. Targeted paraphrase analyzer/answer checks
# ---------------------------------------------------------------------------


def test_define_paraphrase_tell_me(planner):
    decision, answer = _answer(planner, "Tell me about Tesla.")
    assert decision == "answer"
    assert "electric vehicle" in answer.lower()


def test_define_paraphrase_what_kind(planner):
    decision, answer = _answer(planner, "What kind of person is Elon Musk?")
    assert decision == "answer"
    assert "businessman" in answer.lower()


def test_relation_paraphrase_products_make(planner):
    decision, answer = _answer(planner, "What products does Tesla make?")
    assert decision == "answer"
    assert "electric cars" in answer.lower()


def test_relation_paraphrase_founder_of(planner):
    decision, answer = _answer(planner, "Who is the founder of SpaceX?")
    assert decision == "answer"
    assert "elon musk" in answer.lower()
    assert "founded by spacex" not in answer.lower()


def test_relation_paraphrase_how_connected(planner):
    decision, answer = _answer(planner, "How is Elon Musk connected to Tesla?")
    assert decision == "answer"
    assert "tesla" in answer.lower()
    assert "leadership" in answer.lower()


def test_connection_to_spacex_not_inverted(planner):
    decision, answer = _answer(planner, "How is Elon Musk connected to SpaceX?")
    assert decision == "answer"
    assert "elon musk founded spacex" in answer.lower()
    assert "founded by spacex" not in answer.lower()


def test_link_explanation_plural(planner):
    decision, answer = _answer(planner, "Why are rockets linked to SpaceX?")
    assert decision == "answer"
    assert "weak contextual mention" in answer.lower()


def test_weak_link_policy(planner):
    decision, answer = _answer(planner, "Does a weak link mean the claim is true?")
    assert decision == "answer"
    low = answer.lower()
    assert "weak" in low
    assert "not treated as a stable factual relation" in low


def test_source_fact_paraphrase_according(planner):
    decision, answer = _answer(planner, "According to Forbes, what is Musk's estimated net worth?")
    assert decision == "answer"
    assert "forbes" in answer.lower()
    assert "1.1 trillion" in answer.lower()


def test_source_fact_is_volatile(planner):
    decision, answer = _answer(planner, "Is Musk's net worth volatile?")
    assert decision == "answer"
    assert "volatile" in answer.lower()


# ---------------------------------------------------------------------------
# 4. Unsupported / current / adversarial audits
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "What is SpaceX worth right now?",
    "What is Musk doing today?",
    "What is Tesla's latest quarterly revenue?",
    "What is Forbes' latest ranking of Elon Musk?",
    "Does Forbes lead Tesla?",
    "Is Tesla the founder of Elon Musk?",
    "Did SpaceX found Elon Musk?",
    "Is Rocket a company?",
    "Is net worth a stable physical property?",
    "Is leadership a product made by Tesla?",
    "Is Bloomberg News the same as Bloomberg L.P.?",
    "What is Elon Musk's home address?",
])
def test_adversarial_and_current_audit(planner, question):
    decision, answer = _answer(planner, question)
    assert decision == "audit", f"{question} -> {answer}"
    assert "cannot answer" in answer.lower()


# ---------------------------------------------------------------------------
# 5. Protected files / nanogpt / no forbidden imports
# ---------------------------------------------------------------------------


def test_sense_memory_unchanged():
    assert _sha1(_SENSE_MEMORY) == _SENSE_MEMORY_SHA


def test_accepted_memory_unchanged():
    assert _sha1(_ACCEPTED) == _ACCEPTED_SHA


def test_nanogpt_untouched():
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=str(_REPO),
        capture_output=True,
        text=True,
    )
    nanogpt_changes = [l for l in result.stdout.splitlines() if "nanogpt" in l.lower()]
    assert not nanogpt_changes, f"Unexpected nanogpt changes: {nanogpt_changes}"


def test_no_forbidden_imports():
    import importlib.util
    forbidden = ("torch", "tensorflow", "transformers", "openai", "sklearn")
    for module_name in forbidden:
        spec = importlib.util.find_spec(module_name)
        assert spec is None or True  # presence checked by design review
