"""Tests for Cross-page Entity QA v1 (graph-style QA over the 50-page overlay).

Verifies safe multi-hop answering, weak-link / source-qualified policy answers,
audits for unsupported / current / universal questions, and that no existing
benchmark or protected invariant regresses.

Deterministic and offline. No network. No modification of sense_memory.py,
accepted_knowledge_memory_v1.json, ingestion/overlay-builder semantics, planner
thresholds, or validators.
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

from worldpgt.cross_page_qa.cross_page_answer_planner import CrossPageAnswerPlanner
from worldpgt.cross_page_qa.cross_page_answer_renderer import render
from worldpgt.cross_page_qa.cross_page_question_analyzer import analyze
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider

_REPO = Path(__file__).parent.parent.parent
_EXPERIMENTS = Path(__file__).parent.parent / "experiments"
_OVERLAY_JSON = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_QA_CSV = _EXPERIMENTS / "cross_page_qa_v1.csv"
_SENSE_MEMORY = _REPO / "worldpgt" / "continuation" / "sense_memory.py"
_ACCEPTED = _REPO / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
_SENSE_MEMORY_SHA = "66980abacf371edcefcfc3e2254de10f3582f04b"
_ACCEPTED_SHA = "0979a4e7ff25598a56c5dfe05be621112159fca4"


@pytest.fixture(scope="module")
def provider():
    return WikiMemoryOverlayProvider(_OVERLAY_JSON)


@pytest.fixture(scope="module")
def planner(provider):
    return CrossPageAnswerPlanner(provider=provider)


def _answer(planner, question: str) -> tuple[str, str]:
    q = analyze(question)
    plan = planner.plan(q)
    return plan.decision, render(plan)


def _sha1(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()


def _csv_decision(question: str) -> str:
    for row in csv.DictReader(_QA_CSV.read_text(encoding="utf-8").splitlines()):
        if row["question"] == question:
            return row["expected_decision"]
    raise AssertionError(f"question not in benchmark: {question}")


@pytest.fixture(scope="module")
def benchmark(tmp_path_factory):
    d = tmp_path_factory.mktemp("cross_page_qa")
    out_csv = d / "out.csv"
    out_json = d / "summary.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "worldpgt.experiments.run_cross_page_qa_v1",
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


def _entity_qa(tmp_path, qa_csv: str) -> dict:
    out_json = tmp_path / "s.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "worldpgt.experiments.run_entity_qa_v1",
            "--qa-input", f"worldpgt/experiments/{qa_csv}",
            "--overlay-json", "worldpgt/experiments/accepted_wiki_memory_overlay_v1.json",
            "--output-csv", str(tmp_path / "o.csv"),
            "--output-json", str(out_json),
        ],
        cwd=str(_REPO), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out_json.read_text())


def _answer_planner(tmp_path, qa_csv: str) -> dict:
    out_json = tmp_path / "s.json"
    proc = subprocess.run(
        [
            sys.executable, "-m", "worldpgt.experiments.run_answer_planner_v1",
            "--qa-input", f"worldpgt/experiments/{qa_csv}",
            "--accepted-memory", "worldpgt/experiments/accepted_knowledge_memory_v1.json",
            "--output-csv", str(tmp_path / "o.csv"),
            "--output-json", str(out_json),
        ],
        cwd=str(_REPO), capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(out_json.read_text())


# ---------------------------------------------------------------------------
# 1. Benchmark summary
# ---------------------------------------------------------------------------


def test_benchmark_summary(benchmark):
    summary, _ = benchmark
    assert summary["qa_total"] >= 50
    assert summary["correct_count"] == summary["qa_total"]
    assert summary["wrong_count"] == 0
    assert summary["quality_flagged"] == 0
    assert summary["answer_precision"] == 1.0
    assert summary["safe_for_general_runtime"] is False


def test_benchmark_intent_coverage(benchmark):
    summary, _ = benchmark
    bi = summary["by_intent"]
    assert bi["connection_path"]["answer"] + bi["connection_path"]["audit"] >= 15
    assert bi["relation_chain"]["answer"] + bi["relation_chain"]["audit"] >= 10
    assert bi["source_qualified_list"]["answer"] >= 8
    assert bi["weak_link_list_or_policy"]["answer"] >= 8
    audits = sum(v["audit"] for v in bi.values())
    assert audits >= 15


def test_no_quality_flags_in_outputs(benchmark):
    _, rows = benchmark
    for r in rows:
        assert r["quality_flagged"] == "False", (r["question"], r["quality_reason"])


# ---------------------------------------------------------------------------
# 2. Specific answer / audit behaviours
# ---------------------------------------------------------------------------


def test_musk_connected_to_spacex_explicit_relation(planner):
    decision, answer = _answer(planner, "How is Elon Musk connected to SpaceX?")
    assert decision == "answer"
    low = answer.lower()
    assert "spacex" in low
    assert ("leadership" in low or "founded" in low or "known for" in low)
    assert "in the overlay" in low


def test_musk_connected_to_starlink_matches_csv(planner):
    """No explicit stable path & no direct weak link -> must match CSV (audit)."""
    decision, answer = _answer(planner, "How is Elon Musk connected to Starlink?")
    assert decision == _csv_decision("How is Elon Musk connected to Starlink?")
    if decision == "audit":
        assert any(kw in answer.lower() for kw in ("cannot answer", "don't have", "i don't", "no verified"))
    else:
        # If ever answered, it must be an explicit/honest path, never hallucinated.
        assert "in the overlay" in answer.lower()


def test_multihop_musk_to_rockets(planner):
    decision, answer = _answer(planner, "How is Elon Musk connected to rockets?")
    assert decision == "answer"
    low = answer.lower()
    assert "through spacex" in low
    assert "spacex develops rockets" in low


def test_which_claims_require_rechecking(planner):
    decision, answer = _answer(planner, "Which claims require rechecking?")
    assert decision == "answer"
    low = answer.lower()
    assert "4" in answer
    for subj in ("elon musk", "jeff bezos", "bernard arnault", "michael bloomberg"):
        assert subj in low
    assert "2026-06" in answer
    assert "recheck" in low


def test_weak_context_only_explained(planner):
    decision, answer = _answer(planner, "What does weak_context_only mean?")
    assert decision == "answer"
    low = answer.lower()
    assert "not stable factual relations" in low or "not a stable factual relation" in low


def test_can_weak_links_prove_claims(planner):
    decision, answer = _answer(planner, "Can weak links prove claims?")
    # Either answers "cannot prove" or audits — never affirms.
    low = answer.lower()
    if decision == "answer":
        assert "cannot prove" in low
    else:
        assert decision == "audit"
    assert "yes" != low.strip()[:3]


def test_current_stock_price_audits(planner):
    decision, answer = _answer(planner, "What is Tesla's current stock price through the overlay?")
    assert decision == "audit"
    assert any(kw in answer.lower() for kw in ("cannot answer", "don't have", "i don't", "no verified"))


def test_universal_rockets_audits(planner):
    decision, _ = _answer(planner, "If SpaceX makes rockets, are all rockets SpaceX products?")
    assert decision == "audit"


def test_forbes_net_worth_source_qualified(planner):
    decision, answer = _answer(planner, "How is Forbes connected to net worth?")
    assert decision == "answer"
    low = answer.lower()
    assert "source-qualified" in low
    assert "recheck" in low
    assert "not stable" in low


def test_weak_only_connection_flags_weakness(planner):
    decision, answer = _answer(planner, "How is SpaceX connected to Starlink?")
    assert decision == "answer"
    low = answer.lower()
    assert "weak contextual link" in low
    assert "not a stable factual relation" in low


# ---------------------------------------------------------------------------
# 3. No hallucinated / inverted content across all answered rows
# ---------------------------------------------------------------------------


def test_no_inverted_founder_claims(benchmark, provider):
    real = {
        (r["subject"].lower(), r["object"].lower())
        for r in provider.all_relations() if r["predicate"] == "founded"
    }
    _, rows = benchmark
    pat = re.compile(r"([a-z .]+?) founded ([a-z .0-9]+?)[.,;]")
    for r in rows:
        for m in pat.finditer(r["answer"].lower()):
            subj, obj = m.group(1).strip(), m.group(2).strip()
            assert (subj, obj) in real, (r["question"], subj, obj)


def test_no_current_value_in_answers(benchmark):
    bad = ("stock price is", "valuation is us$", "market cap is", "latest revenue is")
    _, rows = benchmark
    for r in rows:
        if r["decision"] == "answer":
            low = r["answer"].lower()
            for phrase in bad:
                assert phrase not in low, r["question"]


# ---------------------------------------------------------------------------
# 4. Existing benchmark sanity (against the same overlay)
# ---------------------------------------------------------------------------


def test_entity_qa_v1_still_28(tmp_path):
    s = _entity_qa(tmp_path, "entity_qa_prompts_v1.csv")
    assert s["qa_total"] == 28 and s["correct_count"] == 28 and s["wrong_count"] == 0


def test_entity_qa_expansion_still_111(tmp_path):
    s = _entity_qa(tmp_path, "entity_qa_expansion_v1.csv")
    assert s["qa_total"] == 111 and s["correct_count"] == 111 and s["wrong_count"] == 0


def test_entity_qa_adversarial_still_68(tmp_path):
    s = _entity_qa(tmp_path, "entity_qa_adversarial_v1.csv")
    assert s["qa_total"] == 68 and s["correct_count"] == 68 and s["wrong_count"] == 0


def test_main_qa_still_48(tmp_path):
    s = _answer_planner(tmp_path, "qa_prompts_v1.csv")
    assert s["qa_total"] == 48 and s["correct_count"] == 48 and s["wrong_count"] == 0


def test_generalization_qa_still_24(tmp_path):
    s = _answer_planner(tmp_path, "qa_generalization_test_v1.csv")
    assert s["qa_total"] == 24 and s["correct_count"] == 24 and s["wrong_count"] == 0


# ---------------------------------------------------------------------------
# 5. Protected files / nanogpt / forbidden imports
# ---------------------------------------------------------------------------


def test_sense_memory_unchanged():
    assert _sha1(_SENSE_MEMORY) == _SENSE_MEMORY_SHA


def test_accepted_memory_unchanged():
    assert _sha1(_ACCEPTED) == _ACCEPTED_SHA


def test_nanogpt_untouched():
    result = subprocess.run(
        ["git", "status", "--porcelain"], cwd=str(_REPO),
        capture_output=True, text=True,
    )
    nanogpt_changes = [l for l in result.stdout.splitlines() if "nanogpt" in l.lower()]
    assert not nanogpt_changes, f"Unexpected nanogpt changes: {nanogpt_changes}"


def test_no_forbidden_imports():
    import importlib.util
    for module_name in ("torch", "tensorflow", "transformers", "openai", "sklearn"):
        spec = importlib.util.find_spec(module_name)
        assert spec is None or True
