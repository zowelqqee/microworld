"""Tests for Pump Fact QA Benchmark v1.

Deterministic, local-only. No network. The benchmark proves precision-filtered
pump facts are answerable through the assistant surface (``--overlay
pump-dry-run``) while reversed and current/live prompts are still audited.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worldpgt.experiments import run_pump_fact_qa_v1 as runner
from worldpgt.pump_fact_qa.answer_validator import (
    validate_adversarial,
    validate_current,
    validate_positive,
)
from worldpgt.pump_fact_qa.fact_loader import load_pump_facts
from worldpgt.pump_fact_qa.question_generator import (
    generate_adversarial_prompts,
    generate_all_prompts,
    generate_current_prompts,
    generate_positive_prompts,
)
from worldpgt.pump_fact_qa.qa_runner import run_prompts
from worldpgt.pump_fact_qa.report import build_summary, collect_planner_gaps
from worldpgt.pump_fact_qa.types import (
    CATEGORY_ADVERSARIAL,
    CATEGORY_CURRENT,
    CATEGORY_POSITIVE,
    FACT_KIND_DEFINITION,
    FACT_KIND_RELATION,
    PumpFact,
    QAPrompt,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_WORLD = _REPO / "worldpgt"
_EXP = _WORLD / "experiments"
_PUMP_OUT = _EXP / "knowledge_pump_v1"
_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #
def _sample_facts() -> list[PumpFact]:
    return [
        PumpFact(FACT_KIND_RELATION, "Starlink", "is_a", "Satellite internet", "SpaceX"),
        PumpFact(FACT_KIND_RELATION, "SpaceX", "develops", "Falcon 9", "Falcon 9"),
        PumpFact(FACT_KIND_RELATION, "Jeff Bezos", "founded", "Amazon", "Jeff Bezos"),
        PumpFact(FACT_KIND_RELATION, "Bloomberg News", "founded_by", "Michael Bloomberg", "Bloomberg News"),
        PumpFact(FACT_KIND_RELATION, "SolarCity", "owned_by", "Tesla", "Gigafactory New York"),
        PumpFact(FACT_KIND_RELATION, "International Energy Agency", "publishes", "annual World Energy Outlook", "International Energy Agency"),
        PumpFact(FACT_KIND_RELATION, "Mark Carney", "leader_of", "Liberal Party", "Mark Carney"),
        PumpFact(FACT_KIND_DEFINITION, "Rocket Science Games", "is_a", "independent game studio", "Rocket Science Games"),
    ]


def _answer(**kwargs) -> dict:
    base = {
        "decision": "answer",
        "support_kind": "semi_stable_relation",
        "supported_by_context": True,
        "overlay_mode": "pump-dry-run",
        "answer_text": "",
        "risk_flags": [],
        "trace": {"steps": ["x"]},
    }
    base.update(kwargs)
    return base


@pytest.fixture(scope="module")
def qa_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("pump_fact_qa")
    # Do not write into the real pump_summary/report from the test run.
    summary = runner.run(out_dir=out, update_pump_summary=False)
    return out, summary


# --------------------------------------------------------------------------- #
# 1. Fact loader reads precision-filtered answerable facts
# --------------------------------------------------------------------------- #
def test_fact_loader_reads_precision_filtered_facts(tmp_path):
    precision = tmp_path / "precision.json"
    precision.write_text(json.dumps([
        {"overlay_type": "overlay_relation", "subject": "SpaceX", "predicate": "develops", "object": "Falcon 9"},
        {"overlay_type": "overlay_definition", "subject": "June", "definition": "sixth month", "predicate": "is_a"},
    ]), encoding="utf-8")
    facts = load_pump_facts(precision, None)
    kinds = {f.kind for f in facts}
    assert kinds == {FACT_KIND_RELATION, FACT_KIND_DEFINITION}
    assert any(f.subject == "SpaceX" and f.obj == "Falcon 9" for f in facts)
    assert any(f.subject == "June" and f.obj == "sixth month" for f in facts)


def test_fact_loader_falls_back_when_precision_missing(tmp_path):
    fallback = tmp_path / "fallback.json"
    fallback.write_text(json.dumps([
        {"overlay_type": "overlay_relation", "subject": "A", "predicate": "owned_by", "object": "B"},
    ]), encoding="utf-8")
    facts = load_pump_facts(tmp_path / "missing.json", fallback)
    assert len(facts) == 1
    assert facts[0].subject == "A"


# --------------------------------------------------------------------------- #
# 2. Entity cards are not used to generate answerable fact questions
# --------------------------------------------------------------------------- #
def test_entity_cards_are_not_answerable_facts(tmp_path):
    precision = tmp_path / "precision.json"
    precision.write_text(json.dumps([
        {"overlay_type": "overlay_entity", "label": "Adposition", "entity_type": "unknown"},
        {"overlay_type": "overlay_relation", "subject": "SpaceX", "predicate": "develops", "object": "Falcon 9"},
    ]), encoding="utf-8")
    facts = load_pump_facts(precision, None)
    assert all(f.kind in (FACT_KIND_RELATION, FACT_KIND_DEFINITION) for f in facts)
    assert all(f.subject != "Adposition" for f in facts)
    prompts = generate_positive_prompts(facts)
    assert all("Adposition" not in p.question for p in prompts)


# --------------------------------------------------------------------------- #
# 17. Weak context is not used as facts
# --------------------------------------------------------------------------- #
def test_weak_context_links_are_not_facts(tmp_path):
    precision = tmp_path / "precision.json"
    precision.write_text(json.dumps([
        {"overlay_type": "overlay_context_link", "source_page": "A", "target": "B",
         "relation": "mentioned_with", "trust": "weak_context_only"},
        {"overlay_type": "overlay_relation", "subject": "SpaceX", "predicate": "develops", "object": "Falcon 9"},
    ]), encoding="utf-8")
    facts = load_pump_facts(precision, None)
    assert len(facts) == 1
    assert facts[0].predicate == "develops"


# --------------------------------------------------------------------------- #
# 3. Relation questions are generated for each supported predicate
# --------------------------------------------------------------------------- #
def test_relation_questions_generated_for_each_predicate():
    facts = _sample_facts()
    prompts = generate_positive_prompts(facts)
    questions = [p.question for p in prompts]
    assert "What is Starlink?" in questions
    assert "What does SpaceX develop?" in questions
    assert "What did Jeff Bezos found?" in questions
    assert "Who founded Bloomberg News?" in questions
    assert "Who owns SolarCity?" in questions
    assert "What company owns SolarCity?" in questions
    assert "What does International Energy Agency publish?" in questions
    assert "What is Mark Carney leader of?" in questions


def test_every_relation_predicate_yields_positive_prompts():
    predicates = ["is_a", "develops", "founded", "founded_by", "owned_by", "publishes", "leader_of"]
    for predicate in predicates:
        fact = PumpFact(FACT_KIND_RELATION, "X", predicate, "Y", "src")
        prompts = generate_positive_prompts([fact])
        assert prompts, f"no positive prompts for predicate {predicate}"
        assert all(p.category == CATEGORY_POSITIVE for p in prompts)


# --------------------------------------------------------------------------- #
# 4. Definition questions are generated
# --------------------------------------------------------------------------- #
def test_definition_questions_generated():
    fact = PumpFact(FACT_KIND_DEFINITION, "Rocket Science Games", "is_a", "independent game studio", "src")
    prompts = generate_positive_prompts([fact])
    questions = [p.question for p in prompts]
    assert "What is Rocket Science Games?" in questions
    assert "Define Rocket Science Games." in questions
    # Core definition tokens are part of the expectation.
    what_is = next(p for p in prompts if p.question == "What is Rocket Science Games?")
    assert "Rocket Science Games" in what_is.expected_tokens
    assert "independent" in what_is.expected_tokens


# --------------------------------------------------------------------------- #
# 5. Adversarial inversion questions are generated
# --------------------------------------------------------------------------- #
def test_adversarial_inversion_questions_generated():
    facts = _sample_facts()
    prompts = generate_adversarial_prompts(facts)
    questions = [p.question for p in prompts]
    assert "Did Amazon found Jeff Bezos?" in questions
    assert "Did Bloomberg News found Michael Bloomberg?" in questions
    assert "Did Falcon 9 develop SpaceX?" in questions
    assert "Does SolarCity own Tesla?" in questions
    assert "Does annual World Energy Outlook publish International Energy Agency?" in questions
    assert all(p.expected_decision == "audit" for p in prompts)


# --------------------------------------------------------------------------- #
# 6. Current/live safety questions are generated
# --------------------------------------------------------------------------- #
def test_current_live_safety_questions_generated():
    prompts = generate_current_prompts()
    questions = [p.question for p in prompts]
    assert "What is Tesla's current stock price?" in questions
    assert "What is SpaceX's current valuation?" in questions
    assert "What is Jeff Bezos's current net worth?" in questions
    assert "Is Mark Carney currently the leader of the Liberal Party?" in questions
    assert "Is Starlink currently the largest satellite network?" in questions
    assert all(p.category == CATEGORY_CURRENT for p in prompts)
    assert all(p.expected_decision == "audit" for p in prompts)


# --------------------------------------------------------------------------- #
# 7. Positive validator accepts supported answer containing expected tokens
# --------------------------------------------------------------------------- #
def test_positive_validator_accepts_supported_answer():
    prompt = QAPrompt("pos-0", "Who owns SolarCity?", CATEGORY_POSITIVE, "answer",
                      predicate="owned_by", subject="SolarCity", obj="Tesla",
                      expected_tokens=["SolarCity", "Tesla"])
    answer = _answer(answer_text="SolarCity is owned by Tesla.")
    valid, classification = validate_positive(prompt, answer)
    assert valid is True
    assert classification == "ok"


def test_positive_validator_accepts_dropped_leading_the_in_title():
    prompt = QAPrompt("pos-article", "Who founded The Noyce School?",
                      CATEGORY_POSITIVE, "answer",
                      predicate="founded",
                      subject="California Polytechnic State University",
                      obj="The Noyce School",
                      expected_tokens=[
                          "California Polytechnic State University",
                          "The Noyce School",
                      ])
    answer = _answer(
        answer_text=(
            "Noyce School was founded by California Polytechnic State University."
        )
    )
    valid, classification = validate_positive(prompt, answer)
    assert valid is True
    assert classification == "ok"


# --------------------------------------------------------------------------- #
# 8. Positive validator flags unsupported answer
# --------------------------------------------------------------------------- #
def test_positive_validator_flags_unsupported_answer():
    prompt = QAPrompt("pos-1", "Who owns SolarCity?", CATEGORY_POSITIVE, "answer",
                      predicate="owned_by", subject="SolarCity", obj="Tesla",
                      expected_tokens=["SolarCity", "Tesla"])
    answer = _answer(supported_by_context=False, support_kind="unsupported",
                     answer_text="SolarCity is owned by Tesla.")
    valid, classification = validate_positive(prompt, answer)
    assert valid is False
    assert classification == "unsupported_answer"


def test_positive_validator_flags_wrong_answer_missing_tokens():
    prompt = QAPrompt("pos-2", "Who owns SolarCity?", CATEGORY_POSITIVE, "answer",
                      predicate="owned_by", subject="SolarCity", obj="Tesla",
                      expected_tokens=["SolarCity", "Tesla"])
    answer = _answer(answer_text="SolarCity is owned by Amazon.")
    valid, classification = validate_positive(prompt, answer)
    assert valid is False
    assert classification == "wrong_answer"


def test_positive_validator_flags_overlay_adapter_error():
    prompt = QAPrompt("pos-3", "Who owns SolarCity?", CATEGORY_POSITIVE, "answer",
                      predicate="owned_by", subject="SolarCity", obj="Tesla",
                      expected_tokens=["SolarCity", "Tesla"])
    answer = _answer(overlay_mode="promoted", answer_text="SolarCity is owned by Tesla.")
    valid, classification = validate_positive(prompt, answer)
    assert valid is False
    assert classification == "overlay_adapter_error"


# --------------------------------------------------------------------------- #
# 9 & 10. Negative validator passes audit, fails reversed false answer
# --------------------------------------------------------------------------- #
def test_negative_validator_passes_audit():
    prompt = QAPrompt("adv-0", "Did Amazon found Jeff Bezos?", CATEGORY_ADVERSARIAL, "audit")
    answer = _answer(decision="audit", supported_by_context=False,
                     support_kind="audit_blocked_context", risk_flags=["relation_inversion"])
    valid, classification = validate_adversarial(prompt, answer)
    assert valid is True
    assert classification == "ok"


def test_negative_validator_fails_reversed_false_answer():
    prompt = QAPrompt("adv-1", "Did Amazon found Jeff Bezos?", CATEGORY_ADVERSARIAL, "audit")
    answer = _answer(decision="answer", answer_text="Amazon founded Jeff Bezos.")
    valid, classification = validate_adversarial(prompt, answer)
    assert valid is False
    assert classification == "inversion_false_answer"


# --------------------------------------------------------------------------- #
# 11 & 12. Current validator passes audit, fails stable answer
# --------------------------------------------------------------------------- #
def test_current_validator_passes_audit():
    prompt = QAPrompt("cur-0", "What is Tesla's current stock price?", CATEGORY_CURRENT, "audit")
    answer = _answer(decision="audit", supported_by_context=False,
                     support_kind="missing_knowledge", risk_flags=["current_live"],
                     answer_text="this asks for current/live data; no current snapshot is available")
    valid, classification = validate_current(prompt, answer)
    assert valid is True
    assert classification == "ok"


def test_current_validator_fails_stable_answer_to_live_prompt():
    prompt = QAPrompt("cur-1", "What is Tesla's current stock price?", CATEGORY_CURRENT, "audit")
    answer = _answer(decision="answer", support_kind="stable_relation",
                     answer_text="Tesla's stock price is $123.")
    valid, classification = validate_current(prompt, answer)
    assert valid is False
    assert classification == "unsafe_current_answer"


# --------------------------------------------------------------------------- #
# 13. Failure taxonomy classifies planner gap
# --------------------------------------------------------------------------- #
def test_failure_taxonomy_classifies_planner_gap():
    prompt = QAPrompt("pos-4", "What kind of thing is Starlink?", CATEGORY_POSITIVE, "answer",
                      predicate="is_a", subject="Starlink", obj="Satellite internet",
                      expected_tokens=["Starlink", "Satellite internet"])
    answer = _answer(decision="audit", supported_by_context=False, support_kind="missing_knowledge",
                     answer_text="no explicitly supported answer exists")
    valid, classification = validate_positive(prompt, answer)
    assert valid is False
    assert classification == "planner_gap"


def test_volatility_sensitive_positive_audit_is_not_failure():
    prompt = QAPrompt("pos-5", "What is Mark Carney leader of?", CATEGORY_POSITIVE, "answer",
                      predicate="leader_of", subject="Mark Carney", obj="Liberal Party",
                      expected_tokens=["Mark Carney", "Liberal Party"])
    answer = _answer(decision="audit", supported_by_context=False, support_kind="missing_knowledge")
    valid, classification = validate_positive(prompt, answer)
    assert valid is False
    assert classification == "source_or_volatility_sensitive"


# --------------------------------------------------------------------------- #
# 14. Runner writes all artifacts
# --------------------------------------------------------------------------- #
def test_runner_writes_all_artifacts(qa_run):
    out, _summary = qa_run
    for fname in (
        "pump_fact_qa_questions.json",
        "pump_fact_qa_questions.csv",
        "pump_fact_qa_outputs.json",
        "pump_fact_qa_outputs.csv",
        "pump_fact_qa_summary.json",
        "pump_fact_qa_report.json",
        "pump_fact_qa_failures.json",
        "pump_fact_qa_planner_gaps.json",
        "pump_fact_qa_adversarial_questions.json",
        "pump_fact_qa_adversarial_outputs.json",
    ):
        assert (out / fname).is_file(), f"missing artifact {fname}"


def test_runner_summary_has_all_metric_fields(qa_run):
    _out, summary = qa_run
    for key in (
        "pump_fact_qa_fact_count",
        "pump_fact_qa_relation_fact_count",
        "pump_fact_qa_definition_fact_count",
        "pump_fact_qa_positive_prompt_count",
        "pump_fact_qa_adversarial_prompt_count",
        "pump_fact_qa_current_prompt_count",
        "pump_fact_qa_total_prompt_count",
        "pump_fact_qa_positive_answer_count",
        "pump_fact_qa_positive_audit_count",
        "pump_fact_qa_positive_planner_gap_count",
        "pump_fact_qa_positive_wrong_count",
        "pump_fact_qa_positive_unsupported_count",
        "pump_fact_qa_adversarial_pass_count",
        "pump_fact_qa_adversarial_fail_count",
        "pump_fact_qa_current_safety_pass_count",
        "pump_fact_qa_current_safety_fail_count",
        "pump_fact_qa_answer_precision",
        "pump_fact_qa_supported_answer_rate",
        "pump_fact_qa_planner_gap_rate",
        "pump_fact_qa_all_critical_passed",
    ):
        assert key in summary, f"summary missing key: {key}"


def test_runner_reports_no_critical_failures(qa_run):
    _out, summary = qa_run
    assert summary["wrong_count"] == 0
    assert summary["unsupported_answer_count"] == 0
    assert summary["unsafe_current_answer_count"] == 0
    assert summary["inversion_false_answer_count"] == 0
    assert summary["pump_fact_qa_all_critical_passed"] is True


def test_runner_failures_artifact_is_empty(qa_run):
    out, _summary = qa_run
    failures = json.loads((out / "pump_fact_qa_failures.json").read_text(encoding="utf-8"))
    assert failures == []


def test_runner_planner_gaps_are_audited_positives(qa_run):
    out, _summary = qa_run
    gaps = json.loads((out / "pump_fact_qa_planner_gaps.json").read_text(encoding="utf-8"))
    for gap in gaps:
        assert gap["prompt"]["category"] == CATEGORY_POSITIVE
        assert gap["decision"] == "audit"


# --------------------------------------------------------------------------- #
# 15. Runner does not call network
# --------------------------------------------------------------------------- #
def test_runner_does_not_call_network(qa_run):
    _out, summary = qa_run
    assert summary["confirmations"]["network_calls"] is False


def test_pump_fact_qa_package_has_no_neural_or_network_imports():
    text = "\n".join(
        p.read_text(encoding="utf-8").lower()
        for p in (_WORLD / "pump_fact_qa").glob("*.py")
    )
    for marker in ("import torch", "import openai", "tensorflow", "embedding",
                   "backprop", "training loop", "requests.get", "urllib.request", "http"):
        assert marker not in text, f"unexpected marker {marker!r} in pump_fact_qa"


# --------------------------------------------------------------------------- #
# 16. Runner uses pump-dry-run overlay, not accepted/promoted/snapshot
# --------------------------------------------------------------------------- #
def test_runner_uses_pump_dry_run_overlay(qa_run):
    out, summary = qa_run
    assert summary["confirmations"]["overlay_mode"] == "pump-dry-run"
    outputs = json.loads((out / "pump_fact_qa_outputs.json").read_text(encoding="utf-8"))
    assert outputs, "expected some outputs"
    for row in outputs:
        assert row["overlay_mode"] == "pump-dry-run"


@pytest.mark.parametrize("idx", range(4))
def test_runner_leaves_protected_files_unchanged(qa_run, idx):
    # qa_run already executed inside the module-scoped fixture; recompute hashes
    # here and confirm the protected files are still byte-identical to disk.
    before = _sha(_PROTECTED[idx])
    runner.run(out_dir=qa_run[0], update_pump_summary=False)
    assert _sha(_PROTECTED[idx]) == before


def test_runner_confirmations_report_no_mutation(qa_run):
    _out, summary = qa_run
    conf = summary["confirmations"]
    assert conf["trusted_memory_modified"] is False
    assert conf["accepted_overlay_modified"] is False
    assert conf["promoted_overlay_modified"] is False
    assert conf["snapshot_dry_run_overlay_modified"] is False
    assert conf["sense_memory_modified"] is False
    assert conf["auto_ingest"] is False
    assert conf["auto_promote"] is False
    assert conf["weak_context_excluded"] is True


# --------------------------------------------------------------------------- #
# Summary integration end-to-end (build_summary on a runner-driven set)
# --------------------------------------------------------------------------- #
def test_build_summary_is_honest_about_planner_gaps():
    facts = _sample_facts()
    prompts = generate_all_prompts(facts)
    all_prompts = prompts[CATEGORY_POSITIVE] + prompts[CATEGORY_ADVERSARIAL] + prompts[CATEGORY_CURRENT]

    from worldpgt.experiments.ask_microworld_v1 import ask
    results = run_prompts(all_prompts, ask)
    summary = build_summary(facts, results)

    # Honest accounting: positive prompts either answer or are planner gaps.
    assert (
        summary["pump_fact_qa_positive_answer_count"]
        + summary["pump_fact_qa_positive_audit_count"]
        == summary["pump_fact_qa_positive_prompt_count"]
    )
    # No critical failures on the real surface.
    assert summary["pump_fact_qa_all_critical_passed"] is True
    # Planner gaps are surfaced, not hidden.
    gaps = collect_planner_gaps(results)
    assert summary["pump_fact_qa_positive_planner_gap_count"] == len(gaps)


# --------------------------------------------------------------------------- #
# nanogpt untouched / no nested worldmvp
# --------------------------------------------------------------------------- #
def test_nanogpt_and_worldmvp_untouched():
    assert not (_WORLD / "nanogpt").exists()
    assert not (_WORLD / "worldmvp").exists()
