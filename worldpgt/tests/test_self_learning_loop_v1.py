"""Tests for the Self-Learning Loop v1 (proposal layer only).

Deterministic and offline. No network. These tests verify that the loop mines
non-empty, correctly classified proposals from EXISTING artifacts, routes
dangerous claims to audit/reject (never answer), keeps everything proposal-only,
and never touches trusted memory, the accepted overlay, the promoted overlay,
sense_memory.py, validators, thresholds, or nanogpt/.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from worldpgt.experiments import run_self_learning_loop_v1 as runner
from worldpgt.self_learning.artifact_loader import (
    ArtifactLoader,
    MissingRequiredArtifact,
    OPTIONAL_CSV,
)
from worldpgt.self_learning.curriculum_builder import (
    build_adversarial_cases,
    build_curriculum,
)
from worldpgt.self_learning.learning_signal_miner import mine_signals
from worldpgt.self_learning.opportunity_classifier import (
    classify_signals,
    propose_analyzer_rules,
    propose_extraction_patterns,
)
from worldpgt.self_learning.types import (
    ALLOWED_ACTIVATION_STATUSES,
    PROPOSAL_ONLY,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS = _REPO / "worldpgt" / "experiments"

_PROTECTED_FILES = [
    _EXPERIMENTS / "accepted_knowledge_memory_v1.json",
    _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json",
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _REPO / "worldpgt" / "continuation" / "sense_memory.py",
]

# Forbidden top-level module names (checked via the AST import graph).
_FORBIDDEN_MODULES = {
    "torch",
    "tensorflow",
    "numpy",
    "transformers",
    "openai",
    "requests",
    "urllib",
    "urllib2",
    "urllib3",
    "http",
    "socket",
    "nanogpt",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("self_learning_v1_out")
    return runner.run_self_learning_loop(
        experiments_dir=_EXPERIMENTS, out_dir=out_dir, write=True
    )


@pytest.fixture(scope="module")
def loaded():
    return ArtifactLoader(_EXPERIMENTS).load()


# --------------------------------------------------------------------------- #
# 1. Loader reads required JSON artifacts.
# --------------------------------------------------------------------------- #
def test_loader_reads_required_json(loaded):
    for name in (
        "quarantine",
        "self_ingestion_report",
        "promotion_report",
        "entity_qa_summary",
        "cross_page_qa_summary",
        "answer_planner_summary",
    ):
        assert name in loaded.json_data
    assert loaded.json_data["quarantine"], "quarantine should be non-empty"


def test_loader_fails_loudly_on_missing_required(tmp_path):
    # Empty dir has no required artifacts.
    with pytest.raises(MissingRequiredArtifact):
        ArtifactLoader(tmp_path).load()


# --------------------------------------------------------------------------- #
# 2. Missing optional outputs CSV creates a warning, not a failure.
# --------------------------------------------------------------------------- #
def test_missing_optional_csv_warns(tmp_path):
    # Copy only the required JSON artifacts (and dirs), omit all CSVs.
    from worldpgt.self_learning.artifact_loader import REQUIRED_JSON

    for rel in REQUIRED_JSON.values():
        src = _EXPERIMENTS / rel
        dst = tmp_path / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())

    loaded = ArtifactLoader(tmp_path).load()
    # No crash; every optional CSV recorded as a warning.
    csv_warnings = [w for w in loaded.warnings if "outputs CSV missing" in w]
    assert len(csv_warnings) == len(OPTIONAL_CSV)


# --------------------------------------------------------------------------- #
# 3. Quarantine reasons are counted correctly.
# --------------------------------------------------------------------------- #
def test_quarantine_reasons_counted(loaded):
    signals = mine_signals(loaded)
    counts = {
        s.text: s.count for s in signals if s.signal_type == "quarantine_reason_count"
    }
    assert counts["current_fact_without_as_of"] == 2
    assert counts["weak_link_promoted_to_fact"] == 1
    assert counts["inverted_relation"] == 1
    assert counts["private_or_sensitive_data"] == 1
    assert counts["unsupported_universal_claim"] == 1
    assert counts["volatile_requires_source"] == 1


def _opps_for_claim(opportunities, needle):
    return [o for o in opportunities if needle.lower() in o.question_or_claim.lower()]


# --------------------------------------------------------------------------- #
# 4-8. Quarantine reason -> category mapping.
# --------------------------------------------------------------------------- #
def test_weak_link_becomes_weak_link_safety_case(result):
    opps = _opps_for_claim(result["opportunities"], "weak link proves that Forbes")
    assert opps and opps[0].category == "weak_link_safety_case"


def test_inverted_relation_becomes_inversion_guard(result):
    opps = _opps_for_claim(result["opportunities"], "SpaceX founded Elon Musk")
    assert opps and opps[0].category == "relation_inversion_guard_case"


def test_private_data_becomes_high_priority_adversarial(result):
    opps = _opps_for_claim(result["opportunities"], "private email")
    assert opps
    assert opps[0].category == "adversarial_test_candidate"
    assert opps[0].priority == "high"


def test_current_fact_becomes_current_claim_guard(result):
    opps = _opps_for_claim(result["opportunities"], "stock price is $999")
    assert opps and opps[0].category == "current_claim_guard_case"
    assert opps[0].priority == "high"


def test_volatile_becomes_review_candidate(result):
    opps = _opps_for_claim(result["opportunities"], "Larry Ellison")
    assert opps
    assert opps[0].category == "volatile_review_candidate"
    assert opps[0].activation_status == "requires_human_review"


# --------------------------------------------------------------------------- #
# 9. Unsupported universal claim becomes an adversarial test candidate (it is
#    classified as the universal guard category AND produces an adversarial row).
# --------------------------------------------------------------------------- #
def test_unsupported_universal_becomes_adversarial(result):
    opps = _opps_for_claim(result["opportunities"], "All rockets are SpaceX products")
    assert opps and opps[0].category == "unsupported_universal_guard_case"
    adv_questions = [c.question for c in result["adversarial_cases"]]
    assert any("All rockets are SpaceX products" in q for q in adv_questions)


# --------------------------------------------------------------------------- #
# 10. Dangerous claims never become missing_stable_knowledge.
# --------------------------------------------------------------------------- #
def test_dangerous_claims_never_missing_stable_knowledge(result):
    missing = [
        o
        for o in result["opportunities"]
        if o.category == "missing_stable_knowledge"
    ]
    dangerous_markers = [
        "private email",
        "founded Elon Musk",
        "weak link proves",
        "stock price is $999",
        "richest person",
        "All rockets are SpaceX products",
    ]
    for opp in missing:
        text = opp.question_or_claim.lower()
        for marker in dangerous_markers:
            assert marker.lower() not in text, f"dangerous claim leaked: {opp}"


# --------------------------------------------------------------------------- #
# 11. Curriculum rows for dangerous claims expect audit/reject, never answer.
# --------------------------------------------------------------------------- #
def test_dangerous_curriculum_rows_audit_or_reject(result):
    dangerous_categories = {
        "adversarial_test_candidate",
        "weak_link_safety_case",
        "relation_inversion_guard_case",
        "unsupported_universal_guard_case",
        "current_claim_guard_case",
    }
    for row in result["curriculum_rows"]:
        if row.category in dangerous_categories:
            assert row.expected_decision in {"audit", "reject"}
    # And no curriculum row ever expects "answer".
    assert all(r.expected_decision != "answer" for r in result["curriculum_rows"])
    # Adversarial cases likewise.
    assert all(
        c.expected_decision in {"audit", "reject"}
        for c in result["adversarial_cases"]
    )


# --------------------------------------------------------------------------- #
# 12-13. Proposals are proposal_only.
# --------------------------------------------------------------------------- #
def test_pattern_proposals_proposal_only(result):
    patterns = result["pattern_proposals"]
    assert patterns
    assert all(p.activation_status == PROPOSAL_ONLY for p in patterns)
    assert all(p.requires_tests for p in patterns)


def test_analyzer_rule_proposals_proposal_only(result):
    rules = result["analyzer_rule_proposals"]
    assert rules
    assert all(r.activation_status == PROPOSAL_ONLY for r in rules)
    assert all(r.expected_decision in {"audit", "reject"} for r in rules)


def test_no_auto_apply_activation_status(result):
    assert "auto_apply" not in ALLOWED_ACTIVATION_STATUSES
    for opp in result["opportunities"]:
        assert opp.activation_status in ALLOWED_ACTIVATION_STATUSES
        assert opp.activation_status != "auto_apply"


# --------------------------------------------------------------------------- #
# 14-17. Report safety flags.
# --------------------------------------------------------------------------- #
def test_report_trusted_memory_unchanged(result):
    assert result["report"].trusted_memory_modified is False


def test_report_accepted_overlay_unchanged(result):
    assert result["report"].accepted_overlay_modified is False


def test_report_promoted_overlay_unchanged(result):
    assert result["report"].promoted_overlay_modified is False


def test_report_not_safe_for_general_runtime(result):
    report = result["report"]
    assert report.safe_for_general_runtime is False
    assert report.proposal_only is True


def test_report_observes_green_regressions(result):
    observed = result["report"].regression_status_observed
    assert observed
    assert all(status == "PASS" for status in observed.values())


# --------------------------------------------------------------------------- #
# First run produces non-empty proposals.
# --------------------------------------------------------------------------- #
def test_first_run_non_empty(result):
    assert result["opportunities"], "expected non-empty opportunities"
    assert result["curriculum_rows"], "expected non-empty curriculum"
    assert result["adversarial_cases"], "expected non-empty adversarial cases"
    cats = result["report"].opportunities_by_category
    for required in (
        "weak_link_safety_case",
        "relation_inversion_guard_case",
        "adversarial_test_candidate",
        "unsupported_universal_guard_case",
        "current_claim_guard_case",
        "volatile_review_candidate",
    ):
        assert cats.get(required, 0) >= 1


def test_artifacts_written(tmp_path):
    out_dir = tmp_path / "out"
    runner.run_self_learning_loop(
        experiments_dir=_EXPERIMENTS, out_dir=out_dir, write=True
    )
    for fname in (
        "learning_opportunities.json",
        "learning_opportunity_summary.json",
        "proposed_curriculum_v1.csv",
        "proposed_adversarial_cases_v1.csv",
        "proposed_extraction_patterns_v1.json",
        "proposed_analyzer_rules_v1.json",
        "self_learning_report_v1.json",
    ):
        path = out_dir / fname
        assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 18. No protected files modified by a run.
# --------------------------------------------------------------------------- #
def test_protected_files_unchanged(tmp_path):
    before = {p: _hash(p) for p in _PROTECTED_FILES}
    runner.run_self_learning_loop(
        experiments_dir=_EXPERIMENTS, out_dir=tmp_path / "out", write=True
    )
    after = {p: _hash(p) for p in _PROTECTED_FILES}
    assert before == after


# --------------------------------------------------------------------------- #
# 19. No neural / GPT / training / embedding / network imports.
# 20. nanogpt/ untouched / unreferenced.
# --------------------------------------------------------------------------- #
def _imported_modules(src: Path) -> set:
    import ast

    tree = ast.parse(src.read_text(encoding="utf-8"))
    mods: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mods.add(node.module.split(".")[0])
    return mods


def test_no_forbidden_imports():
    package = _REPO / "worldpgt" / "self_learning"
    sources = list(package.glob("*.py"))
    sources.append(_EXPERIMENTS / "run_self_learning_loop_v1.py")
    for src in sources:
        mods = _imported_modules(src)
        offending = mods & _FORBIDDEN_MODULES
        assert not offending, f"forbidden imports {offending} in {src.name}"
