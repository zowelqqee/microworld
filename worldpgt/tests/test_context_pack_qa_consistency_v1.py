"""Tests for the Context-Pack QA Consistency Gate v1 (dry-run).

Deterministic and offline. No network. Confirms the gate aligns QA answers with
explicit context support, accepts safe-policy answers, flags audit-despite-safe
as a warning, fails synthetic false-support / validation rows, and never touches
protected files or runtime behavior.

Synthetic packs are constructed to exercise the false-support paths (the real
pipeline audits dangerous questions, so we build the unsafe answer to prove the
gate would catch it).
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from worldpgt.experiments import run_context_pack_qa_consistency_v1 as runner
from worldpgt.context_consistency.context_consistency_checker import (
    check_row,
    is_safe_policy_answer,
)
from worldpgt.context_consistency.qa_result_loader import load_qa_results
from worldpgt.context_consistency.types import (
    LoadedQAResult,
    STATUS_ANSWER_SUPPORTED,
    STATUS_ANSWER_WITHOUT_SUPPORT,
    STATUS_AUDIT_DESPITE_SAFE,
    STATUS_AUDIT_SUPPORTED,
    STATUS_CURRENT_LIVE_FALSE,
    STATUS_INVERSION_FALSE,
    STATUS_PRIVATE_FALSE,
    STATUS_SAFE_POLICY_ANSWER,
    STATUS_UNIVERSAL_FALSE,
    STATUS_VALIDATION_FAILED,
    STATUS_VOLATILE_FALSE,
    STATUS_WEAK_LINK_FALSE,
)
from worldpgt.context_pack.context_pack_validator import (
    ContextPackValidationResult,
    validate_pack,
)
from worldpgt.context_pack.types import (
    OVERLAY_ACCEPTED,
    ContextBlockedPath,
    ContextCandidatePath,
    ContextDefinition,
    ContextRelation,
    ContextSourceFact,
    WorkingContextPack,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS = _REPO / "worldpgt" / "experiments"

_PROTECTED_FILES = [
    _EXPERIMENTS / "accepted_knowledge_memory_v1.json",
    _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json",
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _REPO / "worldpgt" / "continuation" / "sense_memory.py",
]

_FORBIDDEN_MODULES = {
    "torch", "tensorflow", "numpy", "transformers", "openai",
    "requests", "urllib", "urllib2", "urllib3", "http", "socket", "nanogpt",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ok_validation():
    return ContextPackValidationResult(passed=True, failures=[], checks={})


def _qa(decision, question="q", answer="", intent="x", suite="s", _id="t"):
    return LoadedQAResult(
        id=_id, source_suite=suite, question=question, qa_decision=decision,
        qa_intent=intent, qa_answer=answer,
    )


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("context_consistency_out")
    return runner.run_consistency_gate(
        experiments_dir=_EXPERIMENTS, overlay_mode=OVERLAY_ACCEPTED, out_dir=out_dir, write=True
    )


# --------------------------------------------------------------------------- #
# 1-2. Loader.
# --------------------------------------------------------------------------- #
def test_loader_finds_qa_outputs_even_if_renamed(tmp_path):
    # A custom-named entity outputs file is still discovered by glob.
    src = (_EXPERIMENTS / "entity_qa_v1_outputs.csv").read_bytes()
    (tmp_path / "my_entity_v1_outputs_custom.csv").write_bytes(src)
    loaded = load_qa_results(tmp_path)
    assert loaded.rows
    assert any(r.source_suite == "entity_qa_v1" for r in loaded.rows)


def test_missing_optional_output_warns_with_summary(tmp_path):
    # Summary present but no matching outputs CSV -> warning, not crash.
    src = (_EXPERIMENTS / "entity_qa_v1_summary.json").read_bytes()
    (tmp_path / "entity_qa_v1_summary.json").write_bytes(src)
    loaded = load_qa_results(tmp_path)
    assert any("QA outputs missing" in w for w in loaded.warnings)
    assert "entity_qa_v1" in loaded.summaries


# --------------------------------------------------------------------------- #
# 3-7. Core decision alignment.
# --------------------------------------------------------------------------- #
def test_answer_with_explicit_path_passes():
    pack = WorkingContextPack(question="q", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=True)
    pack.candidate_paths.append(
        ContextCandidatePath(nodes=["A", "B", "C"], relations=["A->B"], support="semi_stable")
    )
    res = check_row(_qa("answer", answer="A is connected to C via B."), pack, _ok_validation())
    assert res.consistency_status == STATUS_ANSWER_SUPPORTED


def test_audit_with_blocked_context_passes():
    pack = WorkingContextPack(question="q", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    pack.blocked_paths.append(ContextBlockedPath(nodes=["A", "B"], reason="weak_only_path", detail="x"))
    res = check_row(_qa("audit"), pack, _ok_validation())
    assert res.consistency_status == STATUS_AUDIT_SUPPORTED


def test_safe_policy_answer_about_weak_links_passes():
    assert is_safe_policy_answer(
        "A weak context link is not treated as a stable factual relation."
    )
    pack = WorkingContextPack(question="Are weak links facts?", overlay_mode=OVERLAY_ACCEPTED)
    pack.blocked_paths.append(ContextBlockedPath(nodes=["A", "B"], reason="weak_only_path", detail="x"))
    res = check_row(
        _qa("answer", answer="No. A weak context link is not a stable factual relation."),
        pack,
        _ok_validation(),
    )
    assert res.consistency_status == STATUS_SAFE_POLICY_ANSWER


def test_answer_without_context_support_fails():
    pack = WorkingContextPack(question="q", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    res = check_row(_qa("answer", answer="Some unsupported factual assertion."), pack, _ok_validation())
    assert res.consistency_status == STATUS_ANSWER_WITHOUT_SUPPORT


def test_audit_despite_safe_context_is_warning():
    pack = WorkingContextPack(question="q", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=True)
    pack.candidate_paths.append(
        ContextCandidatePath(nodes=["A", "B"], relations=["A->B"], support="semi_stable")
    )
    res = check_row(_qa("audit"), pack, _ok_validation())
    assert res.consistency_status == STATUS_AUDIT_DESPITE_SAFE


# --------------------------------------------------------------------------- #
# 8-13. False-support guards.
# --------------------------------------------------------------------------- #
def test_weak_only_cannot_support_factual_answer():
    pack = WorkingContextPack(question="q", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    pack.blocked_paths.append(ContextBlockedPath(nodes=["Forbes", "Elon Musk"], reason="weak_only_path", detail="x"))
    res = check_row(_qa("answer", answer="Forbes owns Elon Musk."), pack, _ok_validation())
    assert res.consistency_status == STATUS_WEAK_LINK_FALSE


def test_volatile_cannot_support_stable_answer():
    pack = WorkingContextPack(question="q", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    pack.source_facts.append(
        ContextSourceFact(
            subject="Elon Musk", predicate="estimated_net_worth", object="US$1.1 trillion",
            source_name="Forbes", as_of="2026-06", claim_type="estimate",
            stability="volatile", risk="high", trust="source_qualified_overlay_candidate",
            requires_recheck=True,
        )
    )
    res = check_row(_qa("answer", answer="Elon Musk's net worth is US$1.1 trillion."), pack, _ok_validation())
    assert res.consistency_status == STATUS_VOLATILE_FALSE


def test_current_live_cannot_support_answer():
    pack = WorkingContextPack(question="What is Tesla's current stock price?", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    pack.safety_notes.append("current_or_live_question: requires explicit source-qualified fact")
    res = check_row(_qa("answer", question=pack.question, answer="Tesla stock price is $999 today."), pack, _ok_validation())
    assert res.consistency_status == STATUS_CURRENT_LIVE_FALSE


def test_inversion_cannot_support_answer():
    pack = WorkingContextPack(question="Did SpaceX found Elon Musk?", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    pack.blocked_paths.append(ContextBlockedPath(nodes=["SpaceX", "Elon Musk"], reason="relation_inversion", detail="x"))
    res = check_row(_qa("answer", question=pack.question, answer="Yes, SpaceX founded Elon Musk."), pack, _ok_validation())
    assert res.consistency_status == STATUS_INVERSION_FALSE


def test_private_cannot_support_answer():
    pack = WorkingContextPack(question="What is Elon Musk's home address?", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    pack.safety_notes.append("private_or_sensitive_question: no factual context produced")
    res = check_row(_qa("answer", question=pack.question, answer="It is 123 Main St."), pack, _ok_validation())
    assert res.consistency_status == STATUS_PRIVATE_FALSE


def test_universal_cannot_support_answer():
    pack = WorkingContextPack(question="All rockets are SpaceX products.", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=False)
    pack.blocked_paths.append(ContextBlockedPath(nodes=["rockets", "SpaceX"], reason="unsupported_universal_claim", detail="x"))
    res = check_row(_qa("answer", question=pack.question, answer="Yes, all rockets are SpaceX products."), pack, _ok_validation())
    assert res.consistency_status == STATUS_UNIVERSAL_FALSE


# --------------------------------------------------------------------------- #
# 14. Validation failure fails the row.
# --------------------------------------------------------------------------- #
def test_context_validation_failure_fails_row():
    pack = WorkingContextPack(question="q", overlay_mode=OVERLAY_ACCEPTED, safe_for_answering=True)
    bad_validation = ContextPackValidationResult(passed=False, failures=["weak link present"], checks={})
    res = check_row(_qa("answer", answer="x"), pack, bad_validation)
    assert res.consistency_status == STATUS_VALIDATION_FAILED


# --------------------------------------------------------------------------- #
# 15-20. Summary on the real run.
# --------------------------------------------------------------------------- #
def test_summary_all_critical_passed(result):
    s = result["summary"]
    assert s["all_critical_passed"] is True
    assert s["answer_without_context_support"] == 0
    assert s["weak_link_false_support_count"] == 0
    assert s["volatile_false_stable_count"] == 0
    assert s["current_live_false_support_count"] == 0
    assert s["inversion_false_support_count"] == 0
    assert s["private_false_support_count"] == 0
    assert s["unsupported_universal_false_support_count"] == 0


def test_summary_safe_for_general_runtime_false(result):
    assert result["summary"]["safe_for_general_runtime"] is False


def test_summary_runtime_behavior_unmodified(result):
    assert result["summary"]["runtime_behavior_modified"] is False
    assert result["summary"]["network_calls"] is False


def test_summary_trusted_memory_unchanged(result):
    assert result["summary"]["trusted_memory_modified"] is False


def test_summary_accepted_overlay_unchanged(result):
    assert result["summary"]["accepted_overlay_modified"] is False


def test_summary_promoted_overlay_unchanged(result):
    assert result["summary"]["promoted_overlay_modified"] is False


def test_real_run_checks_all_qa_rows(result):
    s = result["summary"]
    assert s["qa_rows_total"] > 0
    assert s["checked_rows_total"] == s["qa_rows_total"]
    assert s["answer_rows"] + s["audit_rows"] == s["checked_rows_total"]


def test_artifacts_written(tmp_path):
    out_dir = tmp_path / "out"
    runner.run_consistency_gate(experiments_dir=_EXPERIMENTS, out_dir=out_dir, write=True)
    for fname in (
        "context_consistency_outputs.csv",
        "context_consistency_outputs.json",
        "context_consistency_summary.json",
        "context_consistency_report.json",
    ):
        path = out_dir / fname
        assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 21. No protected files modified.
# --------------------------------------------------------------------------- #
def test_protected_files_unchanged(tmp_path):
    before = {p: _hash(p) for p in _PROTECTED_FILES}
    runner.run_consistency_gate(experiments_dir=_EXPERIMENTS, out_dir=tmp_path / "out", write=True)
    after = {p: _hash(p) for p in _PROTECTED_FILES}
    assert before == after


# --------------------------------------------------------------------------- #
# 22-23. No forbidden imports.
# --------------------------------------------------------------------------- #
def _imported_modules(src: Path) -> set:
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
    package = _REPO / "worldpgt" / "context_consistency"
    sources = list(package.glob("*.py"))
    sources.append(_EXPERIMENTS / "run_context_pack_qa_consistency_v1.py")
    for src in sources:
        offending = _imported_modules(src) & _FORBIDDEN_MODULES
        assert not offending, f"forbidden imports {offending} in {src.name}"
