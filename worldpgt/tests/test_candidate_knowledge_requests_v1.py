"""Tests for Candidate Knowledge Request v1 (request layer only).

Deterministic and offline. No network. Confirms that only
missing_stable_knowledge becomes requests, dangerous/unsafe content is skipped
or made high-risk source-qualified, weak links stay weak, dedup + grouping work,
everything is request-only, and no protected file / runtime behavior is touched.
"""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import pytest

from worldpgt.experiments import run_candidate_knowledge_requests_v1 as runner
from worldpgt.knowledge_requests.missing_knowledge_loader import (
    MissingKnowledgeLoader,
    MissingRequiredInput,
    normalize_missing_inputs,
)
from worldpgt.knowledge_requests.request_classifier import classify_requests
from worldpgt.knowledge_requests.source_request_builder import (
    build_source_document_requests,
)
from worldpgt.knowledge_requests.types import (
    ALLOWED_ACTIVATION_STATUSES,
    MissingKnowledgeInput,
)

_REPO = Path(__file__).resolve().parent.parent.parent
_EXPERIMENTS = _REPO / "worldpgt" / "experiments"

_PROTECTED_FILES = [
    _EXPERIMENTS / "accepted_knowledge_memory_v1.json",
    _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json",
    _EXPERIMENTS
    / "self_ingestion_v1"
    / "promotion"
    / "promoted_wiki_memory_overlay_v1.json",
    _REPO / "worldpgt" / "continuation" / "sense_memory.py",
]

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

_FORBIDDEN_STATUSES = {
    "auto_apply",
    "auto_ingest",
    "auto_promote",
    "trusted_memory_write",
}


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("knowledge_requests_out")
    return runner.run_candidate_knowledge_requests(
        experiments_dir=_EXPERIMENTS, out_dir=out_dir, write=True
    )


@pytest.fixture(scope="module")
def loaded():
    return MissingKnowledgeLoader(_EXPERIMENTS).load()


def _mk(category, question, opp_id="slo-x"):
    return MissingKnowledgeInput(
        id=opp_id,
        category=category,
        question=question,
        current_decision="audit",
        current_behavior="audits safely",
        source_artifact="x",
    )


# --------------------------------------------------------------------------- #
# 1-2. Loader reads inputs.
# --------------------------------------------------------------------------- #
def test_loader_reads_learning_opportunities(loaded):
    assert loaded.opportunities
    assert any(o.get("category") == "missing_stable_knowledge" for o in loaded.opportunities)


def test_loader_reads_regression_outputs(loaded):
    assert loaded.regression_outputs
    assert all("source_opportunity_id" in r for r in loaded.regression_outputs)


# --------------------------------------------------------------------------- #
# 3. Missing optional self-ingestion artifacts -> warnings, not crash.
#    Missing required opportunities file -> loud failure.
# --------------------------------------------------------------------------- #
def test_missing_optional_artifacts_warn(tmp_path):
    # Provide only the required learning_opportunities.json; omit everything else.
    sl = tmp_path / "self_learning_v1"
    sl.mkdir(parents=True)
    src = _EXPERIMENTS / "self_learning_v1" / "learning_opportunities.json"
    (sl / "learning_opportunities.json").write_bytes(src.read_bytes())

    loaded = MissingKnowledgeLoader(tmp_path).load()
    assert loaded.opportunities
    assert any("optional artifact missing" in w for w in loaded.warnings)


def test_missing_required_fails_loudly(tmp_path):
    with pytest.raises(MissingRequiredInput):
        MissingKnowledgeLoader(tmp_path).load()


# --------------------------------------------------------------------------- #
# 4. Only missing_stable_knowledge becomes a request by default.
# --------------------------------------------------------------------------- #
def test_only_missing_stable_knowledge_becomes_requests(result):
    inputs = result["missing_inputs"]
    assert inputs and all(i.category == "missing_stable_knowledge" for i in inputs)
    # Requests only ever come from missing_stable_knowledge inputs.
    src_ids = {i.id for i in inputs}
    assert all(r.source_opportunity_id in src_ids for r in result["requests"])


# --------------------------------------------------------------------------- #
# 5. Dangerous opportunity categories are skipped as safety notes.
# --------------------------------------------------------------------------- #
def test_dangerous_categories_skipped():
    inputs = [
        _mk("current_claim_guard_case", "What is Tesla's stock price today?", "d1"),
        _mk("relation_inversion_guard_case", "Did SpaceX found Elon Musk?", "d2"),
        _mk("weak_link_safety_case", "A weak link proves Forbes is owned by Musk.", "d3"),
    ]
    requests, skipped = classify_requests(inputs)
    assert requests == []
    assert len(skipped) == 3
    assert all(s.reason == "dangerous_opportunity_category" for s in skipped)


# --------------------------------------------------------------------------- #
# 6. Weak-link-related missing path remains audit-until-supported.
# --------------------------------------------------------------------------- #
def test_weak_link_related_path_remains_audit_until_supported():
    inputs = [_mk("missing_stable_knowledge", "How is Tesla connected to Forbes?")]
    requests, _ = classify_requests(inputs)
    assert len(requests) == 1
    req = requests[0]
    assert req.must_remain_audit_until_supported is True
    assert "do_not_infer_stable_relation_from_weak_context_links" in req.safety_constraints


# --------------------------------------------------------------------------- #
# 7. Current/live questions become high-risk (source-qualified), not low/stable.
# --------------------------------------------------------------------------- #
def test_current_live_question_is_high_risk():
    inputs = [_mk("missing_stable_knowledge", "What is Tesla's current stock price?")]
    requests, _ = classify_requests(inputs)
    assert len(requests) == 1
    req = requests[0]
    assert req.category == "missing_source_qualified_fact"
    assert req.risk_level == "high"
    assert req.must_remain_audit_until_supported is True


# --------------------------------------------------------------------------- #
# 8. Volatile/source-qualified needs are high-risk + require review/source.
# --------------------------------------------------------------------------- #
def test_source_qualified_requires_review_and_source(result):
    sq = [r for r in result["requests"] if r.category == "missing_source_qualified_fact"]
    assert sq
    for r in sq:
        assert r.risk_level == "high"
        assert r.activation_status == "requires_quarantine_review"
        assert "require_explicit_as_of_and_source" in r.safety_constraints
        assert "never_render_as_stable_or_current_fact" in r.safety_constraints


# --------------------------------------------------------------------------- #
# 9. Deduplication works.
# --------------------------------------------------------------------------- #
def test_deduplication():
    inputs = [
        _mk("missing_stable_knowledge", "How is Tesla connected to NASA?", "a"),
        _mk("missing_stable_knowledge", "How is Tesla connected to NASA?", "b"),
    ]
    requests, _ = classify_requests(inputs)
    assert len(requests) == 1


# --------------------------------------------------------------------------- #
# 10-11. Source document request groups related requests and lists claims_to_avoid.
# --------------------------------------------------------------------------- #
def test_source_doc_groups_related_requests():
    inputs = [
        _mk("missing_stable_knowledge", "How is Elon Musk connected to Starlink?", "a"),
        _mk("missing_stable_knowledge", "What path connects Elon Musk to Starlink?", "b"),
    ]
    requests, _ = classify_requests(inputs)
    docs = build_source_document_requests(requests)
    assert len(requests) == 2
    grouped = [d for d in docs if len(d.related_request_ids) > 1]
    assert grouped, "expected related requests grouped into one source doc"


def test_source_doc_includes_claims_to_avoid(result):
    docs = result["source_docs"]
    assert docs
    for d in docs:
        assert d.claims_to_avoid
        assert any("weak" in c.lower() for c in d.claims_to_avoid)


# --------------------------------------------------------------------------- #
# 12. Every request has an allowed activation_status (no auto_*).
# --------------------------------------------------------------------------- #
def test_requests_have_safe_activation_status(result):
    for r in result["requests"]:
        assert r.activation_status in ALLOWED_ACTIVATION_STATUSES
        assert r.activation_status not in _FORBIDDEN_STATUSES
    assert not (ALLOWED_ACTIVATION_STATUSES & _FORBIDDEN_STATUSES)


# --------------------------------------------------------------------------- #
# 13-18. Summary safety flags.
# --------------------------------------------------------------------------- #
def test_summary_request_only(result):
    assert result["summary"]["request_only"] is True


def test_summary_not_safe_for_general_runtime(result):
    assert result["summary"]["safe_for_general_runtime"] is False


def test_summary_trusted_memory_unchanged(result):
    assert result["summary"]["trusted_memory_modified"] is False


def test_summary_accepted_overlay_unchanged(result):
    assert result["summary"]["accepted_overlay_modified"] is False


def test_summary_promoted_overlay_unchanged(result):
    assert result["summary"]["promoted_overlay_modified"] is False


def test_summary_no_network_calls(result):
    assert result["summary"]["network_calls"] is False
    assert result["summary"]["auto_ingest"] is False
    assert result["summary"]["auto_promote"] is False


# --------------------------------------------------------------------------- #
# First run produces non-empty proposals.
# --------------------------------------------------------------------------- #
def test_first_run_non_empty(result):
    s = result["summary"]
    assert s["requests_total"] > 0
    assert s["source_document_requests_total"] > 0
    assert s["skipped_dangerous_inputs_total"] >= 1
    # No source-qualified / volatile request is ever low-risk.
    for r in result["requests"]:
        if r.category == "missing_source_qualified_fact":
            assert r.risk_level == "high"


def test_artifacts_written(tmp_path):
    out_dir = tmp_path / "out"
    runner.run_candidate_knowledge_requests(
        experiments_dir=_EXPERIMENTS, out_dir=out_dir, write=True
    )
    for fname in (
        "candidate_knowledge_requests.json",
        "candidate_knowledge_requests.csv",
        "source_document_requests.json",
        "source_document_requests.csv",
        "knowledge_request_summary.json",
        "knowledge_request_report.json",
    ):
        path = out_dir / fname
        assert path.is_file() and path.stat().st_size > 0


# --------------------------------------------------------------------------- #
# 19. No protected files modified by a run.
# --------------------------------------------------------------------------- #
def test_protected_files_unchanged(tmp_path):
    before = {p: _hash(p) for p in _PROTECTED_FILES}
    runner.run_candidate_knowledge_requests(
        experiments_dir=_EXPERIMENTS, out_dir=tmp_path / "out", write=True
    )
    after = {p: _hash(p) for p in _PROTECTED_FILES}
    assert before == after


# --------------------------------------------------------------------------- #
# 20-21. No forbidden imports; nanogpt untouched / unreferenced.
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
    package = _REPO / "worldpgt" / "knowledge_requests"
    sources = list(package.glob("*.py"))
    sources.append(_EXPERIMENTS / "run_candidate_knowledge_requests_v1.py")
    for src in sources:
        offending = _imported_modules(src) & _FORBIDDEN_MODULES
        assert not offending, f"forbidden imports {offending} in {src.name}"
