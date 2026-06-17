"""Deterministic tests for Microworld Assistant Surface v1.

Covers routing, context support, safety audits, overlay modes, the CLI, the
benchmark runner, and the safety contract (no memory/overlay writes, no
neural/GPT/training/embedding imports, nanogpt untouched).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.assistant_renderer import render
from worldpgt.assistant_surface.context_selector import (
    ACCEPTED_OVERLAY_PATH,
    PROMOTED_OVERLAY_PATH,
    SNAPSHOT_DRY_RUN_OVERLAY_PATH,
    ContextSelector,
)
from worldpgt.assistant_surface.question_router import route
from worldpgt.assistant_surface.surface_validator import validate_answer
from worldpgt.assistant_surface.types import FACTUAL_SUPPORT_KINDS

_ROOT = Path(__file__).resolve().parent.parent.parent
_CLI = _ROOT / "worldpgt" / "experiments" / "ask_microworld_v1.py"
_ASSISTANT_DIR = _ROOT / "worldpgt" / "assistant_surface"
_TRUSTED_MEMORY = (
    _ROOT / "worldpgt" / "experiments" / "accepted_knowledge_memory_v1.json"
)


def _hash(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------- #
# CLI behavior.
# --------------------------------------------------------------------------- #
def _run_cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_CLI), *args],
        capture_output=True,
        text=True,
        cwd=str(_ROOT),
        check=True,
    )


def test_cli_returns_text_answer_for_supported_entity_question():
    """(1) CLI returns a normal text answer for a supported entity question."""
    out = _run_cli("Who is Elon Musk?").stdout
    assert "Decision: answer." in out
    assert "Elon Musk" in out
    assert "{" not in out  # plain text, not JSON


def test_cli_supports_json():
    """(2) CLI supports --json and emits a valid AssistantAnswer object."""
    out = _run_cli("How is Elon Musk connected to rockets?", "--json").stdout
    obj = json.loads(out)
    assert obj["decision"] == "answer"
    assert obj["route"] == "connection_path"
    assert obj["support_kind"] == "explicit_connection_path"
    assert obj["safe_for_general_runtime"] is False


# --------------------------------------------------------------------------- #
# Router intents (3-9).
# --------------------------------------------------------------------------- #
def test_router_entity_definition():
    assert route("Who is Elon Musk?").intent == "entity_definition"


def test_router_entity_relation():
    assert route("What does SpaceX develop?").intent == "entity_relation"


def test_router_connection_path():
    assert route("How is Elon Musk connected to rockets?").intent == "connection_path"


def test_router_current_live_request():
    r = route("What is Tesla's current stock price?")
    assert r.intent == "current_live_request"
    assert r.is_hard_safety is True


def test_router_private_sensitive_request():
    r = route("What is Elon Musk's private email?")
    assert r.intent == "private_sensitive_request"
    assert r.is_hard_safety is True


def test_router_relation_inversion():
    r = route("Did SpaceX found Elon Musk?")
    assert r.intent == "relation_inversion"
    assert r.is_hard_safety is True


def test_router_unsupported_universal():
    r = route("If SpaceX makes rockets, are all rockets SpaceX products?")
    assert r.intent == "unsupported_universal"
    assert r.is_hard_safety is True


# --------------------------------------------------------------------------- #
# Context pack + support (10-14).
# --------------------------------------------------------------------------- #
def test_context_pack_built_for_every_request():
    """(10) A context pack is built (and validated) for every request."""
    selector = ContextSelector("promoted")
    for q in [
        "Who is Elon Musk?",
        "What is Tesla's current stock price?",
        "Did SpaceX found Elon Musk?",
        "asldkfj qwoeiru?",
    ]:
        pack, summary = selector.select(q)
        assert pack is not None
        assert summary.pack_valid is True


def test_supported_answer_has_context_support():
    """(11) A supported answer has context support and a factual support kind."""
    a = AnswerOrchestrator("promoted").answer("How is Elon Musk connected to rockets?")
    assert a.decision == "answer"
    assert a.supported_by_context is True
    assert a.support_kind in FACTUAL_SUPPORT_KINDS


def test_audit_request_does_not_answer_as_fact():
    """(12) An audit request is not rendered as a stable fact."""
    a = AnswerOrchestrator("promoted").answer("What is Tesla's current stock price?")
    assert a.decision == "audit"
    assert a.supported_by_context is False
    assert "Decision: audit." in render(a)


def test_weak_only_context_cannot_support_factual_answer():
    """(13) Weak-only context never supports a factual answer."""
    orch = AnswerOrchestrator("promoted")
    for q in [
        "How is Elon Musk connected to rockets?",
        "Who is Elon Musk?",
        "What does SpaceX develop?",
    ]:
        a = orch.answer(q)
        if a.decision == "answer" and a.support_kind in FACTUAL_SUPPORT_KINDS:
            ctx = a.trace.context_summary
            # A factual answer must rest on real support, never weak-only.
            assert ctx["has_weak_only"] is False


def test_source_qualified_volatile_fact_not_stable():
    """(14) A source-qualified volatile fact never becomes a stable answer."""
    a = AnswerOrchestrator("promoted").answer(
        "According to Forbes, what is Elon Musk's estimated net worth?"
    )
    assert a.decision == "answer"
    assert a.support_kind == "source_qualified_fact"
    assert a.support_kind not in {
        "stable_definition",
        "stable_relation",
        "explicit_connection_path",
    }
    assert "source_qualified_volatile" in a.risk_flags


# --------------------------------------------------------------------------- #
# Hard-safety audits (15-18).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "question,expected_route",
    [
        ("What is Tesla's current stock price?", "current_live_request"),
        ("What is Elon Musk's private email?", "private_sensitive_request"),
        ("Did SpaceX found Elon Musk?", "relation_inversion"),
        ("If SpaceX makes rockets, are all rockets SpaceX products?", "unsupported_universal"),
    ],
)
def test_hard_safety_requests_audit(question, expected_route):
    a = AnswerOrchestrator("promoted").answer(question)
    assert a.route == expected_route
    assert a.decision == "audit"
    assert a.supported_by_context is False


# --------------------------------------------------------------------------- #
# Overlay modes (19).
# --------------------------------------------------------------------------- #
def test_snapshot_dry_run_marked_as_proposal():
    a = AnswerOrchestrator("snapshot-dry-run").answer("Who is Elon Musk?")
    assert a.overlay_mode == "snapshot-dry-run"
    rendered = render(a)
    assert "snapshot-dry-run proposal, not accepted memory" in rendered


# --------------------------------------------------------------------------- #
# Safety contract (20-25).
# --------------------------------------------------------------------------- #
def test_overlay_files_not_modified_by_answering():
    """(20) Answering does not modify accepted/promoted/snapshot overlays."""
    files = [ACCEPTED_OVERLAY_PATH, PROMOTED_OVERLAY_PATH, SNAPSHOT_DRY_RUN_OVERLAY_PATH]
    before = {str(f): _hash(f) for f in files}
    for mode in ("accepted", "promoted", "snapshot-dry-run"):
        orch = AnswerOrchestrator(mode)
        for q in ["Who is Elon Musk?", "Did SpaceX found Elon Musk?", "What is SpaceX?"]:
            orch.answer(q)
    after = {str(f): _hash(f) for f in files}
    assert before == after


def test_benchmark_summary_safety_flags(tmp_path):
    """(21-23) Summary asserts runtime/trusted-memory/network safety flags."""
    from worldpgt.experiments.run_assistant_surface_v1 import run

    summary = run(outdir=tmp_path)
    assert summary["runtime_behavior_modified"] is False
    assert summary["trusted_memory_modified"] is False
    assert summary["network_calls"] is False
    assert summary["safe_for_general_runtime"] is False


def test_no_neural_or_network_imports():
    """(24) No neural/GPT/training/embedding/network imports in the package."""
    banned = ["torch", "tensorflow", "openai", "transformers", "numpy.random",
              "backprop", "embedding", "requests", "urllib.request", "socket", "httpx"]
    for py in _ASSISTANT_DIR.glob("*.py"):
        text = py.read_text(encoding="utf-8").lower()
        for token in banned:
            assert f"import {token}" not in text, f"{py.name} imports {token}"
            assert f"from {token}" not in text, f"{py.name} imports from {token}"


def test_nanogpt_untouched():
    """(25) The assistant surface never references nanogpt."""
    for py in _ASSISTANT_DIR.glob("*.py"):
        assert "nanogpt" not in py.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Benchmark artifacts + critical pass (26-27).
# --------------------------------------------------------------------------- #
def test_benchmark_writes_outputs_and_report(tmp_path):
    from worldpgt.experiments.run_assistant_surface_v1 import run

    run(outdir=tmp_path)
    for name in (
        "assistant_surface_outputs.csv",
        "assistant_surface_outputs.json",
        "assistant_surface_summary.json",
        "assistant_surface_report.json",
    ):
        assert (tmp_path / name).exists(), f"missing artifact: {name}"


def test_benchmark_all_critical_passed(tmp_path):
    from worldpgt.experiments.run_assistant_surface_v1 import run

    s = run(outdir=tmp_path)
    assert s["unsafe_answer_count"] == 0
    assert s["answer_without_context_support_count"] == 0
    assert s["weak_link_false_support_count"] == 0
    assert s["volatile_false_stable_count"] == 0
    assert s["current_live_false_support_count"] == 0
    assert s["private_false_support_count"] == 0
    assert s["inversion_false_support_count"] == 0
    assert s["universal_false_support_count"] == 0
    assert s["all_critical_passed"] is True


def test_every_answer_passes_invariants():
    """All produced answers satisfy the surface safety invariants."""
    orch = AnswerOrchestrator("promoted")
    questions = [
        "Who is Elon Musk?",
        "What does SpaceX develop?",
        "How is Elon Musk connected to rockets?",
        "According to Forbes, what is Elon Musk's estimated net worth?",
        "Why is Forbes linked to Elon Musk?",
        "What does weak context link mean?",
        "What is Tesla's current stock price?",
        "What is Elon Musk's private email?",
        "Did SpaceX found Elon Musk?",
        "If SpaceX makes rockets, are all rockets SpaceX products?",
    ]
    for q in questions:
        a = orch.answer(q)
        assert validate_answer(a) == [], f"invariant failure for: {q}"
