"""Tests for audit_event_logger — entity extraction, JSONL writes, guard logic."""

from __future__ import annotations

import json

import pytest

from worldpgt.assistant_surface.types import AssistantAnswer
from worldpgt.knowledge_pump.audit_event_logger import (
    _extract_entity,
    log_audit_event,
    log_multihop_audit_event,
)
from worldpgt.multihop_qa.assistant_adapter import AssistantMultihopResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _answer(
    question="What is SpaceX?",
    decision="audit",
    support_kind="missing_knowledge",
    answer_text="no fact found",
) -> AssistantAnswer:
    return AssistantAnswer(
        question=question,
        decision=decision,
        route="entity_definition",
        answer_text=answer_text,
        overlay_mode="promoted",
        supported_by_context=False,
        support_kind=support_kind,
    )


def _multihop(
    question="How is SpaceX connected to Elon Musk?",
    decision="audit",
    audit_reason="missing_hop",
) -> AssistantMultihopResult:
    return AssistantMultihopResult(
        question=question,
        decision=decision,
        answer_text="",
        audit_reason=audit_reason,
        chain_type="connection",
        chain_label="connection",
        hop1=None,
        hop2=None,
    )


# ---------------------------------------------------------------------------
# Entity extraction
# ---------------------------------------------------------------------------


class TestExtractEntity:
    def test_what_is(self):
        assert _extract_entity("What is SpaceX?") == "SpaceX"

    def test_who_is(self):
        assert _extract_entity("Who is Elon Musk?") == "Elon Musk"

    def test_what_does_develop(self):
        assert _extract_entity("What does SpaceX develop?") == "SpaceX"

    def test_how_connected(self):
        assert _extract_entity("How is Elon Musk connected to rockets?") == "Elon Musk"

    def test_case_insensitive(self):
        assert _extract_entity("what is Tesla?") == "Tesla"

    def test_unrecognized_returns_empty(self):
        assert _extract_entity("Tell me everything about everything") == ""

    def test_empty_string(self):
        assert _extract_entity("") == ""


# ---------------------------------------------------------------------------
# log_audit_event — AssistantAnswer
# ---------------------------------------------------------------------------


class TestLogAuditEvent:
    def test_writes_jsonl_on_audit(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_audit_event(_answer(), log_path=log)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["question"] == "What is SpaceX?"
        assert entry["entity"] == "SpaceX"
        assert entry["source"] == "assistant_surface"
        assert entry["support_kind"] == "missing_knowledge"
        assert entry["reason"] == "no fact found"
        assert entry["temporal_mismatch"] is False
        assert "timestamp" in entry

    def test_skips_non_audit_decision(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_audit_event(_answer(decision="answer", support_kind="stable_definition"), log_path=log)
        assert not log.exists()

    def test_appends_multiple_events(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        for _ in range(3):
            log_audit_event(_answer(), log_path=log)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 3

    def test_creates_parent_directories(self, tmp_path):
        log = tmp_path / "deep" / "nested" / "dir" / "audit.jsonl"
        log_audit_event(_answer(), log_path=log)
        assert log.exists()

    def test_policy_blocked_kind_recorded(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_audit_event(_answer(support_kind="audit_blocked_context", answer_text="policy block"), log_path=log)
        entry = json.loads(log.read_text().strip())
        assert entry["support_kind"] == "audit_blocked_context"

    def test_temporal_mismatch_recorded_from_reason(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_audit_event(_answer(answer_text="snapshot_requires_as_of:leader_of"), log_path=log)
        entry = json.loads(log.read_text().strip())
        assert entry["temporal_mismatch"] is True

    def test_empty_answer_text_fallback(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_audit_event(_answer(answer_text=""), log_path=log)
        entry = json.loads(log.read_text().strip())
        assert entry["reason"] == "no supported answer exists"


# ---------------------------------------------------------------------------
# log_multihop_audit_event — AssistantMultihopResult
# ---------------------------------------------------------------------------


class TestLogMultihopAuditEvent:
    def test_writes_on_audit(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_multihop_audit_event(_multihop(), log_path=log)
        entry = json.loads(log.read_text().strip())
        assert entry["source"] == "multihop"
        assert entry["support_kind"] == "multihop_audit"
        assert entry["entity"] == "SpaceX"
        assert entry["reason"] == "missing_hop"
        assert entry["temporal_mismatch"] is False

    def test_skips_non_audit(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_multihop_audit_event(_multihop(decision="answer"), log_path=log)
        assert not log.exists()

    def test_empty_audit_reason_fallback(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_multihop_audit_event(_multihop(audit_reason=""), log_path=log)
        entry = json.loads(log.read_text().strip())
        assert entry["reason"] == "no supported multi-hop relation chain found"

    def test_multihop_temporal_mismatch_recorded(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_multihop_audit_event(
            _multihop(audit_reason="path_blocked: snapshot_missing_as_of:owned_by"),
            log_path=log,
        )
        entry = json.loads(log.read_text().strip())
        assert entry["temporal_mismatch"] is True

    def test_both_sources_append_to_same_log(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        log_audit_event(_answer(), log_path=log)
        log_multihop_audit_event(_multihop(), log_path=log)
        lines = log.read_text().strip().splitlines()
        assert len(lines) == 2
        sources = {json.loads(l)["source"] for l in lines}
        assert sources == {"assistant_surface", "multihop"}
