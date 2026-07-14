from __future__ import annotations

import json
from pathlib import Path

from worldpgt.assistant_surface.types import AssistantAnswer
from worldpgt.knowledge_pump.audit_event_logger import log_audit_event
from worldpgt.knowledge_pump.open_web_feedback import (
    build_open_web_feedback_frontier,
    feedback_topics,
)


def test_feedback_frontier_uses_acquisition_gaps_but_excludes_policy_blocks(tmp_path: Path):
    audit_log = tmp_path / "audit.jsonl"
    audit_log.write_text(
        "\n".join([
            json.dumps({
                "timestamp": "2026-07-14T00:00:00+00:00",
                "question": "What does Nexerra-R1 enable?",
                "entity": "Nexerra-R1",
                "support_kind": "missing_knowledge",
                "reason": "no stable relation exists for this question",
                "source": "api_feedback",
                "temporal_mismatch": False,
            }),
            json.dumps({
                "timestamp": "2026-07-14T00:00:00+00:00",
                "question": "What is the current price?",
                "entity": "Example Corp",
                "support_kind": "audit_blocked_context",
                "reason": "asks for current/live data",
                "source": "api_feedback",
                "temporal_mismatch": False,
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps([
        {"evidence_quality": {"issues": ["object_starts_as_discourse_fragment"]}},
    ]), encoding="utf-8")

    frontier = build_open_web_feedback_frontier(
        output_path=tmp_path / "frontier.json",
        audit_log_path=audit_log,
        review_paths=[review_path],
        period_days=0,
    )

    assert frontier["query_count"] == 1
    assert frontier["queries"][0]["query"] == "Nexerra-R1"
    assert frontier["policy_blocked"][0]["entity"] == "Example Corp"
    assert frontier["review"] == {
        "review_relation_count": 1,
        "review_issue_counts": {"object_starts_as_discourse_fragment": 1},
    }
    assert feedback_topics(frontier)[0].query == "Nexerra-R1"
    assert (tmp_path / "frontier.json").is_file()


def test_audit_logger_keeps_parser_entity_and_relation_hint_for_feedback(tmp_path: Path):
    answer = AssistantAnswer(
        question="What does it enable?", decision="audit", route="entity_relation",
        answer_text="no stable relation exists", overlay_mode="pump-dry-run",
        supported_by_context=False, support_kind="missing_knowledge",
    )
    log_path = tmp_path / "audit.jsonl"

    log_audit_event(
        answer, log_path=log_path, entity="Nexerra-R1", relation_hint="enables", source="api_feedback",
    )

    entry = json.loads(log_path.read_text(encoding="utf-8"))
    assert entry["entity"] == "Nexerra-R1"
    assert entry["relation_hint"] == "enables"
    assert entry["source"] == "api_feedback"
