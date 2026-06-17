"""Tests for Pump Promotion Readiness Audit v1.

The audit is local-only and read-only with respect to accepted memory and
accepted/promoted/snapshot overlays. It may write only audit artifacts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from worldpgt.experiments import run_pump_promotion_readiness_audit_v1 as audit

_REPO = Path(__file__).resolve().parent.parent.parent
_EXP = _REPO / "worldpgt" / "experiments"
_PUMP_DIR = _EXP / "knowledge_pump_v1"
_PRECISION = _PUMP_DIR / "pump_precision_answerable_delta.json"
_PROTECTED = [
    _EXP / "accepted_knowledge_memory_v1.json",
    _EXP / "accepted_wiki_memory_overlay_v1.json",
    _EXP / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json",
    _EXP / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json",
    _REPO / "worldpgt" / "continuation" / "sense_memory.py",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _qa_output(subject: str, predicate: str, obj: str) -> dict:
    return {
        "prompt": {
            "category": "positive",
            "subject": subject,
            "predicate": predicate,
            "obj": obj,
        },
        "classification": "ok",
    }


def _minimal_pump_dir(tmp_path: Path, facts: list[dict], *, qa_count: int | None = None) -> Path:
    pump_dir = tmp_path / "pump"
    qa_count = len([f for f in facts if f.get("overlay_type") in {"overlay_relation", "overlay_definition"}]) if qa_count is None else qa_count
    rel_count = len([f for f in facts if f.get("overlay_type") == "overlay_relation"])
    def_count = len([f for f in facts if f.get("overlay_type") == "overlay_definition"])
    qa_outputs = [
        _qa_output(
            str(f.get("subject", "")),
            str(f.get("predicate", "is_a")),
            str(f.get("object", f.get("definition", ""))),
        )
        for f in facts
        if f.get("overlay_type") in {"overlay_relation", "overlay_definition"}
    ]
    _write_json(pump_dir / "pump_precision_answerable_delta.json", facts)
    _write_json(pump_dir / "pump_fact_qa_v1" / "pump_fact_qa_outputs.json", qa_outputs)
    _write_json(
        pump_dir / "pump_fact_qa_v1" / "pump_fact_qa_summary.json",
        {
            "pump_fact_qa_fact_count": qa_count,
            "pump_fact_qa_relation_fact_count": rel_count,
            "pump_fact_qa_definition_fact_count": def_count,
            "pump_fact_qa_all_critical_passed": True,
            "pump_fact_qa_positive_wrong_count": 0,
            "pump_fact_qa_positive_unsupported_count": 0,
            "pump_fact_qa_adversarial_fail_count": 0,
            "pump_fact_qa_current_safety_fail_count": 0,
        },
    )
    _write_json(
        pump_dir / "pump_summary.json",
        {"pump_fact_qa_status": "current_from_qa_artifact"},
    )
    return pump_dir


@pytest.fixture(scope="module")
def real_audit(tmp_path_factory):
    if not _PRECISION.exists():
        pytest.skip("pump precision artifact not found")
    out_dir = tmp_path_factory.mktemp("promotion_readiness")
    summary = audit.run(pump_dir=_PUMP_DIR, out_dir=out_dir)
    return out_dir, summary


def test_audit_is_read_only_for_input_artifact(tmp_path) -> None:
    before = _sha(_PRECISION)
    audit.run(pump_dir=_PUMP_DIR, out_dir=tmp_path / "audit")
    assert _sha(_PRECISION) == before


def test_audit_does_not_modify_protected_files(tmp_path) -> None:
    before = {path: _sha(path) for path in _PROTECTED if path.exists()}
    audit.run(pump_dir=_PUMP_DIR, out_dir=tmp_path / "audit")
    after = {path: _sha(path) for path in before}
    assert after == before


def test_classifies_all_current_answerable_facts(real_audit) -> None:
    _out_dir, summary = real_audit
    pump_summary = json.loads((_PUMP_DIR / "pump_summary.json").read_text(encoding="utf-8"))
    expected_fact_count = pump_summary["pump_answerable_fact_delta_count"]
    qa_fact_count = pump_summary.get("pump_fact_qa_fact_count", expected_fact_count)

    assert expected_fact_count == qa_fact_count
    assert summary["total_answerable_facts"] == expected_fact_count
    assert summary["relations_count"] == pump_summary["pump_relation_delta_count"]
    assert summary["definitions_count"] == pump_summary["pump_definition_delta_count"]

    classified = (
        summary["promotion_candidate_count"]
        + summary["proposal_only_count"]
        + summary["needs_review_count"]
        + summary["reject_recommendation_count"]
    )
    assert classified == expected_fact_count


def test_entity_cards_are_not_promotion_candidates(tmp_path) -> None:
    pump_dir = _minimal_pump_dir(tmp_path, [
        {
            "overlay_type": "overlay_entity",
            "label": "Entity Card Only",
            "source_candidate_type": "entity_card",
        },
        {
            "overlay_type": "overlay_definition",
            "subject": "Clean Subject",
            "predicate": "is_a",
            "definition": "stable encyclopedic class",
            "source_page": "Clean Subject",
            "stability": "stable",
            "risk": "low",
        },
    ])
    summary = audit.run(pump_dir=pump_dir, out_dir=tmp_path / "audit")
    assert summary["entity_card_excluded_count"] == 1
    assert summary["total_answerable_facts"] == 1


def test_weak_context_is_not_promotion_candidate(tmp_path) -> None:
    pump_dir = _minimal_pump_dir(tmp_path, [
        {
            "overlay_type": "overlay_context_link",
            "subject": "A",
            "object": "B",
            "trust": "weak_context_only",
        },
        {
            "overlay_type": "overlay_relation",
            "subject": "SpaceX",
            "predicate": "founded_by",
            "object": "Elon Musk",
            "source_page": "SpaceX",
            "stability": "semi_stable",
            "risk": "medium",
        },
    ])
    summary = audit.run(pump_dir=pump_dir, out_dir=tmp_path / "audit")
    assert summary["weak_context_excluded_count"] == 1
    assert summary["total_answerable_facts"] == 1


def test_current_or_volatile_facts_are_not_promotion_candidates(tmp_path) -> None:
    pump_dir = _minimal_pump_dir(tmp_path, [
        {
            "overlay_type": "overlay_definition",
            "subject": "Current Price",
            "predicate": "is_a",
            "definition": "current market price",
            "source_page": "Current Price",
            "stability": "current",
            "risk": "low",
        },
    ])
    summary = audit.run(pump_dir=pump_dir, out_dir=tmp_path / "audit")
    rejects = json.loads((tmp_path / "audit" / "promotion_readiness_reject_recommendations.json").read_text())
    assert summary["promotion_candidate_count"] == 0
    assert rejects[0]["classification"] == audit.CLASS_REJECT
    assert "current_or_volatile" in rejects[0]["reasons"]


def test_qa_freshness_is_checked(real_audit) -> None:
    _out_dir, summary = real_audit
    assert summary["fact_count_matches_qa_fact_count"] is True
    assert summary["summary_says_pump_fact_qa_status_current_from_qa_artifact"] is True
    assert summary["qa_is_current"] is True


def test_qa_count_mismatch_marks_not_promotion_ready(tmp_path) -> None:
    pump_dir = _minimal_pump_dir(tmp_path, [
        {
            "overlay_type": "overlay_definition",
            "subject": "Clean Subject",
            "predicate": "is_a",
            "definition": "stable encyclopedic class",
            "source_page": "Clean Subject",
            "stability": "stable",
            "risk": "low",
        },
    ], qa_count=2)
    summary = audit.run(pump_dir=pump_dir, out_dir=tmp_path / "audit")
    assert summary["fact_count_matches_qa_fact_count"] is False
    assert summary["qa_is_current"] is False
    assert summary["promotion_ready"] is False


def test_output_artifacts_are_written(real_audit) -> None:
    out_dir, _summary = real_audit
    expected = [
        "promotion_readiness_summary.json",
        "promotion_readiness_report.json",
        "promotion_readiness_candidates.json",
        "promotion_readiness_candidates.csv",
        "promotion_readiness_needs_review.json",
        "promotion_readiness_reject_recommendations.json",
        "promotion_readiness_by_predicate.json",
        "promotion_readiness_by_source_page.json",
        "promotion_readiness_suspicious_facts.json",
    ]
    for name in expected:
        assert (out_dir / name).exists(), f"missing {name}"


def test_candidates_are_qa_covered(real_audit) -> None:
    out_dir, summary = real_audit
    candidates = json.loads((out_dir / "promotion_readiness_candidates.json").read_text())
    assert summary["all_promotion_candidates_covered_by_qa"] is True
    assert all(row["qa_covered"] for row in candidates)
