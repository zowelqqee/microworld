"""Tests for pump ``is_a`` promotion into a separate QA overlay."""

from __future__ import annotations

import json
from pathlib import Path

from worldpgt.experiments import promote_pump_is_a_edges_v1 as promoter


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


def _minimal_pump_dir(tmp_path: Path, facts: list[dict]) -> Path:
    pump_dir = tmp_path / "pump"
    answerable = [
        fact for fact in facts
        if fact.get("overlay_type") in {"overlay_relation", "overlay_definition"}
    ]
    _write_json(pump_dir / "pump_precision_answerable_delta.json", facts)
    _write_json(
        pump_dir / "pump_fact_qa_v1" / "pump_fact_qa_outputs.json",
        [
            _qa_output(
                str(fact.get("subject", "")),
                str(fact.get("predicate", "is_a")),
                str(fact.get("object", fact.get("definition", ""))),
            )
            for fact in answerable
        ],
    )
    _write_json(
        pump_dir / "pump_fact_qa_v1" / "pump_fact_qa_summary.json",
        {
            "pump_fact_qa_fact_count": len(answerable),
            "pump_fact_qa_relation_fact_count": sum(
                1 for fact in answerable if fact.get("overlay_type") == "overlay_relation"
            ),
            "pump_fact_qa_definition_fact_count": sum(
                1 for fact in answerable if fact.get("overlay_type") == "overlay_definition"
            ),
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


def test_promotes_only_readiness_accepted_is_a_items(tmp_path) -> None:
    base = tmp_path / "base_overlay.json"
    _write_json(base, [])
    pump_dir = _minimal_pump_dir(tmp_path, [
        {
            "overlay_type": "overlay_definition",
            "subject": "Clean Subject",
            "predicate": "is_a",
            "definition": "stable encyclopedic class",
            "source_page": "Clean Subject",
            "evidence_text": "Clean Subject is a stable encyclopedic class.",
            "stability": "stable",
            "risk": "low",
        },
        {
            "overlay_type": "overlay_relation",
            "subject": "Clean Subject",
            "predicate": "founded_by",
            "object": "Ada Lovelace",
            "source_page": "Clean Subject",
            "evidence_text": "Clean Subject was founded by Ada Lovelace.",
            "stability": "semi_stable",
            "risk": "medium",
        },
    ])

    report = promoter.run(
        pump_dir=pump_dir,
        out_dir=tmp_path / "promotion",
        base_overlay_path=base,
        experiments_dir=tmp_path / "experiments",
        root=tmp_path / "root",
    )

    promoted = json.loads(Path(report["promoted_overlay_path"]).read_text(encoding="utf-8"))
    assert report["readiness_is_a_candidate_count"] == 1
    assert report["validation_accepted_count"] == 1
    assert promoted == [
        {
            "overlay_type": "overlay_definition",
            "subject": "Clean Subject",
            "predicate": "is_a",
            "definition": "stable encyclopedic class",
            "source_page": "Clean Subject",
            "evidence_text": "Clean Subject is a stable encyclopedic class.",
            "stability": "stable",
            "risk": "low",
        }
    ]


def test_current_sensitive_is_a_is_not_promoted(tmp_path) -> None:
    base = tmp_path / "base_overlay.json"
    _write_json(base, [])
    pump_dir = _minimal_pump_dir(tmp_path, [
        {
            "overlay_type": "overlay_definition",
            "subject": "Current Widget",
            "predicate": "is_a",
            "definition": "current market product",
            "source_page": "Current Widget",
            "evidence_text": "Current Widget is a current market product.",
            "stability": "current",
            "risk": "low",
        },
    ])

    report = promoter.run(
        pump_dir=pump_dir,
        out_dir=tmp_path / "promotion",
        base_overlay_path=base,
        experiments_dir=tmp_path / "experiments",
        root=tmp_path / "root",
    )

    promoted = json.loads(Path(report["promoted_overlay_path"]).read_text(encoding="utf-8"))
    assert report["readiness_is_a_candidate_count"] == 0
    assert report["validation_accepted_count"] == 0
    assert promoted == []


def test_promotion_runner_does_not_consume_ontology_layer_artifact(tmp_path) -> None:
    base = tmp_path / "base_overlay.json"
    out_dir = tmp_path / "promotion"
    _write_json(base, [])
    _write_json(
        out_dir / "wikidata_p279_ontology_layer.json",
        [
            {
                "overlay_type": "overlay_relation",
                "subject": "businessman",
                "predicate": "is_a",
                "object": "worker",
                "source_page": "Wikidata",
                "evidence_text": "Wikidata P279 subclass of: businessman -> worker",
                "stability": "stable",
                "risk": "low",
            }
        ],
    )
    pump_dir = _minimal_pump_dir(tmp_path, [])

    report = promoter.run(
        pump_dir=pump_dir,
        out_dir=out_dir,
        base_overlay_path=base,
        experiments_dir=tmp_path / "experiments",
        root=tmp_path / "root",
    )

    promoted = json.loads(Path(report["promoted_overlay_path"]).read_text(encoding="utf-8"))
    assert report["validation_accepted_count"] == 0
    assert promoted == []
