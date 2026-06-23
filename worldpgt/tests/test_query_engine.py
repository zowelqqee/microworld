"""Tests for the composable query execution engine.

Covers: each primitive, plan builder for each question type, executor
(success path, empty→audit, safety→audit), and integration via the
AnswerOrchestrator for new question types.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from worldpgt.entity_qa.types import SemanticQuery
from worldpgt.knowledge.wiki_memory_overlay_provider import WikiMemoryOverlayProvider
from worldpgt.query_engine import executor as qe_executor
from worldpgt.query_engine import plan_builder as qe_plan_builder
from worldpgt.query_engine.primitives import (
    Classify,
    Compare,
    Count,
    Filter,
    Find,
    Traverse,
)
from worldpgt.query_engine.types import PlanResult, QueryPlan


# ── Test-overlay fixture helpers ───────────────────────────────────────────

def _entity(label: str, entity_type: str = "organization") -> dict:
    return {
        "overlay_type": "overlay_entity",
        "entity_id": f"test:{label}",
        "label": label,
        "aliases": [],
        "entity_type": entity_type,
    }


def _relation(
    subject: str,
    predicate: str,
    obj: str,
    *,
    stability: str = "stable",
    risk: str = "low",
    temporal_class: str = "evergreen",
) -> dict:
    return {
        "overlay_type": "overlay_relation",
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "stability": stability,
        "risk": risk,
        "temporal_class": temporal_class,
    }


def _volatile_relation(subject: str, predicate: str, obj: str) -> dict:
    return _relation(subject, predicate, obj, stability="volatile")


def _overlay(tmp_path: Path, items: list[dict]) -> WikiMemoryOverlayProvider:
    path = tmp_path / "qe_test_overlay.json"
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return WikiMemoryOverlayProvider(path)


def _semantic(
    entity_a: str | None,
    relation: str | None = None,
    entity_b: str | None = None,
    unknown: str = "object",
    query_type: str = "lookup",
    confidence: float = 0.9,
) -> SemanticQuery:
    return SemanticQuery(
        entity_a=entity_a,
        entity_b=entity_b,
        relation_intent=relation,
        unknown_position=unknown,  # type: ignore[arg-type]
        query_type=query_type,  # type: ignore[arg-type]
        confidence=confidence,
    )


# ── Standard fixture overlay used by most tests ───────────────────────────

@pytest.fixture()
def provider(tmp_path: Path) -> WikiMemoryOverlayProvider:
    return _overlay(
        tmp_path,
        [
            _entity("Elon Musk", "person"),
            _entity("Jeff Bezos", "person"),
            _entity("SpaceX"),
            _entity("Neuralink"),
            _entity("Tesla"),
            _entity("Blue Origin"),
            _entity("Amazon"),
            _relation("Elon Musk", "founded", "SpaceX"),
            _relation("Elon Musk", "founded", "Neuralink"),
            _relation("Elon Musk", "founded", "Tesla"),
            _relation("Jeff Bezos", "founded", "Amazon"),
            _relation("Jeff Bezos", "founded", "Blue Origin"),
            _relation("SpaceX", "develops", "rockets"),
            _relation("SpaceX", "is_a", "aerospace company"),
            _relation("Blue Origin", "develops", "rockets"),
            _relation("Blue Origin", "is_a", "aerospace company"),
            _relation("Tesla", "develops", "electric vehicles"),
        ],
    )


# ── Primitive: Find ────────────────────────────────────────────────────────

def test_find_object_returns_matching_relations(provider):
    step = Find(subject="Elon Musk", predicate="founded", target="object", result_key="find")
    plan = QueryPlan(steps=[step], confidence=0.9, question_type_hint="lookup")
    result = qe_executor.execute(plan, provider)
    assert result.success
    objects = [r["object"] for r in result.data]
    assert "SpaceX" in objects
    assert "Neuralink" in objects
    assert "Tesla" in objects
    assert len(objects) == 3


def test_find_subject_returns_inverse_matching(tmp_path):
    p = _overlay(
        tmp_path,
        [
            _relation("SpaceX", "founded_by", "Elon Musk"),
            _relation("Neuralink", "founded_by", "Elon Musk"),
        ],
    )
    step = Find(
        subject=None,
        predicate="founded_by",
        target="subject",
        known_value="Elon Musk",
        result_key="find",
    )
    plan = QueryPlan(steps=[step], confidence=0.9, question_type_hint="inverse_lookup")
    result = qe_executor.execute(plan, p)
    assert result.success
    subjects = [r["subject"] for r in result.data]
    assert "SpaceX" in subjects
    assert "Neuralink" in subjects


def test_find_excludes_volatile_relations(tmp_path):
    p = _overlay(
        tmp_path,
        [
            _relation("SpaceX", "founded", "stable_company"),
            _volatile_relation("SpaceX", "founded", "volatile_company"),
        ],
    )
    step = Find(subject="SpaceX", predicate="founded", target="object", result_key="find")
    plan = QueryPlan(steps=[step], confidence=0.9, question_type_hint="lookup")
    result = qe_executor.execute(plan, p)
    assert result.success
    objects = [r["object"] for r in result.data]
    assert "stable_company" in objects
    assert "volatile_company" not in objects
    assert any("safety_excluded" in t for t in result.trace)


def test_find_empty_result_causes_audit(tmp_path):
    p = _overlay(tmp_path, [])
    step = Find(subject="Elon Musk", predicate="founded", target="object", result_key="find")
    plan = QueryPlan(steps=[step], confidence=0.9, question_type_hint="lookup")
    result = qe_executor.execute(plan, p)
    assert not result.success
    assert result.decision == "audit"


# ── Primitive: Count ───────────────────────────────────────────────────────

def test_count_returns_correct_number(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find"),
            Count(source_key="find", result_key="count"),
        ],
        confidence=0.9,
        question_type_hint="count",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    assert result.data == 3


# ── Primitive: Filter ──────────────────────────────────────────────────────

def test_filter_by_secondary_predicate(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find"),
            Filter(
                source_key="find",
                condition={"predicate": "develops"},
                result_key="filter",
            ),
        ],
        confidence=0.9,
        question_type_hint="filtered_lookup",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    objects = [r["object"] for r in result.data]
    # SpaceX develops rockets; Tesla develops electric vehicles; Neuralink has no develops
    assert "SpaceX" in objects
    assert "Tesla" in objects
    assert "Neuralink" not in objects


def test_filter_by_stability_metadata(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find"),
            Filter(
                source_key="find",
                condition={"stability": "stable"},
                result_key="filter",
            ),
        ],
        confidence=0.9,
        question_type_hint="filtered_lookup",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    assert len(result.data) == 3  # all are stable


# ── Primitive: Compare – intersection ─────────────────────────────────────

def test_compare_intersection_finds_common_relations(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="SpaceX", predicate=None, target="object", result_key="find_a"),
            Find(subject="Blue Origin", predicate=None, target="object", result_key="find_b"),
            Compare(
                key_a="find_a",
                key_b="find_b",
                mode="intersection",
                label_a="SpaceX",
                label_b="Blue Origin",
                result_key="compare",
            ),
        ],
        confidence=0.9,
        question_type_hint="intersection",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    items = result.data["items"]
    predicates = [r["predicate"] for r in items]
    objects = [r["object"] for r in items]
    assert "develops" in predicates
    assert "rockets" in objects
    assert "is_a" in predicates
    assert "aerospace company" in objects


def test_compare_intersection_empty_when_no_common(tmp_path):
    p = _overlay(
        tmp_path,
        [
            _relation("Alpha", "develops", "widgets"),
            _relation("Beta", "develops", "gadgets"),
        ],
    )
    plan = QueryPlan(
        steps=[
            Find(subject="Alpha", predicate=None, target="object", result_key="find_a"),
            Find(subject="Beta", predicate=None, target="object", result_key="find_b"),
            Compare(
                key_a="find_a",
                key_b="find_b",
                mode="intersection",
                label_a="Alpha",
                label_b="Beta",
                result_key="compare",
            ),
        ],
        confidence=0.9,
        question_type_hint="intersection",
    )
    result = qe_executor.execute(plan, p)
    # Find steps succeed; Compare returns empty intersection
    assert result.success
    assert result.data["items"] == []


# ── Primitive: Compare – size_compare ─────────────────────────────────────

def test_compare_size_compare_identifies_larger(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find_a"),
            Find(subject="Jeff Bezos", predicate="founded", target="object", result_key="find_b"),
            Compare(
                key_a="find_a",
                key_b="find_b",
                mode="size_compare",
                label_a="Elon Musk",
                label_b="Jeff Bezos",
                result_key="compare",
            ),
        ],
        confidence=0.9,
        question_type_hint="size_compare",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    d = result.data
    assert d["larger"] == "a"   # Elon Musk founded 3 vs Bezos 2
    assert d["count_a"] == 3
    assert d["count_b"] == 2
    assert "Elon Musk" in result.answer_text
    assert "Jeff Bezos" in result.answer_text


# ── Primitive: Classify ────────────────────────────────────────────────────

def test_classify_returns_true_when_is_a_exists(provider):
    plan = QueryPlan(
        steps=[Classify(entity="SpaceX", target_class="aerospace company", result_key="classify")],
        confidence=0.9,
        question_type_hint="is_a",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    assert result.data["is_member"] is True
    assert "Yes" in result.answer_text


def test_classify_returns_false_when_no_is_a(provider):
    plan = QueryPlan(
        steps=[Classify(entity="Elon Musk", target_class="aerospace company", result_key="classify")],
        confidence=0.9,
        question_type_hint="is_a",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    assert result.data["is_member"] is False
    assert "not confirmed" in result.answer_text


# ── Primitive: Traverse ────────────────────────────────────────────────────

def test_traverse_finds_2hop_path(provider):
    plan = QueryPlan(
        steps=[Traverse(entity="Elon Musk", path=("founded", "develops"), result_key="traverse")],
        confidence=0.9,
        question_type_hint="traverse",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    # Elon Musk → founded → SpaceX → develops → rockets (and other combos)
    endpoints = [r["endpoint"] for r in result.data]
    assert "rockets" in endpoints
    assert "electric vehicles" in endpoints


# ── Primitive: Aggregate ───────────────────────────────────────────────────

def test_aggregate_count(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find"),
            Traverse.__class__,  # placeholder — just test aggregate
        ],
        confidence=0.9,
        question_type_hint="count",
    )
    # Test aggregate directly via explicit plan
    from worldpgt.query_engine.primitives import Aggregate
    plan2 = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find"),
            Aggregate(source_key="find", operation="count", result_key="agg"),
        ],
        confidence=0.9,
        question_type_hint="count",
    )
    result = qe_executor.execute(plan2, provider)
    assert result.success
    assert result.data == 3


def test_aggregate_list(provider):
    from worldpgt.query_engine.primitives import Aggregate
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find"),
            Aggregate(source_key="find", operation="list", result_key="agg"),
        ],
        confidence=0.9,
        question_type_hint="lookup",
    )
    result = qe_executor.execute(plan, provider)
    assert result.success
    assert isinstance(result.data, list)
    assert "SpaceX" in result.data


# ── Plan builder ───────────────────────────────────────────────────────────

def test_plan_builder_count():
    # Semantic parser maps "found/founded" → "founded_by"; _coerce_relation fixes it.
    semantic = _semantic("Elon Musk", "founded_by", unknown="object")
    plan = qe_plan_builder.build("How many companies did Elon Musk found?", semantic)
    assert plan.question_type_hint == "count"
    assert any(isinstance(s, Find) for s in plan.steps)
    assert any(isinstance(s, Count) for s in plan.steps)
    assert plan.confidence > 0.7
    # Verify normalization: Find should use "founded", not "founded_by"
    find_step = next(s for s in plan.steps if isinstance(s, Find))
    assert find_step.predicate == "founded"


def test_plan_builder_size_compare():
    semantic = _semantic("Elon Musk", "founded_by", entity_b="Jeff Bezos", unknown="object")
    plan = qe_plan_builder.build(
        "Who founded more companies — Elon Musk or Jeff Bezos?", semantic
    )
    assert plan.question_type_hint == "size_compare"
    compare_steps = [s for s in plan.steps if isinstance(s, Compare)]
    assert len(compare_steps) == 1
    assert compare_steps[0].mode == "size_compare"
    # Both Find steps should use "founded" (normalized from "founded_by")
    find_steps = [s for s in plan.steps if isinstance(s, Find)]
    assert all(s.predicate == "founded" for s in find_steps)


def test_plan_builder_filtered_lookup():
    # Parser gives rel="develops", pos="subject" — plan builder detects "that ... founded"
    # and extracts primary=founded, secondary=develops.
    semantic = _semantic("Elon Musk", "develops", unknown="subject", query_type="inverse")
    plan = qe_plan_builder.build(
        "Which companies that Elon Musk founded develop rockets?", semantic
    )
    assert plan.question_type_hint == "filtered_lookup"
    assert any(isinstance(s, Filter) for s in plan.steps)
    find_step = next(s for s in plan.steps if isinstance(s, Find))
    filter_step = next(s for s in plan.steps if isinstance(s, Filter))
    assert find_step.predicate == "founded"
    assert filter_step.condition["predicate"] == "develops"


def test_plan_builder_intersection():
    semantic = _semantic("SpaceX", None, entity_b="Blue Origin", unknown="intersection", query_type="comparative")
    plan = qe_plan_builder.build("What do SpaceX and Blue Origin have in common?", semantic)
    assert plan.question_type_hint == "intersection"
    compare_steps = [s for s in plan.steps if isinstance(s, Compare)]
    assert compare_steps[0].mode == "intersection"


def test_plan_builder_is_a():
    semantic = _semantic("SpaceX", "is_a", entity_b="aerospace company", query_type="is_a")
    plan = qe_plan_builder.build("Is SpaceX an aerospace company?", semantic)
    assert plan.question_type_hint == "is_a"
    assert any(isinstance(s, Classify) for s in plan.steps)


def test_plan_builder_standard_lookup():
    semantic = _semantic("SpaceX", "develops", unknown="object")
    plan = qe_plan_builder.build("What does SpaceX develop?", semantic)
    assert plan.question_type_hint == "lookup"
    find_steps = [s for s in plan.steps if isinstance(s, Find)]
    assert find_steps[0].subject == "SpaceX"
    assert find_steps[0].predicate == "develops"


def test_plan_builder_unrecognized_returns_zero_confidence():
    semantic = _semantic(None, None, unknown="relation", query_type="lookup", confidence=0.1)
    plan = qe_plan_builder.build("some random unrecognized text", semantic)
    assert plan.confidence == 0.0
    assert plan.question_type_hint == "unknown"


# ── Executor safety: empty Find → audit ───────────────────────────────────

def test_executor_empty_find_audits_with_reason(tmp_path):
    p = _overlay(tmp_path, [])  # empty overlay
    plan = QueryPlan(
        steps=[
            Find(subject="NoEntity", predicate="founded", target="object", result_key="find"),
            Count(source_key="find", result_key="count"),
        ],
        confidence=0.9,
        question_type_hint="count",
    )
    result = qe_executor.execute(plan, p)
    assert not result.success
    assert result.decision == "audit"
    assert result.audit_reason is not None


def test_executor_safety_excluded_all_causes_audit(tmp_path):
    p = _overlay(
        tmp_path,
        [_volatile_relation("Alpha", "founded", "Beta")],
    )
    plan = QueryPlan(
        steps=[Find(subject="Alpha", predicate="founded", target="object", result_key="find")],
        confidence=0.9,
        question_type_hint="lookup",
    )
    result = qe_executor.execute(plan, p)
    assert not result.success
    assert result.decision == "audit"


def test_executor_no_steps_audits():
    p = WikiMemoryOverlayProvider()
    plan = QueryPlan(steps=[], confidence=0.0, question_type_hint="unknown")
    result = qe_executor.execute(plan, p)
    assert not result.success
    assert result.decision == "audit"


# ── Rendering ──────────────────────────────────────────────────────────────

def test_count_render_contains_entity_and_number(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find"),
            Count(source_key="find", result_key="count"),
        ],
        confidence=0.9,
        question_type_hint="count",
    )
    result = qe_executor.execute(plan, provider)
    assert "Elon Musk" in result.answer_text
    assert "3" in result.answer_text


def test_size_compare_render_names_winner(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="Elon Musk", predicate="founded", target="object", result_key="find_a"),
            Find(subject="Jeff Bezos", predicate="founded", target="object", result_key="find_b"),
            Compare(
                key_a="find_a",
                key_b="find_b",
                mode="size_compare",
                label_a="Elon Musk",
                label_b="Jeff Bezos",
                result_key="compare",
            ),
        ],
        confidence=0.9,
        question_type_hint="size_compare",
    )
    result = qe_executor.execute(plan, provider)
    text = result.answer_text
    assert "Elon Musk" in text
    assert "more" in text.lower()


def test_intersection_render_lists_common_facts(provider):
    plan = QueryPlan(
        steps=[
            Find(subject="SpaceX", predicate=None, target="object", result_key="find_a"),
            Find(subject="Blue Origin", predicate=None, target="object", result_key="find_b"),
            Compare(
                key_a="find_a",
                key_b="find_b",
                mode="intersection",
                label_a="SpaceX",
                label_b="Blue Origin",
                result_key="compare",
            ),
        ],
        confidence=0.9,
        question_type_hint="intersection",
    )
    result = qe_executor.execute(plan, provider)
    text = result.answer_text
    assert "SpaceX" in text
    assert "Blue Origin" in text
    assert "rockets" in text


# ── Integration via AnswerOrchestrator ─────────────────────────────────────

def _orchestrator_overlay(tmp_path: Path, items: list[dict]) -> str:
    path = tmp_path / "orch_overlay.json"
    path.write_text(json.dumps(items, indent=2), encoding="utf-8")
    return str(path)


def test_orchestrator_count_question(tmp_path):
    from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
    overlay_path = _orchestrator_overlay(
        tmp_path,
        [
            _entity("Elon Musk", "person"),
            _entity("SpaceX"),
            _entity("Neuralink"),
            _entity("Tesla"),
            _relation("Elon Musk", "founded", "SpaceX"),
            _relation("Elon Musk", "founded", "Neuralink"),
            _relation("Elon Musk", "founded", "Tesla"),
        ],
    )
    orch = AnswerOrchestrator(overlay_path=overlay_path)
    answer = orch.answer("How many companies did Elon Musk found?")
    assert answer.decision == "answer"
    assert "3" in answer.answer_text
    assert answer.source_system == "query_engine"


def test_orchestrator_size_compare_question(tmp_path):
    from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
    overlay_path = _orchestrator_overlay(
        tmp_path,
        [
            _entity("Elon Musk", "person"),
            _entity("Jeff Bezos", "person"),
            _entity("SpaceX"),
            _entity("Neuralink"),
            _entity("Tesla"),
            _entity("Amazon"),
            _entity("Blue Origin"),
            _relation("Elon Musk", "founded", "SpaceX"),
            _relation("Elon Musk", "founded", "Neuralink"),
            _relation("Elon Musk", "founded", "Tesla"),
            _relation("Jeff Bezos", "founded", "Amazon"),
            _relation("Jeff Bezos", "founded", "Blue Origin"),
        ],
    )
    orch = AnswerOrchestrator(overlay_path=overlay_path)
    answer = orch.answer(
        "Who founded more companies — Elon Musk or Jeff Bezos?"
    )
    assert answer.decision == "answer"
    assert answer.source_system == "query_engine"
    text = answer.answer_text
    assert "Elon Musk" in text
    assert "more" in text.lower()


def test_orchestrator_filtered_lookup(tmp_path):
    from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
    overlay_path = _orchestrator_overlay(
        tmp_path,
        [
            _entity("Elon Musk", "person"),
            _entity("SpaceX"),
            _entity("Neuralink"),
            _entity("Tesla"),
            _relation("Elon Musk", "founded", "SpaceX"),
            _relation("Elon Musk", "founded", "Neuralink"),
            _relation("Elon Musk", "founded", "Tesla"),
            _relation("SpaceX", "develops", "rockets"),
            _relation("Tesla", "develops", "electric vehicles"),
        ],
    )
    orch = AnswerOrchestrator(overlay_path=overlay_path)
    answer = orch.answer(
        "Which companies that Elon Musk founded develop rockets?"
    )
    assert answer.decision == "answer"
    assert answer.source_system == "query_engine"
    assert "SpaceX" in answer.answer_text


def test_orchestrator_existing_lookup_unchanged(tmp_path):
    """Standard lookups must still work through the existing planner."""
    from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
    overlay_path = _orchestrator_overlay(
        tmp_path,
        [
            _entity("SpaceX"),
            {"overlay_type": "overlay_definition", "subject": "SpaceX",
             "definition": "aerospace manufacturer", "stability": "stable",
             "trust": "overlay_candidate"},
            _relation("SpaceX", "develops", "Falcon 9"),
        ],
    )
    orch = AnswerOrchestrator(overlay_path=overlay_path)
    answer = orch.answer("What does SpaceX develop?")
    assert answer.decision == "answer"
    # Should be answered by entity_qa, not query_engine
    assert answer.source_system != "query_engine"
