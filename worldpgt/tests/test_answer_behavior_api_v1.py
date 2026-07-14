"""API-level tests for the answer-behavior layer.

Covers the three integration guarantees:

* a failure inside the optional layer never breaks the request;
* the old QA answer is preserved verbatim when no valid plan exists;
* /ask with reasoning enabled does not 500 on the large experimental
  open-web graph, and any returned plan is fully evidence-traceable.

The large-graph fixture performs one real server startup for the whole
module (inference over the composed overlay is expensive), so these tests
are slower than the pure-layer suite in test_answer_behavior_v1.py.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from worldpgt.api import server


class _StubSemanticQuery:
    entity_a = "anything"
    entity_b = None


def test_answer_plan_builder_failure_is_isolated(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("experimental graph edge case")

    monkeypatch.setattr(server, "build_answer_plan", fail)

    assert server._build_optional_answer_plan("What is anything?", _StubSemanticQuery()) is None


def test_answer_plan_renderer_failure_is_isolated(monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("render edge case")

    monkeypatch.setattr(server, "render_answer_plan", fail)

    assert server._render_optional_answer_plan(object()) == ""


@pytest.fixture(scope="module")
def experimental_client() -> TestClient:
    server._startup("pump-dry-run", include_experimental_web_graph=True)
    assert server._experimental_web_graph["enabled"] is True
    return TestClient(server.app)


def test_ask_with_reasoning_does_not_500_on_large_experimental_graph(
    experimental_client: TestClient,
):
    for question in (
        "What does artificial intelligence enable?",
        "What is known about machine learning?",
        "Why does gravity feel heavy on Mondays?",
    ):
        response = experimental_client.post(
            "/ask", json={"question": question, "enable_reasoning": True}
        )
        assert response.status_code == 200, question
        payload = response.json()
        assert payload["decision"] in {"answer", "no", "audit", "analysis", "partial"}


def test_answer_plan_blocks_are_evidence_traceable_over_experimental_graph(
    experimental_client: TestClient,
):
    response = experimental_client.post(
        "/ask",
        json={
            "question": "What does artificial intelligence enable?",
            "enable_reasoning": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    plan = payload["answer_plan"]
    assert plan is not None
    assert len(plan["blocks"]) >= 2
    for block in plan["blocks"]:
        edge = block["step"]["edge"]
        assert edge["evidence_text"]
        assert edge["sources"]
    assert payload["support"] == "evidence_backed_answer_plan"


def test_audit_stays_audit_and_never_carries_a_plan(experimental_client: TestClient):
    response = experimental_client.post(
        "/ask",
        json={
            "question": "What is the flurbonic quantization pledge?",
            "enable_reasoning": True,
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "audit"
    assert payload["answer_plan"] is None


def test_old_qa_answer_is_preserved_when_no_plan_exists(
    experimental_client: TestClient, monkeypatch
):
    question = "What does artificial intelligence enable?"
    baseline = experimental_client.post(
        "/ask", json={"question": question, "enable_reasoning": False}
    ).json()

    monkeypatch.setattr(server, "build_answer_plan", lambda *a, **k: None)
    degraded = experimental_client.post(
        "/ask", json={"question": question, "enable_reasoning": True}
    ).json()

    assert degraded["decision"] == baseline["decision"]
    assert degraded["answer"] == baseline["answer"]
    assert degraded["support"] == baseline["support"]
    assert degraded["answer_plan"] is None


def test_accepted_memory_answers_are_never_expanded_by_the_behavior_layer(
    experimental_client: TestClient,
):
    """Hard boundary: the layer may only draw on proposal/experimental
    knowledge, never on accepted/promoted memory — even though both sit in
    the same composed in-memory overlay once the experimental graph is
    included for serving."""
    for question in ("Who founded SpaceX?", "Tell me about SpaceX"):
        baseline = experimental_client.post(
            "/ask", json={"question": question, "enable_reasoning": False}
        ).json()
        reasoned = experimental_client.post(
            "/ask", json={"question": question, "enable_reasoning": True}
        ).json()

        assert reasoned["answer"] == baseline["answer"], question
        assert reasoned["support"] == baseline["support"], question
        assert reasoned["answer_plan"] is None, question


def test_experimental_relation_items_excludes_accepted_memory_facts(
    experimental_client: TestClient,
):
    items = server._experimental_relation_items()
    assert items
    overlay_types = {item.get("overlay_type") for item in items}
    assert overlay_types == {"overlay_relation"}
    assert all(
        str(item.get("experimental_tier") or "").startswith("evidence_grounded_")
        for item in items
    )
    subjects = {str(item.get("subject") or "").casefold() for item in items}
    assert "spacex" not in subjects


def test_prepared_edges_cache_follows_startup_lifecycle(
    experimental_client: TestClient, tmp_path
):
    """Defined last in this module: it swaps the loaded overlay for a tiny
    one, so no later test may rely on the module fixture's big overlay."""
    first = server._experimental_evidence_edges()
    assert first  # non-empty over the experimental graph
    assert server._experimental_evidence_edges() is first  # cached

    tiny = tmp_path / "tiny_overlay.json"
    tiny.write_text(
        json.dumps([
            {
                "overlay_type": "overlay_relation",
                "subject": "gorven relay",
                "predicate": "steers",
                "object": "the outer lattice",
                "evidence_text": "gorven relay steers the outer lattice.",
                "source_url": "https://example.test/tiny",
                "experimental_tier": "evidence_grounded_abstract_relation_v1",
            }
        ]),
        encoding="utf-8",
    )
    server._startup("pump-dry-run", overlay_path=str(tiny))
    second = server._experimental_evidence_edges()
    assert second is not first  # reset on overlay reload
    assert [e.evidence_id for e in second] == ["edge:gorven relay|steers|the outer lattice"]
