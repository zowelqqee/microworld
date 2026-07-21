from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from worldpgt.public_demo import app as demo


@pytest.fixture(scope="module")
def engine():
    return demo._get_engine()


@pytest.fixture(scope="module")
def client(engine):
    return TestClient(demo.app)


def test_public_overlay_is_bounded_and_excludes_sensitive_or_proposal_items(engine):
    assert 1 <= len(engine.items) <= 40
    assert all(item.get("overlay_type") != "overlay_source_fact" for item in engine.items)
    assert all(item.get("overlay_type") != "overlay_context_link" for item in engine.items)
    assert all(item.get("risk") != "high" for item in engine.items)
    assert all("experimental_tier" not in item for item in engine.items)

    for item in engine.items:
        if item.get("overlay_type") == "overlay_entity":
            assert item["label"] in demo._PUBLIC_NODES
        elif item.get("overlay_type") == "overlay_definition":
            assert item["subject"] in demo._PUBLIC_NODES
        elif item.get("overlay_type") == "overlay_relation":
            assert item["subject"] in demo._PUBLIC_NODES
            assert item["object"] in demo._PUBLIC_NODES


def test_health_is_render_safe_and_does_not_expose_paths(client):
    payload = client.get("/health").json()
    assert payload == {
        "status": "ok",
        "engine_status": "ready",
        "overlay_scope": "bounded_promoted_public_subset",
        "overlay_items": len(demo._engine.items),
        "graph_edges": len(demo._engine.edges),
    }


def test_relation_answer_returns_exact_used_edges(client):
    response = client.post("/ask", json={"question": "What does SpaceX develop?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "answer"
    assert payload["support_kind"] == "semi_stable_relation"
    assert "rockets" in payload["answer"]
    assert {(edge["subject"], edge["predicate"], edge["object"]) for edge in payload["edges_used"]} == {
        ("SpaceX", "develops", "rockets"),
        ("SpaceX", "develops", "spacecraft"),
    }
    assert payload["latency_ms"] >= 0
    assert all(edge["evidence_id"].startswith("promoted:") for edge in payload["edges_used"])


def test_connection_answer_returns_the_selected_two_hop_path(client):
    response = client.post(
        "/ask", json={"question": "How is Elon Musk connected to rockets?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["support_kind"] == "explicit_connection_path"
    assert len(payload["edges_used"]) == 2
    assert payload["edges_used"][0]["subject"] == "Elon Musk"
    assert payload["edges_used"][1] == {
        "subject": "SpaceX",
        "predicate": "develops",
        "object": "rockets",
        "evidence_id": payload["edges_used"][1]["evidence_id"],
    }


def test_unsupported_current_question_is_an_honest_audit(client):
    response = client.post(
        "/ask", json={"question": "What is the current stock price of Tesla?"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["decision"] == "audit"
    assert payload["edges_used"] == []
    assert payload["support_kind"] in {"audit_blocked_context", "missing_knowledge"}


def test_cors_preflight_allows_the_local_frontend(client):
    response = client.options(
        "/ask",
        headers={
            "Origin": "http://localhost:8000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:8000"


def test_rate_limiter_returns_retry_after_without_sleeping():
    limiter = demo._SlidingWindowLimiter(limit=2, window_seconds=60)
    assert limiter.check("ip", now=100.0) is None
    assert limiter.check("ip", now=101.0) is None
    assert limiter.check("ip", now=102.0) == 59
    assert limiter.check("ip", now=161.1) is None

