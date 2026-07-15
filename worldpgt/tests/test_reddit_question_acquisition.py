import pytest

from worldpgt.knowledge_pump.reddit_question_acquisition import (
    official_reddit_fetcher_from_env,
    question_candidates_from_listing,
    run_live_question_acquisition,
)


def _listing(title: str, post_id: str = "abc") -> dict:
    return {"data": {"children": [{"data": {"id": post_id, "title": title, "permalink": "/r/artificial/comments/abc/test/", "score": 4}}]}}


def test_live_reddit_acquisition_collects_question_candidates_only(tmp_path):
    clock_values = iter((0.0,) * 8 + (1.0,) * 8)
    summary = run_live_question_acquisition(
        tmp_path, duration_seconds=1, subreddits=("artificial",), poll_seconds=0,
        fetch=lambda _subreddit: _listing("What does AI make possible?"),
        clock=lambda: next(clock_values), sleep=lambda _seconds: None,
    )
    assert summary["proposal_only"] is True
    assert summary["accepted_memory_modified"] is False
    assert summary["runtime_graph_modified"] is False
    assert summary["factual_support_allowed"] is False
    assert summary["candidate_count"] == 1
    assert (tmp_path / "reddit_ai_question_candidates.jsonl").exists()


def test_reddit_question_extraction_drops_non_questions_and_preserves_review_boundary():
    payload = {"data": {"children": [
        {"data": {"id": "q", "title": "How does AI help with research?", "permalink": "/r/artificial/comments/q/", "score": 2}},
        {"data": {"id": "s", "title": "New paper release", "permalink": "/r/artificial/comments/s/", "score": 2}},
    ]}}
    candidates = question_candidates_from_listing(payload, "artificial")
    assert [item["candidate_id"] for item in candidates] == ["reddit:q"]
    assert candidates[0]["review_status"] == "unreviewed"
    assert candidates[0]["factual_support_allowed"] is False


def test_official_reddit_fetcher_requires_explicit_oauth_credentials(monkeypatch):
    monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
    monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
    with pytest.raises(RuntimeError, match="REDDIT_CLIENT_ID"):
        official_reddit_fetcher_from_env()
