"""Tests for the low-trust Reddit/community-context engine."""

from __future__ import annotations

import json

from worldpgt.community_context.reddit_engine import (
    build_speaking_profile,
    build_reddit_community_context,
    classify_reddit_records,
    load_reddit_records,
    query_community_context,
    render_community_context,
)
from worldpgt.community_context.cognitive_pattern_pump import (
    build_cognitive_pattern_graph,
    extract_cognitive_pattern_events,
    plan_answer_with_cognitive_patterns,
    query_cognitive_patterns,
)
from worldpgt.community_context.types import RedditRecord
from worldpgt.experiments import run_reddit_community_pump_v1 as pump


def test_load_reddit_listing_and_jsonl_records(tmp_path) -> None:
    listing = tmp_path / "listing.json"
    listing.write_text(
        json.dumps(
            {
                "data": {
                    "children": [
                        {
                            "kind": "t3",
                            "data": {
                                "id": "abc",
                                "subreddit": "learnprogramming",
                                "title": "How do people learn Python?",
                                "selftext": "I see beginners ask about projects, loops, and debugging workflows.",
                                "score": 12,
                                "permalink": "/r/learnprogramming/comments/abc/how/",
                            },
                        }
                    ]
                }
            }
        ),
        encoding="utf-8",
    )
    jsonl = tmp_path / "comments.jsonl"
    jsonl.write_text(
        json.dumps(
            {
                "kind": "t1",
                "data": {
                    "id": "c1",
                    "subreddit": "AskReddit",
                    "body": "People usually frame this as a tradeoff between speed, comfort, and reliability.",
                    "score": 5,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = load_reddit_records([listing, jsonl])

    assert [record.source_kind for record in records] == ["post", "comment"]
    assert records[0].subreddit == "learnprogramming"
    assert records[1].body.startswith("People usually frame")


def test_classify_reddit_records_keeps_context_out_of_overlay_format() -> None:
    records = [
        RedditRecord(
            source_id="good",
            source_kind="post",
            subreddit="python",
            title="What helped you understand decorators?",
            body="Several commenters explain decorators by comparing them to wrapping a function with extra behavior.",
            author="human",
            score=9,
        ),
        RedditRecord(
            source_id="pii",
            source_kind="comment",
            subreddit="advice",
            title="",
            body="Email me at person@example.com and I will share the private details.",
            author="human",
            score=3,
        ),
        RedditRecord(
            source_id="nsfw",
            source_kind="post",
            subreddit="x",
            title="Hidden",
            body="This should not be indexed into context.",
            author="human",
            score=10,
            over_18=True,
        ),
    ]

    accepted, quarantine = classify_reddit_records(records)

    assert len(accepted) == 1
    assert accepted[0].trust == "community_context_only"
    assert accepted[0].source_system == "reddit"
    assert "overlay_type" not in accepted[0].to_dict()
    assert {item.reason for item in quarantine} == {"private_or_sensitive_data", "nsfw"}


def test_query_and_render_community_context_disclaims_fact_support() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="learning-python",
                source_kind="post",
                subreddit="learnpython",
                title="Learning Python without getting stuck",
                body="People often recommend small projects, reading errors carefully, and asking focused debugging questions.",
                author="human",
                score=14,
                permalink="/r/learnpython/comments/learning/",
            )
        ]
    )

    results = query_community_context(accepted, "how to learn python debugging")
    rendered = render_community_context("how to learn python debugging", results)

    assert results
    assert results[0].item.subreddit == "learnpython"
    assert "A practical debugging rhythm" in rendered
    assert "not factual support" in rendered
    assert "Do not promote it to stable facts" in rendered


def test_programming_beginner_mistakes_render_is_not_raw_irrelevant_dump() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="good-programming",
                source_kind="comment",
                subreddit="HackerNews",
                title="Ask HN: High school student - is learning programming still worthwhile?",
                body=(
                    "Short answer: yes. Beginners benefit from building small programs, "
                    "reading errors carefully, and learning how to debug instead of only "
                    "watching tutorials."
                ),
                author="human",
                score=5,
            ),
            RedditRecord(
                source_id="bad-mri",
                source_kind="comment",
                subreddit="HackerNews",
                title="I used Claude Code to get a second opinion on my MRI",
                body="This draws too strong a line between matrix math and a harness.",
                author="human",
                score=5,
            ),
            RedditRecord(
                source_id="bad-guns",
                source_kind="comment",
                subreddit="HackerNews",
                title="Gun Mistakes in Fiction Writing",
                body="Buckshot spread is another common mistake in fiction.",
                author="human",
                score=5,
            ),
        ]
    )

    results = query_community_context(
        accepted,
        "What are common mistakes beginners make when learning programming?",
    )
    rendered = render_community_context(
        "What are common mistakes beginners make when learning programming?",
        results,
    )

    assert results
    assert "Common beginner programming mistakes" in rendered
    assert "Copying code without stopping to explain" in rendered
    assert "MRI" not in rendered
    assert "Gun Mistakes" not in rendered
    assert "Signals I used" not in rendered


def test_community_style_answers_programming_question_shape() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="question-style",
                source_kind="comment",
                subreddit="HackerNews",
                title="How to ask useful questions",
                body="People can help faster when the question includes the exact error, a small example, and what was already tried.",
                author="human",
                score=4,
            )
        ]
    )

    results = query_community_context(accepted, "How should I ask a good programming question?")
    rendered = render_community_context("How should I ask a good programming question?", results)

    assert "small, clean bug report" in rendered
    assert "What I expected to happen" in rendered
    assert "Signals I used" not in rendered


def test_community_style_answers_simple_recursion_explanation() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="simple-explain",
                source_kind="comment",
                subreddit="HackerNews",
                title="Simple explanations",
                body="Good explanations use plain words, analogies, and one idea at a time.",
                author="human",
                score=4,
            )
        ]
    )

    results = query_community_context(accepted, "Explain recursion in simple terms like people on forums do.")
    rendered = render_community_context("Explain recursion in simple terms like people on forums do.", results)

    assert "calling itself on a smaller version" in rendered
    assert "nested boxes" in rendered
    assert "Signals I used" not in rendered


def test_build_reddit_community_context_writes_isolated_artifacts(tmp_path) -> None:
    source = tmp_path / "reddit.json"
    source.write_text(
        json.dumps(
            [
                {
                    "id": "one",
                    "subreddit": "travel",
                    "title": "What do people worry about before moving countries?",
                    "selftext": "Common concerns include paperwork, housing, language, healthcare, and finding a routine.",
                    "score": 20,
                },
                {
                    "id": "deleted",
                    "subreddit": "travel",
                    "title": "Gone",
                    "selftext": "[deleted]",
                    "score": 4,
                },
            ]
        ),
        encoding="utf-8",
    )

    summary = build_reddit_community_context([source], tmp_path / "out")

    assert summary["accepted_context_items_count"] == 1
    assert summary["cognitive_pattern_events_count"] >= 1
    assert summary["quarantine_count"] == 1
    assert summary["factual_support_allowed"] is False
    assert summary["community_patterns_factual_support_allowed"] is False
    assert summary["accepted_overlay_modified"] is False
    context = json.loads((tmp_path / "out" / "reddit_community_context.json").read_text(encoding="utf-8"))
    assert context[0]["trust"] == "community_context_only"
    assert "overlay_type" not in context[0]
    profile = json.loads((tmp_path / "out" / "reddit_speaking_profile.json").read_text(encoding="utf-8"))
    assert profile["factual_support_allowed"] is False
    assert profile["item_count"] == 1
    patterns = json.loads((tmp_path / "out" / "cognitive_pattern_events.json").read_text(encoding="utf-8"))
    graphs = json.loads((tmp_path / "out" / "cognitive_pattern_graphs.json").read_text(encoding="utf-8"))
    assert patterns
    assert all(event["factual_support_allowed"] is False for event in patterns)
    assert all("overlay_type" not in event for event in patterns)
    assert "style_tone_graph" in graphs
    assert "evidence_graph" in graphs


def test_speaking_profile_extracts_phrasing_and_common_terms() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="phrasing",
                source_kind="post",
                subreddit="learnpython",
                title="Why do beginners get stuck debugging Python?",
                body="People struggle when error messages feel confusing and the project is too large.",
                author="human",
                score=11,
            )
        ]
    )

    profile = build_speaking_profile(accepted)

    assert profile["trust"] == "community_context_only"
    assert profile["factual_support_allowed"] is False
    assert profile["sample_question_phrasings"][0]["text"].startswith("Why do beginners")
    assert any(item["term"] == "python" for item in profile["top_topic_terms"])


def test_cognitive_pattern_pump_extracts_reusable_events_not_facts() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="debug-one",
                source_kind="comment",
                subreddit="learnprogramming",
                title="How to ask a better debugging question",
                body=(
                    "People help faster when you include expected behavior, actual behavior, "
                    "the exact error, and the smallest reproducible example."
                ),
                author="human",
                score=18,
            ),
            RedditRecord(
                source_id="debug-two",
                source_kind="comment",
                subreddit="learnpython",
                title="Stuck debugging Python",
                body=(
                    "Reduce the bug to a small repro, read the traceback, and say what you "
                    "already tried before asking for help."
                ),
                author="human",
                score=9,
            ),
        ]
    )

    events = extract_cognitive_pattern_events(accepted)
    debugging = [event for event in events if event.kind == "debugging_pattern"]

    assert debugging
    assert debugging[0].pattern == "reduce the problem to a minimal reproducible example"
    assert debugging[0].trust == "behavioral_pattern"
    assert debugging[0].factual_support_allowed is False
    assert debugging[0].signal_count == 2
    assert "include the exact error message" in debugging[0].steps
    assert "overlay_type" not in debugging[0].to_dict()


def test_cognitive_pattern_graph_keeps_evidence_separate() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="explain-one",
                source_kind="comment",
                subreddit="explainlikeimfive",
                title="Why plain explanations work",
                body="Good explanations use simple words, concrete examples, analogies, and one idea at a time.",
                author="human",
                score=20,
                permalink="/r/explainlikeimfive/comments/explain-one/plain/",
            )
        ]
    )
    events = extract_cognitive_pattern_events(accepted)

    graphs = build_cognitive_pattern_graph(events)

    assert graphs["explanation_graph"]["nodes"]
    assert graphs["evidence_graph"]["nodes"]
    assert any(edge["relation"] == "suggests_pattern" for edge in graphs["evidence_graph"]["edges"])
    assert not any(
        node.get("type") == "fact"
        for graph in graphs.values()
        for node in graph["nodes"]
    )


def test_cognitive_pattern_retrieval_and_answer_plan_are_source_separated() -> None:
    accepted, _quarantine = classify_reddit_records(
        [
            RedditRecord(
                source_id="recursion",
                source_kind="comment",
                subreddit="learnprogramming",
                title="Explaining recursion to beginners",
                body="Use a simple analogy, then explain the base case and the smaller subproblem.",
                author="human",
                score=12,
            ),
            RedditRecord(
                source_id="tone",
                source_kind="comment",
                subreddit="HackerNews",
                title="Explaining technical ideas",
                body="People prefer plain wording, concrete examples, and less formal language when stuck.",
                author="human",
                score=8,
            ),
        ]
    )
    events = extract_cognitive_pattern_events(accepted)

    results = query_cognitive_patterns(events, "Explain recursion simply to a beginner")
    plan = plan_answer_with_cognitive_patterns(
        "Explain recursion simply to a beginner",
        events,
    )

    assert results
    assert results[0].event.source == "community_context"
    assert plan["factual_support_allowed_from_patterns"] is False
    assert plan["known_facts_source"] == "factual_memory_or_live_search_required"
    assert any(
        event["kind"] == "explanation_pattern"
        for event in plan["cognitive_patterns"]
    )
    assert "Do not treat community pattern text as factual support." in plan["checks_before_answering"]


def test_reddit_community_pump_once_creates_status_and_outputs(tmp_path) -> None:
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "batch.jsonl").write_text(
        json.dumps(
            {
                "id": "pump-one",
                "subreddit": "AskReddit",
                "title": "What makes explanations easy to understand?",
                "selftext": "People usually prefer concrete examples, short steps, and less formal wording.",
                "score": 18,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    status = pump.run_once(input_dir=inbox, out_dir=tmp_path / "out")

    assert status["input_files_count"] == 1
    assert status["accepted_context_items_count"] == 1
    assert status["cognitive_pattern_events_count"] >= 1
    assert status["factual_support_allowed"] is False
    assert status["community_patterns_factual_support_allowed"] is False
    assert (tmp_path / "out" / "reddit_community_context.json").is_file()
    assert (tmp_path / "out" / "reddit_speaking_profile.json").is_file()
    assert (tmp_path / "out" / "cognitive_pattern_events.json").is_file()
    assert (tmp_path / "out" / "cognitive_pattern_graphs.json").is_file()
    assert (tmp_path / "out" / "reddit_community_pump_status.json").is_file()


def test_reddit_community_pump_fetches_public_subreddits_to_inbox(tmp_path, monkeypatch) -> None:
    def fake_fetch(subreddit, **_kwargs):
        return {
            "data": {
                "children": [
                    {
                        "kind": "t3",
                        "data": {
                            "id": f"{subreddit}-one",
                            "subreddit": subreddit,
                            "title": "What makes explanations easier to understand?",
                            "selftext": (
                                f"People in {subreddit} prefer examples, plain wording, "
                                "and clear step by step answers."
                            ),
                            "score": 22,
                            "permalink": f"/r/{subreddit}/comments/one/example/",
                        },
                    }
                ]
            }
        }

    monkeypatch.setattr(pump, "fetch_subreddit_listing", fake_fetch)

    status = pump.run_once(
        input_dir=tmp_path / "inbox",
        out_dir=tmp_path / "out",
        subreddits=["AskReddit", "NoStupidQuestions"],
    )

    assert status["fetch_enabled"] is True
    assert [row["status"] for row in status["fetch_report"]] == ["ok", "ok"]
    assert status["input_files_count"] == 2
    assert status["accepted_context_items_count"] == 2
    assert len(list((tmp_path / "inbox" / "fetched").glob("*.json"))) == 2


def test_reddit_community_pump_rejects_invalid_subreddit_name(tmp_path) -> None:
    report = pump.fetch_subreddits_to_inbox(
        ["../../bad"],
        input_dir=tmp_path / "inbox",
    )

    assert report == [{"subreddit": "../../bad", "status": "skipped", "reason": "invalid_name"}]


def test_reddit_fetch_candidates_include_old_reddit_fallback() -> None:
    urls = pump.candidate_listing_urls("AskReddit", listing="top", time_filter="week", limit=150)

    assert urls[0].startswith("https://www.reddit.com/r/AskReddit/top.json?")
    assert urls[1].startswith("https://old.reddit.com/r/AskReddit/top.json?")
    assert "limit=100" in urls[0]
    assert "raw_json=1" in urls[0]


def test_hn_hits_convert_to_reddit_like_records() -> None:
    records = pump.hn_hits_to_reddit_like_records(
        {
            "hits": [
                {
                    "objectID": "123",
                    "story_title": "Learning to explain things",
                    "comment_text": "People like <b>small examples</b> and concrete language.",
                    "author": "ada",
                    "created_at_i": 123456,
                }
            ]
        },
        query="explain simply",
    )

    assert records == [
        {
            "id": "hn-123",
            "subreddit": "HackerNews",
            "title": "Learning to explain things",
            "body": "People like small examples and concrete language.",
            "author": "ada",
            "score": 3,
            "permalink": "https://news.ycombinator.com/item?id=123",
            "created_utc": "123456",
        }
    ]


def test_reddit_pump_uses_hn_fallback_when_reddit_blocked(tmp_path, monkeypatch) -> None:
    def fake_reddit(_subreddit, **_kwargs):
        raise RuntimeError("blocked")

    def fake_hn(query, **_kwargs):
        return {
            "hits": [
                {
                    "objectID": f"{query}-1",
                    "story_title": f"{query} discussion",
                    "comment_text": f"People discussing {query} prefer examples and plain wording.",
                    "author": "user",
                }
            ]
        }

    monkeypatch.setattr(pump, "fetch_subreddit_listing", fake_reddit)
    monkeypatch.setattr(pump, "fetch_hn_query", fake_hn)

    status = pump.run_once(
        input_dir=tmp_path / "inbox",
        out_dir=tmp_path / "out",
        subreddits=["AskReddit"],
        hn_queries=["explain simply"],
    )

    assert status["fetch_report"][0]["status"] == "error"
    assert status["hn_fetch_report"][0]["status"] == "ok"
    assert status["accepted_context_items_count"] == 1
    assert len(list((tmp_path / "inbox" / "fetched_hn").glob("*.jsonl"))) == 1


def test_reddit_pump_default_fetch_snapshot_does_not_accumulate_files(tmp_path, monkeypatch) -> None:
    def fake_hn(query, **_kwargs):
        return {
            "hits": [
                {
                    "objectID": f"{query}-1",
                    "story_title": f"{query} discussion",
                    "comment_text": f"People discussing {query} prefer examples and plain wording.",
                    "author": "user",
                }
            ]
        }

    monkeypatch.setattr(pump, "fetch_hn_query", fake_hn)
    inbox = tmp_path / "inbox"
    out = tmp_path / "out"

    first = pump.run_once(
        input_dir=inbox,
        out_dir=out,
        subreddits=[],
        hn_queries=["explain simply", "common mistakes"],
    )
    second = pump.run_once(
        input_dir=inbox,
        out_dir=out,
        subreddits=[],
        hn_queries=["explain simply", "common mistakes"],
    )

    assert first["input_files_count"] == 2
    assert second["input_files_count"] == 2
    assert len(list((inbox / "fetched_hn").glob("*.jsonl"))) == 2
    assert second["accepted_context_items_count"] == first["accepted_context_items_count"]
