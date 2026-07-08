"""Tests for the Claude-backed web search provider (DuckDuckGo replacement).

No live Anthropic credentials exist in this environment, so these tests
cover (1) graceful no-network degradation when no credentials are
configured, and (2) response-parsing behavior against a fake client -- the
same style used elsewhere in this codebase for un-testable-live providers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from worldpgt.web_search.claude_search import ClaudeWebSearchProvider


def test_search_returns_empty_without_making_a_network_call_when_no_credentials(monkeypatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", "/nonexistent/path/for/test")

    calls = []

    class _ExplodingClient:
        def __init__(self, *a, **k):
            calls.append("constructed")
            raise AssertionError("should never construct a client without credentials")

    import worldpgt.web_search.claude_search as claude_search

    class _FakeAnthropicModule:
        Anthropic = _ExplodingClient

    monkeypatch.setattr(claude_search, "anthropic", _FakeAnthropicModule())

    provider = ClaudeWebSearchProvider()
    results = provider.search("who invented the lightning rod?")

    assert results == []
    assert calls == []


def test_search_returns_empty_for_blank_query() -> None:
    provider = ClaudeWebSearchProvider(client=object())  # any non-None sentinel
    assert provider.search("   ") == []


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeSearchResultItem:
    url: str
    title: str


@dataclass
class _FakeSearchToolResultBlock:
    content: list
    type: str = "web_search_tool_result"


@dataclass
class _FakeResponse:
    content: list = field(default_factory=list)


class _FakeMessages:
    def __init__(self, response) -> None:
        self._response = response
        self.last_kwargs: dict = {}

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


class _FakeClient:
    def __init__(self, response) -> None:
        self.messages = _FakeMessages(response)


def test_search_wraps_claude_answer_text_and_first_citation_as_a_result() -> None:
    response = _FakeResponse(content=[
        _FakeSearchToolResultBlock(content=[
            _FakeSearchResultItem(url="https://en.wikipedia.org/wiki/Benjamin_Franklin", title="Benjamin Franklin"),
        ]),
        _FakeTextBlock(text="Benjamin Franklin invented the lightning rod and bifocals."),
    ])
    client = _FakeClient(response)

    provider = ClaudeWebSearchProvider(client=client)
    results = provider.search("what else did ben franklin invent?")

    assert len(results) == 1
    assert results[0].snippet == "Benjamin Franklin invented the lightning rod and bifocals."
    assert results[0].title == "Benjamin Franklin"
    assert results[0].url == "https://en.wikipedia.org/wiki/Benjamin_Franklin"
    # web_search tool must actually be declared on the request
    assert client.messages.last_kwargs["tools"][0]["type"] == "web_search_20260209"


def test_search_returns_empty_when_claude_produces_no_text() -> None:
    response = _FakeResponse(content=[
        _FakeSearchToolResultBlock(content=[]),
    ])
    client = _FakeClient(response)

    provider = ClaudeWebSearchProvider(client=client)
    results = provider.search("an unanswerable question")

    assert results == []


def test_search_returns_empty_when_the_api_call_raises() -> None:
    class _BoomMessages:
        def create(self, **kwargs):
            raise RuntimeError("network is down")

    class _BoomClient:
        messages = _BoomMessages()

    provider = ClaudeWebSearchProvider(client=_BoomClient())
    results = provider.search("does this survive an exception?")

    assert results == []
