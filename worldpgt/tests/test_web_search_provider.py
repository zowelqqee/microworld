"""Tests for optional live web-search providers."""

from __future__ import annotations

from worldpgt.web_search.duckduckgo import _results_from_html, _results_from_lite_html


def test_duckduckgo_html_parser_extracts_source_results() -> None:
    html = """
    <div class="result">
      <a rel="nofollow" class="result__a" href="https://example.com/france">
        President of France
      </a>
      <a class="result__snippet">The current president of France is listed here.</a>
    </div>
    """

    results = _results_from_html(html, max_results=3)

    assert len(results) == 1
    assert results[0].title == "President of France"
    assert results[0].snippet == "The current president of France is listed here."
    assert results[0].url == "https://example.com/france"


def test_duckduckgo_lite_parser_extracts_redirect_target() -> None:
    html = """
    <tr>
      <td>
        <a rel="nofollow"
           href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FEmmanuel_Macron&amp;rut=abc"
           class='result-link'>Emmanuel Macron - Wikipedia</a>
      </td>
    </tr>
    <tr>
      <td class='result-snippet'>
        Emmanuel Macron is a French politician serving as <b>President</b> of France.
      </td>
    </tr>
    """

    results = _results_from_lite_html(html, max_results=3)

    assert len(results) == 1
    assert results[0].title == "Emmanuel Macron - Wikipedia"
    assert results[0].snippet == "Emmanuel Macron is a French politician serving as President of France."
    assert results[0].url == "https://en.wikipedia.org/wiki/Emmanuel_Macron"
