"""DuckDuckGo Instant Answer provider for optional live QA fallback."""

from __future__ import annotations

import json
import re
from html import unescape
from urllib.parse import parse_qs, quote_plus, unquote, urlparse
from urllib.request import Request, urlopen

from worldpgt.assistant_surface.web_search import (
    WebSearchResult,
    _results_from_instant_answer,
)


class DuckDuckGoInstantAnswerProvider:
    """Small stdlib-only DuckDuckGo Instant Answer client."""

    def __init__(self, *, timeout_sec: float = 5.0, user_agent: str | None = None) -> None:
        self.timeout_sec = timeout_sec
        self.user_agent = user_agent or "mini-worldgrad-web-search/1.0"

    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        q = quote_plus((query or "").strip())
        if not q:
            return []
        url = f"https://api.duckduckgo.com/?q={q}&format=json&no_html=1&skip_disambig=1"
        req = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(req, timeout=self.timeout_sec) as response:
            data = json.loads(response.read().decode("utf-8"))
        results = _results_from_instant_answer(data, max_results=max_results)
        if results:
            return results

        html_url = f"https://lite.duckduckgo.com/lite/?q={q}"
        html_req = Request(html_url, headers={"User-Agent": self.user_agent})
        with urlopen(html_req, timeout=self.timeout_sec) as response:
            html = response.read().decode("utf-8", errors="replace")
        results = _results_from_lite_html(html, max_results=max_results)
        if results:
            return results

        html_url = f"https://html.duckduckgo.com/html/?q={q}"
        html_req = Request(html_url, headers={"User-Agent": self.user_agent})
        with urlopen(html_req, timeout=self.timeout_sec) as response:
            html = response.read().decode("utf-8", errors="replace")
        return _results_from_html(html, max_results=max_results)


_RESULT_RE = re.compile(
    r'<a[^>]+class="result__a"[^>]+href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>'
    r".*?"
    r'<a[^>]+class="result__snippet"[^>]*>(?P<snippet>.*?)</a>',
    re.IGNORECASE | re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")
_LITE_RESULT_RE = re.compile(
    r"<a[^>]+href=\"(?P<url>[^\"]+)\"[^>]+class='result-link'[^>]*>(?P<title>.*?)</a>"
    r".*?"
    r"<td[^>]+class='result-snippet'[^>]*>(?P<snippet>.*?)</td>",
    re.IGNORECASE | re.DOTALL,
)


def _strip_html(text: str) -> str:
    return re.sub(r"\s+", " ", unescape(_TAG_RE.sub(" ", text or ""))).strip()


def _results_from_html(html: str, *, max_results: int) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    for match in _RESULT_RE.finditer(html or ""):
        title = _strip_html(match.group("title"))
        snippet = _strip_html(match.group("snippet"))
        url = unescape(match.group("url")).strip()
        if not title or not snippet or not url:
            continue
        results.append(WebSearchResult(title=title, snippet=snippet, url=url))
        if len(results) >= max_results:
            break
    return results


def _normalize_duckduckgo_url(url: str) -> str:
    url = unescape(url or "").strip()
    if url.startswith("//"):
        url = "https:" + url
    parsed = urlparse(url)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path.startswith("/l/"):
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return unquote(target)
    return url


def _results_from_lite_html(html: str, *, max_results: int) -> list[WebSearchResult]:
    results: list[WebSearchResult] = []
    for match in _LITE_RESULT_RE.finditer(html or ""):
        title = _strip_html(match.group("title"))
        snippet = _strip_html(match.group("snippet"))
        url = _normalize_duckduckgo_url(match.group("url"))
        if not title or not snippet or not url:
            continue
        results.append(WebSearchResult(title=title, snippet=snippet, url=url))
        if len(results) >= max_results:
            break
    return results
