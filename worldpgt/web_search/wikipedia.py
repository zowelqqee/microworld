"""Wikipedia article provider for the live web-search fallback.

Unlike DuckDuckGo's tiny Instant-Answer abstract (usually just a page's intro
line), this fetches the full plain-text article extract in a single combined
search+extract call. The larger text lets the answer-extraction layer find the
specific relational fact a question asks (spouse, cause of death, school,
discovery, ...), which the DDG abstract almost never contains.

Resolution is resilient: it searches on the full natural-language question,
then falls back to a bare-entity reformulation of the question if that finds
no page. Stdlib-only; shares :class:`ResilientHttpClient` for UA rotation,
pacing, and retry/backoff.
"""

from __future__ import annotations

import json
import re
import time
from urllib.parse import quote, quote_plus
from urllib.request import urlopen

from worldpgt.assistant_surface.web_search import WebSearchResult
from worldpgt.web_search.http import ResilientHttpClient

_API = "https://en.wikipedia.org/w/api.php"
# Cap the extract we keep: enough to cover the intro + early sections where
# key relational facts live, without holding whole articles in memory/cache.
_MAX_EXTRACT_CHARS = 8000

# Leading interrogative scaffolding stripped to derive a bare-entity query.
_WH_PREFIX_RE = re.compile(
    r"^\s*(?:who|what|when|where|which|whom|whose|why|how|name|is|are|was|were|"
    r"the|a|an)\b[\s,]*",
    re.IGNORECASE,
)
# "what X did ENTITY verb ..." — the entity follows a did/does/do pivot.
_DID_PIVOT_RE = re.compile(r"\b(?:did|does|do)\b\s+", re.IGNORECASE)
# Trailing relation clauses ("... married to", "... born", "... go to school").
_RELATION_TAIL_RE = re.compile(
    r"\b(?:married\s+to|married|marry|born|die[d]?|death|killed|from|located|"
    r"go\s+to|went\s+to|attend(?:ed)?|study|studied|educated|graduate[d]?|"
    r"invent(?:ed)?|discover(?:ed)?|found(?:ed)?|create[d]?|"
    r"play\s+for|play(?:s|ed)?|sign(?:ed)?\s+with|"
    r"known\s+for|famous\s+for|write|wrote|written|represent[s]?|"
    r"do|does|did|go|went)\b.*$",
    re.IGNORECASE,
)
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for",
    "with", "from", "is", "are", "was", "were", "did", "does", "do",
})


def bare_entity_query(question: str) -> str:
    """Best-effort bare entity name from a natural-language question.

    "who did michael j fox marry?" -> "michael j fox". Deterministic and
    lossy; Wikipedia search is fuzzy, so this only needs to be close.
    """
    q = (question or "").strip().rstrip("?.!").strip().lower()
    # Drop leading wh-scaffolding.
    prev = None
    while prev != q:
        prev = q
        q = _WH_PREFIX_RE.sub("", q).strip()
    # "what school did ben roethlisberger go to" -> keep text after the FIRST
    # auxiliary pivot ("did"), so a later main verb ("...do before...") is not
    # mistaken for it.
    pivot = _DID_PIVOT_RE.search(q)
    if pivot is not None:
        q = q[pivot.end():].strip()
    # Drop a trailing relation clause.
    q = _RELATION_TAIL_RE.sub("", q).strip()
    return q.strip(" ,'\"")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS}


def _title_overlap(title: str, entity_terms: str) -> int:
    return len(_tokens(title) & _tokens(entity_terms))


def _title_has_all_significant(title: str, entity_terms: str) -> bool:
    # Significant = tokens with length >= 4 (skip initials like "j", "k").
    significant = {t for t in _tokens(entity_terms) if len(t) >= 4}
    return bool(significant) and significant <= _tokens(title)


class WikipediaProvider:
    """Fetches a full Wikipedia article extract for a question/entity."""

    def __init__(
        self,
        *,
        timeout_sec: float = 6.0,
        max_retries: int = 3,
        backoff_base_sec: float = 0.8,
        min_interval_sec: float = 1.2,
        opener=urlopen,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        self._http = ResilientHttpClient(
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_base_sec=backoff_base_sec,
            min_interval_sec=min_interval_sec,
            opener=opener,
            sleep=sleep,
            monotonic=monotonic,
        )

    def _fetch_page(self, search_terms: str) -> WebSearchResult | None:
        terms = (search_terms or "").strip()
        if not terms:
            return None
        url = (
            f"{_API}?action=query&format=json&redirects=1"
            f"&generator=search&gsrlimit=1&gsrsearch={quote_plus(terms)}"
            f"&prop=extracts&explaintext=1&exlimit=1"
        )
        body = self._http.get_text(url)
        if not body:
            return None
        try:
            data = json.loads(body)
        except json.JSONDecodeError:
            return None
        pages = (data.get("query") or {}).get("pages") or {}
        for page in pages.values():
            title = str(page.get("title") or "").strip()
            extract = str(page.get("extract") or "").strip()
            if title and extract:
                return WebSearchResult(
                    title=f"{title} - Wikipedia",
                    snippet=extract[:_MAX_EXTRACT_CHARS],
                    url=f"https://en.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}",
                )
        return None

    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        bare = bare_entity_query(query)
        full = self._fetch_page(query)

        # Relation words in the full question can drag the search to the wrong
        # page (e.g. "...fox marry" -> "Emilia Fox"). Disambiguate with a
        # bare-entity search and prefer whichever title better matches the
        # entity. Only pay the extra request when it can actually change the
        # answer (bare differs, or the full-question result looks off/empty).
        need_bare = bool(bare) and bare != (query or "").strip().lower()
        full_ok = full is not None and _title_has_all_significant(full.title, bare)
        if need_bare and not full_ok:
            alt = self._fetch_page(bare)
            if alt is not None:
                if full is None or _title_overlap(alt.title, bare) > _title_overlap(full.title, bare):
                    full = alt
        return [full] if full is not None else []
