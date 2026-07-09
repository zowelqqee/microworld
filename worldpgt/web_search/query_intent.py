"""Small deterministic helpers for live web-search question intent.

The web-search fallback is only useful when three things line up: the query
names the subject, the retrieved source is about that subject, and the chosen
sentence matches the relation asked. These helpers keep that bookkeeping out
of individual providers while staying stdlib-only and conservative.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from worldpgt.assistant_surface.web_search import WebSearchResult
from worldpgt.web_search.wikipedia import bare_entity_query

_WORD_RE = re.compile(r"[a-z0-9]+")
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")
_STOPWORDS = frozenset({
    "the", "a", "an", "of", "in", "on", "at", "and", "or", "to", "for",
    "with", "from", "is", "are", "was", "were", "did", "does", "do",
    "who", "what", "when", "where", "which", "whom", "whose", "why",
    "how", "current", "now", "people", "person", "called", "kind", "type",
})
_RELATION_WORDS = frozenset({
    "speak", "speaks", "spoken", "language", "languages", "born", "birth",
    "birthplace", "from", "died", "death", "married", "marry", "spouse",
    "wife", "husband", "college", "school", "university", "attend",
    "attended", "educated", "go", "went", "play", "plays", "played",
    "team", "invent", "invented", "discover", "discovered", "write",
    "wrote", "written", "author", "capital", "currency", "government",
    "governor", "president", "mayor", "minister", "officeholder",
    "timezone", "time", "zone", "famous", "known", "work", "works",
})
_COUNTRY_PEOPLE_RE = re.compile(
    r"\b(?:what\s+)?(?:language|languages)\s+"
    r"(?P<place>[a-z][a-z .'-]+?)\s+(?:people|person)s?\s+speaks?\b"
    r"|\bwhat\s+does\s+(?P<place2>[a-z][a-z .'-]+?)\s+people\s+speak\b",
    re.IGNORECASE,
)
_TIMEZONE_RE = re.compile(
    r"\b(?:timezone|time\s+zone)\b.*\b(?:in|of)\s+(?P<place>[a-z][a-z .'-]+)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class WebQueryIntent:
    original_query: str
    subject_query: str
    search_query: str
    relation_terms: tuple[str, ...] = ()
    year: int | None = None


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _clean_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip(" ?.!,'\"")).strip()


def _year(question: str) -> int | None:
    match = _YEAR_RE.search(question or "")
    return int(match.group(1)) if match else None


def _content_terms(text: str) -> set[str]:
    return {
        t for t in _tokens(text)
        if t not in _STOPWORDS and t not in _RELATION_WORDS and len(t) > 2
    }


def build_web_query_intent(question: str) -> WebQueryIntent:
    """Build a slightly more targeted live-search query for factoid QA."""

    q = _clean_phrase(question)
    lowered = q.lower()
    relation_terms: list[str] = []
    subject_query = bare_entity_query(q) or q
    search_query = q

    m = _COUNTRY_PEOPLE_RE.search(lowered)
    if m:
        place = _clean_phrase(m.group("place") or m.group("place2") or "")
        if place:
            subject_query = place
            relation_terms = ["language"]
            search_query = f"{place} languages spoken"
            return WebQueryIntent(q, subject_query, search_query, tuple(relation_terms), _year(q))

    m = _TIMEZONE_RE.search(lowered)
    if m:
        place = _clean_phrase(m.group("place"))
        if place:
            subject_query = place
            relation_terms = ["time", "zone", "timezone"]
            search_query = f"{place} time zone"
            return WebQueryIntent(q, subject_query, search_query, tuple(relation_terms), _year(q))

    relation_map: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
        (("born", "birthplace", "from"), ("born", "birthplace")),
        (("died", "death", "die"), ("died", "death")),
        (("married", "marry", "wife", "husband", "spouse"), ("married", "spouse")),
        (("college", "school", "university", "attend", "attended"), ("educated", "college", "school")),
        (("play", "plays", "played", "team"), ("played", "team")),
        (("currency",), ("currency",)),
        (("capital",), ("capital",)),
        (("government",), ("government",)),
        (("famous", "known"), ("known", "famous")),
        (("work", "works", "mechanism"), ("works", "mechanism")),
    )
    qtokens = set(_tokens(lowered))
    for triggers, terms in relation_map:
        if qtokens & set(triggers):
            relation_terms.extend(terms)

    if relation_terms and subject_query:
        deduped_terms = tuple(dict.fromkeys(relation_terms))
        # bare_entity_query() doesn't strip every relation word from the
        # subject (e.g. "capital" in "what capital of austria?" stays part
        # of the bare entity) -- appending it again here produced a
        # self-duplicating, worse-matching query ("capital of austria
        # capital"). Only append terms the subject doesn't already contain.
        subject_tokens = set(_tokens(subject_query))
        new_terms = [t for t in deduped_terms if t not in subject_tokens]
        search_query = f"{subject_query} {' '.join(new_terms)}".strip() if new_terms else subject_query
        year = _year(q)
        if year is not None:
            search_query = f"{search_query} {year}"
        return WebQueryIntent(q, subject_query, search_query, deduped_terms, year)

    return WebQueryIntent(q, subject_query, search_query, tuple(), _year(q))


def _subject_overlap(intent: WebQueryIntent, result: WebSearchResult) -> int:
    """How many subject-query tokens the result's title or snippet names.

    Kept separate from the full score: relation terms are often ordinary
    English words ("time", "known") that coincidentally appear in an
    unrelated snippet, so a nonzero *total* score alone doesn't prove the
    result is actually about the right subject — only real subject-token
    overlap does.
    """

    title_text = re.sub(r"\s+-\s+(?:Wikipedia|Wikidata)$", "", result.title or "", flags=re.IGNORECASE)
    title_terms = _content_terms(title_text)
    subject_terms = _content_terms(intent.subject_query)
    snippet_terms = set(_tokens(result.snippet or ""))
    return len(title_terms & subject_terms) + len(snippet_terms & subject_terms)


def result_relevance_score(intent: WebQueryIntent, result: WebSearchResult) -> float:
    """Score whether a retrieved result is about the intended subject/relation."""

    title_text = re.sub(r"\s+-\s+(?:Wikipedia|Wikidata)$", "", result.title or "", flags=re.IGNORECASE)
    title_terms = _content_terms(title_text)
    subject_terms = _content_terms(intent.subject_query)
    snippet_terms = set(_tokens(result.snippet or ""))

    score = 0.0
    if subject_terms:
        title_overlap = len(title_terms & subject_terms)
        snippet_overlap = len(snippet_terms & subject_terms)
        score += title_overlap * 4 + min(snippet_overlap, 3)
        if title_overlap == len(subject_terms):
            score += 4
    if intent.relation_terms:
        relation_overlap = len(set(intent.relation_terms) & snippet_terms)
        score += relation_overlap * 2
    if intent.year is not None and str(intent.year) in snippet_terms:
        score += 3
    return score


def is_relevant_result(intent: WebQueryIntent, result: WebSearchResult) -> bool:
    subject_terms = _content_terms(intent.subject_query)
    if not subject_terms:
        return True
    score = result_relevance_score(intent, result)
    return score >= max(4.0, min(8.0, len(subject_terms) * 3.0))


def filter_and_rank_results(
    question: str,
    results: list[WebSearchResult],
) -> tuple[WebQueryIntent, list[WebSearchResult]]:
    """Rank results by relevance, but don't let a near-miss reject everything.

    ``is_relevant_result``'s token-overlap score is brittle for anything
    short of an exact-title match — a nickname ("Ben" vs "Benjamin"), a
    partial name, or a differently-worded subject can legitimately score
    below its threshold even when the result is the right page. Silently
    turning "we found something plausible" into "we found nothing" cost real
    answer-rate for no offsetting precision gain (the disclosure in
    ``render_web_answer`` already marks every live result unverified).

    So: results that clear the strict threshold are used as-is. If none do,
    fall back to results that at least name the subject somewhere (title or
    snippet) — that covers a near-miss like a nickname, without being fooled
    by a relation term ("time", "known") that coincidentally appears in an
    unrelated snippet. A result that never mentions the subject at all (a
    different topic entirely, e.g. a TV-series page for a geography
    question) stays rejected, so a composite search can still fall through
    to a better provider instead of keeping a wrong-topic result.
    """

    intent = build_web_query_intent(question)
    ranked = sorted(results, key=lambda r: result_relevance_score(intent, r), reverse=True)
    filtered = [r for r in ranked if is_relevant_result(intent, r)]
    if filtered:
        return intent, filtered
    plausible = [r for r in ranked if _subject_overlap(intent, r) > 0]
    return intent, plausible
