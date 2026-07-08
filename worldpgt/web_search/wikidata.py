"""Wikidata structured-facts provider for the live web-search fallback.

Official Wikidata API (``wbsearchentities`` + ``wbgetentities``) — no API key,
no signup, and (unlike scraping a page meant for humans) not something that
gets bot-detected and soft-blocked the way DuckDuckGo's HTML endpoints do.

Wikipedia's full-text extract is rich but requires the answer-extraction
layer to *find* the specific fact inside free-form prose, which is exactly
where precision misses happen (a relation keyword shows up in an unrelated
sentence, or the right fact is written in a phrasing the extractor doesn't
recognize). Wikidata instead holds facts as explicit (subject, property,
value) claims — "date of birth", "educated at", "spouse" — so this renders
each claim as its own plain, unambiguous sentence: "X was educated at Y."
That sentence already names its own relation, which is what
``answer_extraction.extract_answer`` scores on, so it tends to be more
precisely on-target than a mined sentence from prose.

Deterministic and rule-based like the rest of this project: a fixed
property -> sentence-template map, no ML, no free-text generation. Only the
curated properties below are ever rendered; anything else on the entity is
ignored rather than guessed at.
"""

from __future__ import annotations

import json
import re
import time
from datetime import datetime, timezone
from urllib.parse import quote_plus
from urllib.request import urlopen

from worldpgt.assistant_surface.web_search import WebSearchResult
from worldpgt.web_search.http import ResilientHttpClient, SharedRateLimiter
from worldpgt.web_search.wikipedia import bare_entity_query

_API = "https://www.wikidata.org/w/api.php"
_SPARQL_API = "https://query.wikidata.org/sparql"
_MAX_SNIPPET_CHARS = 2000

# property id -> (sentence template with {subj}/{value}, value kind hint).
# Order here is also render order (roughly biography-shaped, then place/org
# facts), so the snippet reads coherently top to bottom.
_PROPERTY_TEMPLATES: dict[str, str] = {
    "P1308": "The officeholder of {subj} is {value}.",
    "P569": "{subj} was born on {value}.",
    "P19": "{subj} was born in {value}.",
    "P570": "{subj} died on {value}.",
    "P20": "{subj} died in {value}.",
    "P26": "{subj} was married to {value}.",
    "P69": "{subj} was educated at {value}.",
    "P106": "{subj} worked as {value}.",
    "P39": "{subj} held the position of {value}.",
    "P108": "{subj} was employed by {value}.",
    "P54": "{subj} was a member of {value}.",
    "P27": "{subj} is a citizen of {value}.",
    "P112": "{subj} was founded by {value}.",
    "P170": "{subj} was created by {value}.",
    "P50": "{subj} was written by {value}.",
    "P571": "{subj} was founded/established on {value}.",
    "P36": "The capital of {subj} is {value}.",
    "P37": "The official language of {subj} is {value}.",
    "P38": "The currency of {subj} is {value}.",
    "P1082": "{subj} has a population of {value}.",
    "P159": "{subj} is headquartered in {value}.",
}

_OFFICE_QUERY_RE = re.compile(
    r"\b(?:current\s+)?(?P<office>president|prime minister|mayor|governor)\s+of\s+"
    r"(?:the\s+)?(?P<place>[a-z .'-]+)",
    re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(1[6-9]\d{2}|20\d{2}|21\d{2})\b")

_PLACE_ALIASES = {
    "us": "the United States",
    "u.s": "the United States",
    "u.s.": "the United States",
    "usa": "the United States",
    "u.s.a": "the United States",
    "u.s.a.": "the United States",
    "america": "the United States",
    "united states": "the United States",
    "united states of america": "the United States",
}

_TIME_PRECISION_YEAR = 9
_TIME_PRECISION_MONTH = 10
_TIME_PRECISION_DAY = 11


def _format_time(value: dict) -> str | None:
    raw = str(value.get("time") or "")
    if not raw:
        return None
    sign = -1 if raw.startswith("-") else 1
    raw = raw.lstrip("+-")
    try:
        year_str, rest = raw.split("-", 1)
        month_str, day_str = rest.split("-", 1)
        day_str = day_str.split("T", 1)[0]
        year, month, day = int(year_str) * sign, int(month_str), int(day_str)
    except (ValueError, IndexError):
        return None
    precision = value.get("precision", _TIME_PRECISION_DAY)
    if precision < _TIME_PRECISION_YEAR or year <= 0:
        return None
    if precision == _TIME_PRECISION_YEAR or month == 0:
        return str(year)
    if precision == _TIME_PRECISION_MONTH or day == 0:
        try:
            return datetime(year, month, 1).strftime("%B %Y")
        except ValueError:
            return str(year)
    try:
        return datetime(year, month, day).strftime("%B %d, %Y")
    except ValueError:
        return str(year)


def _format_quantity(value: dict) -> str | None:
    amount = str(value.get("amount") or "").lstrip("+")
    if not amount:
        return None
    try:
        num = float(amount)
    except ValueError:
        return amount
    if num == int(num):
        return f"{int(num):,}"
    return f"{num:,.2f}"


def officeholder_entity_query(question: str) -> str | None:
    """Normalize current-office questions to the Wikidata item for the office.

    A query like "Who is the current president of the US?" should search for
    "President of the United States", not the generic phrase "current
    president of the us".
    """

    m = _OFFICE_QUERY_RE.search((question or "").strip().rstrip("?.!"))
    if not m:
        return None
    office = " ".join(m.group("office").lower().split())
    place = " ".join(m.group("place").lower().strip(" .'").split())
    place = re.sub(r"\b(?:in|during|as\s+of)$", "", place).strip()
    place_label = _PLACE_ALIASES.get(place, place.title())
    if office == "prime minister":
        office_label = "Prime Minister"
    else:
        office_label = office.title()
    return f"{office_label} of {place_label}"


def officeholder_query_year(question: str) -> int | None:
    match = _YEAR_RE.search(question or "")
    return int(match.group(1)) if match else None


class WikidataProvider:
    """Fetches curated structured facts for a question's entity from Wikidata."""

    def __init__(
        self,
        *,
        timeout_sec: float = 6.0,
        max_retries: int = 2,
        backoff_base_sec: float = 0.5,
        min_interval_sec: float = 0.3,
        rate_limiter: SharedRateLimiter | None = None,
        opener=urlopen,
        sleep=time.sleep,
        monotonic=time.monotonic,
    ) -> None:
        self._http = ResilientHttpClient(
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            backoff_base_sec=backoff_base_sec,
            min_interval_sec=min_interval_sec,
            rate_limiter=rate_limiter,
            opener=opener,
            sleep=sleep,
            monotonic=monotonic,
        )

    def _get_json(self, url: str) -> dict | None:
        body = self._http.get_text(url)
        if not body:
            return None
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            return None

    def _search_entity(self, terms: str) -> dict | None:
        terms = (terms or "").strip()
        if not terms:
            return None
        url = (
            f"{_API}?action=wbsearchentities&search={quote_plus(terms)}"
            f"&language=en&format=json&limit=1&type=item"
        )
        data = self._get_json(url)
        if not data:
            return None
        results = data.get("search") or []
        return results[0] if results else None

    def _get_claims(self, qid: str) -> dict | None:
        url = (
            f"{_API}?action=wbgetentities&ids={qid}&languages=en"
            f"&props=claims|labels|descriptions&format=json"
        )
        data = self._get_json(url)
        if not data:
            return None
        return (data.get("entities") or {}).get(qid)

    def _resolve_labels(self, qids: set[str]) -> dict[str, str]:
        if not qids:
            return {}
        url = (
            f"{_API}?action=wbgetentities&ids={'|'.join(sorted(qids))}"
            f"&languages=en&props=labels|sitelinks&sitefilter=enwiki&format=json"
        )
        data = self._get_json(url)
        if not data:
            return {}
        labels: dict[str, str] = {}
        for qid, entity in (data.get("entities") or {}).items():
            label = ((entity.get("labels") or {}).get("en") or {}).get("value")
            if not label:
                label = ((entity.get("sitelinks") or {}).get("enwiki") or {}).get("title")
            if label:
                labels[qid] = label
        return labels

    def _current_officeholder_label(self, office_qid: str) -> str | None:
        query = f"""
        SELECT ?holder ?holderLabel ?start WHERE {{
          ?holder p:P39 ?statement .
          ?statement ps:P39 wd:{office_qid} .
          OPTIONAL {{ ?statement pq:P580 ?start . }}
          FILTER NOT EXISTS {{ ?statement pq:P582 ?end . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?start)
        LIMIT 1
        """
        url = f"{_SPARQL_API}?query={quote_plus(query)}&format=json"
        data = self._get_json(url)
        if not data:
            return None
        bindings = ((data.get("results") or {}).get("bindings") or [])
        if not bindings:
            return None
        binding = bindings[0]
        label = ((binding.get("holderLabel") or {}).get("value") or "").strip()
        if re.fullmatch(r"Q\d+", label):
            holder_uri = ((binding.get("holder") or {}).get("value") or "").strip()
            holder_qid = holder_uri.rsplit("/", 1)[-1] if holder_uri else label
            labels = self._resolve_labels({holder_qid})
            label = labels.get(holder_qid, label)
        return label or None

    def _officeholder_label_for_year(self, office_qid: str, year: int) -> str | None:
        start_bound = f"{year}-12-31T23:59:59Z"
        end_bound = f"{year}-01-01T00:00:00Z"
        query = f"""
        SELECT ?holder ?holderLabel ?start WHERE {{
          ?holder p:P39 ?statement .
          ?statement ps:P39 wd:{office_qid} .
          OPTIONAL {{ ?statement pq:P580 ?start . }}
          OPTIONAL {{ ?statement pq:P582 ?end . }}
          FILTER(!BOUND(?start) || ?start <= "{start_bound}"^^xsd:dateTime)
          FILTER(!BOUND(?end) || ?end >= "{end_bound}"^^xsd:dateTime)
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        ORDER BY DESC(?start)
        LIMIT 1
        """
        url = f"{_SPARQL_API}?query={quote_plus(query)}&format=json"
        data = self._get_json(url)
        if not data:
            return None
        bindings = ((data.get("results") or {}).get("bindings") or [])
        if not bindings:
            return None
        binding = bindings[0]
        label = ((binding.get("holderLabel") or {}).get("value") or "").strip()
        if re.fullmatch(r"Q\d+", label):
            holder_uri = ((binding.get("holder") or {}).get("value") or "").strip()
            holder_qid = holder_uri.rsplit("/", 1)[-1] if holder_uri else label
            labels = self._resolve_labels({holder_qid})
            label = labels.get(holder_qid, label)
        return label or None

    def search(self, query: str, *, max_results: int = 3) -> list[WebSearchResult]:
        office_terms = officeholder_entity_query(query)
        terms = office_terms or bare_entity_query(query) or query
        match = self._search_entity(terms)
        if match is None:
            return []

        qid = str(match.get("id") or "")
        subject = str(match.get("label") or match.get("display", {}).get("label", {}).get("value") or terms)
        if not qid:
            return []

        if office_terms:
            year = officeholder_query_year(query)
            if year is not None:
                holder = self._officeholder_label_for_year(qid, year)
                if holder:
                    return [WebSearchResult(
                        title=f"{subject} - Wikidata",
                        snippet=f"The officeholder of {subject} in {year} was {holder}.",
                        url=f"https://www.wikidata.org/wiki/{qid}",
                    )]
            holder = self._current_officeholder_label(qid)
            if holder:
                return [WebSearchResult(
                    title=f"{subject} - Wikidata",
                    snippet=f"The officeholder of {subject} is {holder}.",
                    url=f"https://www.wikidata.org/wiki/{qid}",
                )]

        entity = self._get_claims(qid)
        if not entity:
            return []
        claims = entity.get("claims") or {}
        property_ids = ("P1308",) if office_terms else tuple(_PROPERTY_TEMPLATES)

        # First pass: pull raw values, collecting entity-ref QIDs to resolve.
        raw_values: list[tuple[str, dict]] = []
        ref_qids: set[str] = set()
        for pid in property_ids:
            prop_claims = claims.get(pid)
            if not prop_claims:
                continue
            mainsnak = prop_claims[0].get("mainsnak") or {}
            datavalue = mainsnak.get("datavalue")
            if not datavalue:
                continue
            raw_values.append((pid, datavalue))
            if datavalue.get("type") == "wikibase-entityid":
                ref_id = datavalue.get("value", {}).get("id")
                if ref_id:
                    ref_qids.add(ref_id)

        labels = self._resolve_labels(ref_qids)

        sentences: list[str] = []
        for pid, datavalue in raw_values:
            kind = datavalue.get("type")
            value = datavalue.get("value")
            rendered: str | None = None
            if kind == "time":
                rendered = _format_time(value)
            elif kind == "quantity":
                rendered = _format_quantity(value)
            elif kind == "wikibase-entityid":
                rendered = labels.get(value.get("id"))
            elif kind in ("string", "monolingualtext"):
                rendered = value.get("text") if isinstance(value, dict) else str(value)
            if not rendered:
                continue
            sentences.append(_PROPERTY_TEMPLATES[pid].format(subj=subject, value=rendered))

        if not sentences:
            return []

        snippet = " ".join(sentences)[:_MAX_SNIPPET_CHARS]
        return [WebSearchResult(
            title=f"{subject} - Wikidata",
            snippet=snippet,
            url=f"https://www.wikidata.org/wiki/{qid}",
        )]
