"""Proposal-only extraction of structured relations from Crossref DOI metadata.

This is a data-scaling lane, distinct from the quarantine re-parser: it asks
the official Crossref Works API for fresh DOI records and extracts only fields
the record itself explicitly provides.  It never promotes rows into accepted
or serving memory.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
from time import sleep
from typing import Any, Callable, Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen
import json


_API = "https://api.crossref.org/works"


def _compact(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm(value: object) -> str:
    return _compact(value).casefold()


def _authors(item: dict[str, Any]) -> list[str]:
    return [
        _compact(" ".join(str(author.get(key) or "") for key in ("given", "family")))
        for author in item.get("author", [])
        if isinstance(author, dict)
        and _compact(" ".join(str(author.get(key) or "") for key in ("given", "family")))
    ]


def _title(item: dict[str, Any]) -> str:
    values = item.get("title")
    return _compact(values[0]) if isinstance(values, list) and values else ""


def extract_doi_relation_rows(item: dict[str, Any], *, topic_bucket: str) -> list[dict[str, Any]]:
    """Return direct author/publisher facts from one Crossref work payload."""

    doi = _compact(item.get("DOI")).casefold()
    title = _title(item)
    publisher = _compact(item.get("publisher"))
    authors = _authors(item)
    if not doi or not title:
        return []
    source_url = f"{_API}/{quote(doi, safe='')}"
    common = {
        "overlay_type": "overlay_relation",
        "experimental_tier": "evidence_grounded_proposal",
        "canonical_doi": doi,
        "canonical_entity": doi,
        "subject": title,
        "source_page": "Crossref DOI metadata",
        "source_url": source_url,
        "source_kind": "crossref_doi",
        "topic_bucket": topic_bucket,
        "license_note": "Crossref public DOI metadata; proposal-only.",
        "candidate_source": "crossref_doi_structured_metadata_v1",
        "extraction_pattern": "crossref_doi_structured_metadata_v1",
        "open_web_extraction": "crossref_doi_structured_metadata_v1",
        "pump_source_kind": "open_web_proposal",
        "trust": "proposal_open_web",
        "safe_for_general_runtime": False,
        "requires_review": True,
    }
    rows: list[dict[str, Any]] = []
    for author in sorted(set(authors), key=_norm):
        rows.append({
            **common,
            "predicate": "created_by",
            "object": author,
            "evidence_text": f'Crossref DOI metadata for "{title}" lists {author} as an author.',
            "evidence_span": f'Crossref DOI metadata for "{title}" lists {author} as an author.',
        })
    if publisher:
        rows.append({
            **common,
            "predicate": "published_by",
            "object": publisher,
            "evidence_text": f'Crossref DOI metadata for "{title}" lists {publisher} as its publisher.',
            "evidence_span": f'Crossref DOI metadata for "{title}" lists {publisher} as its publisher.',
        })
    return rows


def select_multi_predicate_doi_rows(rows: Iterable[dict[str, Any]], *, max_entities: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Deduplicate by DOI and retain only works with two explicit fact types."""

    by_doi: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        doi, predicate = _compact(row.get("canonical_doi")).casefold(), _compact(row.get("predicate"))
        if doi and predicate:
            by_doi[doi][predicate].append(row)
    chosen: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for doi in sorted(by_doi, key=lambda value: sha256(value.encode()).hexdigest()):
        groups = by_doi[doi]
        if not {"created_by", "published_by"}.issubset(groups):
            continue
        all_rows = [row for predicate in sorted(groups) for row in groups[predicate]]
        title_key = _norm(all_rows[0]["subject"])
        if title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        chosen.extend(all_rows)
        manifest.append({
            "canonical_doi": doi,
            "title": all_rows[0]["subject"],
            "predicate_groups": sorted(groups),
            "relation_count": len(all_rows),
            "source_url": all_rows[0]["source_url"],
        })
        if len(manifest) >= max_entities:
            break
    return chosen, manifest


def _get_json(url: str, *, user_agent: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:  # nosec B310: official Crossref HTTPS API
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def fetch_crossref_doi_records(
    queries: Iterable[tuple[str, str]],
    *,
    max_queries: int,
    records_per_query: int,
    request_delay_sec: float,
    user_agent: str,
    get_json: Callable[[str], dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] = sleep,
) -> tuple[list[tuple[dict[str, Any], str]], dict[str, Any]]:
    """Fetch a bounded, deduplicated Crossref query batch for the seed lane."""

    if max_queries < 1 or not 1 <= records_per_query <= 200 or request_delay_sec < 0:
        raise ValueError("invalid bounded Crossref batch configuration")
    fetch = get_json or (lambda url: _get_json(url, user_agent=user_agent))
    records: list[tuple[dict[str, Any], str]] = []
    errors: list[dict[str, str]] = []
    seen_dois: set[str] = set()
    selected_queries = list(queries)[:max_queries]
    for index, (bucket, query) in enumerate(selected_queries):
        if index:
            sleep_fn(request_delay_sec)
        url = f"{_API}?query.bibliographic={quote(query, safe='')}&rows={records_per_query}"
        try:
            payload = fetch(url)
        except Exception as exc:  # keep a campaign resumable after one failed query
            errors.append({"bucket": bucket, "query": query, "error": str(exc)[:240]})
            continue
        message = payload.get("message") if isinstance(payload.get("message"), dict) else {}
        for item in message.get("items", []) if isinstance(message.get("items"), list) else []:
            if not isinstance(item, dict):
                continue
            doi = _compact(item.get("DOI")).casefold()
            if doi and doi not in seen_dois:
                seen_dois.add(doi)
                records.append((item, bucket))
    return records, {
        "requested_query_count": len(selected_queries),
        "records_per_query": records_per_query,
        "unique_doi_records": len(records),
        "errors": errors,
        "records_by_bucket": dict(sorted(Counter(bucket for _item, bucket in records).items())),
    }
