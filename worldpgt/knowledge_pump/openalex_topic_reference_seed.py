"""Bounded, structured OpenAlex work extraction for predicate diversity."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from time import sleep
from typing import Any, Callable, Iterable
from urllib.parse import quote, urlsplit
from urllib.request import Request, urlopen
import json


_API = "https://api.openalex.org/works"
_EXTRACTION = "openalex_api_topic_reference_structured_v1"


def _compact(value: object) -> str:
    return " ".join(str(value or "").split())


def _norm(value: object) -> str:
    return _compact(value).casefold()


def doi_from_url(value: object) -> str:
    raw = _compact(value)
    if not raw:
        return ""
    parts = urlsplit(raw)
    return (parts.path.lstrip("/") if parts.netloc.casefold() == "doi.org" else raw).casefold()


def _get_json(url: str, *, user_agent: str) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": user_agent, "Accept": "application/json"})
    with urlopen(request, timeout=30) as response:  # nosec B310: official OpenAlex HTTPS API
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _work_url(identifier: str) -> str:
    return f"{_API}/{quote(identifier, safe=':')}?select=id,display_name,doi,topics,referenced_works"


def fetch_diverse_openalex_records(
    dois: Iterable[str],
    *,
    request_delay_sec: float,
    user_agent: str,
    get_json: Callable[[str], dict[str, Any]] | None = None,
    sleep_fn: Callable[[float], None] = sleep,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fetch each seed work plus at most one resolved citation endpoint."""

    if request_delay_sec < 0:
        raise ValueError("request_delay_sec must be non-negative")
    fetch = get_json or (lambda url: _get_json(url, user_agent=user_agent))
    seeds = sorted({_norm(doi) for doi in dois if _norm(doi)})
    works: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    attempted_queries = 0
    for index, doi in enumerate(seeds):
        if index:
            sleep_fn(request_delay_sec)
        try:
            attempted_queries += 1
            payload = fetch(_work_url(f"doi:{doi}"))
        except Exception as exc:
            errors.append({"stage": "seed_work", "identifier": doi, "error": str(exc)[:240]})
            continue
        payload["_seed_doi"] = doi
        works.append(payload)

    references: dict[str, dict[str, Any]] = {}
    for work in works:
        reference_ids = [str(value) for value in work.get("referenced_works", []) if isinstance(value, str)]
        if not reference_ids:
            continue
        ref_id = reference_ids[0].rsplit("/", 1)[-1]
        if ref_id in references:
            continue
        if works or references:
            sleep_fn(request_delay_sec)
        try:
            attempted_queries += 1
            references[ref_id] = fetch(_work_url(ref_id))
        except Exception as exc:
            errors.append({"stage": "reference_work", "identifier": ref_id, "error": str(exc)[:240]})
    return works, {
        "seed_query_count": len(seeds),
        "reference_lookup_count": len(references),
        "total_queries": attempted_queries,
        "errors": errors,
        "reference_records": references,
    }


def extract_topic_reference_rows(
    works: Iterable[dict[str, Any]],
    reference_records: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract exactly one topic and one named cited work per source work."""

    rows: list[dict[str, Any]] = []
    for work in works:
        work_id = _compact(work.get("id"))
        title = _compact(work.get("display_name"))
        if not work_id or not title:
            continue
        topics = [topic for topic in work.get("topics", []) if isinstance(topic, dict) and _compact(topic.get("display_name"))]
        if not topics:
            continue
        topic = max(topics, key=lambda item: float(item.get("score") or 0.0))
        topic_name = _compact(topic.get("display_name"))
        source_url = _work_url(work_id.rsplit("/", 1)[-1])
        common = {
            "overlay_type": "overlay_relation",
            "experimental_tier": "evidence_grounded_proposal",
            "canonical_openalex_id": work_id,
            "canonical_entity": work_id,
            "canonical_doi": doi_from_url(work.get("doi")),
            "subject": title,
            "source_page": "OpenAlex work metadata",
            "source_url": source_url,
            "source_kind": "openalex_api",
            "candidate_source": _EXTRACTION,
            "extraction_pattern": _EXTRACTION,
            "open_web_extraction": _EXTRACTION,
            "pump_source_kind": "open_web_proposal",
            "license_note": "OpenAlex public work metadata; proposal-only.",
            "trust": "proposal_open_web",
            "safe_for_general_runtime": False,
            "requires_review": True,
        }
        rows.append({
            **common,
            "predicate": "has_topic",
            "object": topic_name,
            "evidence_text": f'OpenAlex work metadata for "{title}" lists {topic_name} as its highest-scored topic.',
            "evidence_span": f'OpenAlex work metadata for "{title}" lists {topic_name} as its highest-scored topic.',
        })
        references = [str(value).rsplit("/", 1)[-1] for value in work.get("referenced_works", []) if isinstance(value, str)]
        if not references:
            continue
        reference = reference_records.get(references[0]) or {}
        reference_title = _compact(reference.get("display_name"))
        if not reference_title:
            continue
        reference_url = _work_url(references[0])
        rows.append({
            **common,
            "predicate": "references_work",
            "object": reference_title,
            "supporting_sources": [source_url, reference_url],
            "evidence_text": f'OpenAlex citation metadata for "{title}" lists "{reference_title}" among its referenced works.',
            "evidence_span": f'OpenAlex citation metadata for "{title}" lists "{reference_title}" among its referenced works.',
        })
    return rows


def select_topic_reference_entities(rows: Iterable[dict[str, Any]], *, max_entities: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Retain only source works with the distinct topic+citation pair."""

    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        identifier, predicate = _compact(row.get("canonical_openalex_id")), _compact(row.get("predicate"))
        if identifier and predicate:
            grouped[identifier][predicate].append(row)
    selected: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    for identifier in sorted(grouped):
        groups = grouped[identifier]
        if not {"has_topic", "references_work"}.issubset(groups):
            continue
        pair = [groups["has_topic"][0], groups["references_work"][0]]
        selected.extend(pair)
        manifest.append({
            "canonical_openalex_id": identifier,
            "title": pair[0]["subject"],
            "predicate_groups": ["has_topic", "references_work"],
            "relation_count": 2,
            "source_url": pair[0]["source_url"],
        })
        if len(manifest) >= max_entities:
            break
    return selected, manifest
