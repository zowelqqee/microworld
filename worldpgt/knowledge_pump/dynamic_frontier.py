"""Dynamic Wikipedia graph frontier for Knowledge Pump v1.

The static frontier is now only a seed. Successful snapshot fetches can add
namespace-0 Wikipedia links to this proposal-only frontier artifact, so later
pump runs can keep crawling without hand-editing title lists.
"""

from __future__ import annotations

import csv
import json
import re
import urllib.parse
from pathlib import Path
from typing import Any

from worldpgt.knowledge_pump.frontier_title_extractor import is_usable_frontier_title
from worldpgt.knowledge_pump.title_ranker import normalize_title
from worldpgt.knowledge_pump.types import ExpandedAllowlistEntry, FrontierTitle

_WIKI_BRACKET_RE = re.compile(r"\[\[([^]|#]+)(?:\|[^]]+)?\]\]")
_WIKI_URL_RE = re.compile(r"(?:https?://en\.wikipedia\.org)?/wiki/([^)\]\s#?]+)")


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _norm_key(title: str) -> str:
    return normalize_title(title).casefold()


def extract_internal_link_titles(text: str) -> list[str]:
    """Extract explicit Wikipedia internal links from raw/markdown-like text."""
    out: list[str] = []
    seen: set[str] = set()
    for match in _WIKI_BRACKET_RE.finditer(text or ""):
        title = normalize_title(match.group(1))
        key = _norm_key(title)
        if title and key not in seen:
            out.append(title)
            seen.add(key)
    for match in _WIKI_URL_RE.finditer(text or ""):
        raw = urllib.parse.unquote(match.group(1)).replace("_", " ")
        title = normalize_title(raw)
        key = _norm_key(title)
        if title and key not in seen:
            out.append(title)
            seen.add(key)
    return out


def load_dynamic_frontier(path: str | Path) -> list[FrontierTitle]:
    rows = _read_json(Path(path), [])
    out: list[FrontierTitle] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        try:
            out.append(
                FrontierTitle(
                    title=str(row.get("title") or ""),
                    source=str(row.get("source") or "dynamic_wiki_link"),
                    reason=str(row.get("reason") or "dynamic frontier"),
                    weight=int(row.get("weight") or 1),
                )
            )
        except (TypeError, ValueError):
            continue
    return [item for item in out if item.title]


def write_dynamic_frontier(frontier: list[FrontierTitle], json_path: str | Path, csv_path: str | Path) -> None:
    rows = [item.to_dict() for item in sorted(frontier, key=lambda x: (-x.weight, x.title.casefold()))]
    _write_json(Path(json_path), rows)
    csv_out = Path(csv_path)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["title", "source", "reason", "weight"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def merge_frontiers(frontiers: list[list[FrontierTitle]]) -> list[FrontierTitle]:
    merged: dict[str, FrontierTitle] = {}
    for frontier in frontiers:
        for item in frontier:
            title = normalize_title(item.title)
            if not title:
                continue
            key = title.casefold()
            if key in merged:
                merged[key].weight += item.weight
                if item.source not in merged[key].source.split("|"):
                    merged[key].source += f"|{item.source}"
                continue
            merged[key] = FrontierTitle(title, item.source, item.reason, item.weight)
    return sorted(merged.values(), key=lambda x: (-x.weight, x.title.casefold()))


def _allowlist_titles(allowlist: list[ExpandedAllowlistEntry]) -> set[str]:
    out: set[str] = set()
    for entry in allowlist:
        out.add(_norm_key(entry.title))
        out.add(_norm_key(entry.normalized_title))
    out.discard("")
    return out


def _raw_snapshot_for_row(row: dict[str, Any]) -> dict[str, Any]:
    raw_path = Path(str(row.get("raw_snapshot_path") or ""))
    if raw_path.exists():
        raw = _read_json(raw_path, {})
        if isinstance(raw, dict):
            return raw
    return {}


def _candidate_titles_from_row(row: dict[str, Any]) -> list[str]:
    raw = _raw_snapshot_for_row(row)
    titles: list[str] = []
    for value in raw.get("links") or row.get("links") or []:
        if isinstance(value, str):
            titles.append(value)
        elif isinstance(value, dict):
            titles.append(str(value.get("title") or ""))
    titles.extend(extract_internal_link_titles(str(raw.get("raw_text") or "")))
    titles.extend(extract_internal_link_titles(str(row.get("raw_text") or "")))

    seen: set[str] = set()
    out: list[str] = []
    for title in titles:
        clean = normalize_title(title)
        key = clean.casefold()
        if clean and key not in seen:
            out.append(clean)
            seen.add(key)
    return out


def update_dynamic_frontier_from_fetch_rows(
    rows: list[dict[str, Any]],
    *,
    dynamic_frontier_path: str | Path,
    dynamic_frontier_csv_path: str | Path,
    already_fetched_titles: set[str],
    current_allowlist: list[ExpandedAllowlistEntry],
) -> dict[str, Any]:
    existing = load_dynamic_frontier(dynamic_frontier_path)
    by_key = {item.title.casefold(): item for item in existing}
    fetched = {_norm_key(title) for title in already_fetched_titles}
    allowlisted = _allowlist_titles(current_allowlist)

    pages_processed = 0
    candidate_count = 0
    rejected_already_fetched = 0
    rejected_current_allowlist = 0
    rejected_hygiene = 0
    added: list[str] = []

    for row in rows:
        if row.get("fetch_status") != "success":
            continue
        pages_processed += 1
        source_page = normalize_title(str(row.get("normalized_title") or row.get("title") or ""))
        source_key = source_page.casefold()
        for title in _candidate_titles_from_row(row):
            candidate_count += 1
            key = title.casefold()
            if key == source_key or key in fetched:
                rejected_already_fetched += 1
                continue
            if key in allowlisted:
                rejected_current_allowlist += 1
                continue
            if not is_usable_frontier_title(title, "dynamic_wiki_link"):
                rejected_hygiene += 1
                continue
            if key in by_key:
                by_key[key].weight += 1
                continue
            by_key[key] = FrontierTitle(
                title=title,
                source="dynamic_wiki_link",
                reason=f"internal Wikipedia link from {source_page}",
                weight=6,
            )
            added.append(title)

    updated = sorted(by_key.values(), key=lambda x: (-x.weight, x.title.casefold()))
    write_dynamic_frontier(updated, dynamic_frontier_path, dynamic_frontier_csv_path)
    return {
        "dynamic_frontier_previous_total": len(existing),
        "dynamic_frontier_added_this_run": len(added),
        "dynamic_frontier_total": len(updated),
        "dynamic_frontier_pages_processed": pages_processed,
        "dynamic_frontier_candidate_count": candidate_count,
        "dynamic_frontier_rejected_already_fetched": rejected_already_fetched,
        "dynamic_frontier_rejected_current_allowlist": rejected_current_allowlist,
        "dynamic_frontier_rejected_hygiene": rejected_hygiene,
        "dynamic_frontier_added_sample": added[:20],
    }
