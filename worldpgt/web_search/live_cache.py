"""Volatile TTL cache for live web-search answers.

Deliberately separate from the accepted/promoted overlay: entries here never
passed through knowledge_pump's precision_firewall or promotion review, so
they must never be conflated with verified memory. They are also time-boxed,
because "current" facts (who leads a country, a live price, ...) go stale.

Two lookup granularities:

- **by-question** — exact repeat of a previously asked question. Fast path,
  unchanged in spirit from v1.
- **by-entity** — keyed by the resolved article/entity name (e.g. "Sanae
  Takaichi"), holding the full retrieved text. Once any question resolves an
  entity via a live search, every *later* question that names the same
  entity — even with completely different wording — is answered by running
  local (offline, sub-millisecond) answer extraction over the cached text
  instead of paying for another network round-trip. This is what lets the
  system get faster as it is used, without a network call per question.

A cache hit still renders through the exact same "live web search, not
Microworld memory" disclosure as a fresh network call — it only avoids
repeating the network round-trip within the TTL window.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

from worldpgt.assistant_surface.web_search import WebSearchResult

DEFAULT_TTL_HOURS = 168.0  # 7 days
_NORMALIZE_RE = re.compile(r"[^a-z0-9 ]+")
_SPACE_RE = re.compile(r" +")
_WIKI_SUFFIX_RE = re.compile(r"\s*[-–]\s*Wikipedia\s*$", re.IGNORECASE)


def _normalize_question(question: str) -> str:
    cleaned = _NORMALIZE_RE.sub(" ", (question or "").lower())
    return _SPACE_RE.sub(" ", cleaned).strip()


def entity_key_from_title(title: str) -> str:
    """Normalize a result title into an entity cache key.

    "Sanae Takaichi - Wikipedia" -> "sanae takaichi".
    """
    return _normalize_question(_WIKI_SUFFIX_RE.sub("", title or ""))


@dataclass(frozen=True)
class LiveCacheEntry:
    question: str
    fetched_at: str  # ISO 8601, UTC
    results: list[WebSearchResult] = field(default_factory=list)


def _results_to_dicts(results: list[WebSearchResult]) -> list[dict]:
    return [{"title": r.title, "snippet": r.snippet, "url": r.url} for r in results]


def _results_from_dicts(raw: list[dict]) -> list[WebSearchResult]:
    return [
        WebSearchResult(
            title=str(r.get("title") or ""),
            snippet=str(r.get("snippet") or ""),
            url=str(r.get("url") or ""),
        )
        for r in raw
    ]


class LiveSearchCache:
    """JSON-file-backed TTL cache for live web-search results.

    Stores two independent maps in one file: ``by_question`` (exact-question
    repeats) and ``by_entity`` (resolved entity name -> cached article, reused
    across any question naming that entity).
    """

    def __init__(self, path: str | Path, *, ttl_hours: float = DEFAULT_TTL_HOURS) -> None:
        self._path = Path(path)
        self._ttl_hours = ttl_hours

    def _load(self) -> dict:
        if not self._path.is_file():
            return {"by_question": {}, "by_entity": {}}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"by_question": {}, "by_entity": {}}
        if not isinstance(data, dict) or "by_question" not in data or "by_entity" not in data:
            # Unrecognized/legacy schema — this is a volatile cache, not
            # durable state, so start clean rather than migrate.
            return {"by_question": {}, "by_entity": {}}
        return data

    def _save(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

    def _entry_from_raw(self, raw: dict | None, fallback_label: str) -> LiveCacheEntry | None:
        if not raw:
            return None
        fetched_at = str(raw.get("fetched_at") or "")
        try:
            fetched = datetime.fromisoformat(fetched_at)
        except ValueError:
            return None
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        if datetime.now(timezone.utc) - fetched > timedelta(hours=self._ttl_hours):
            return None
        results = _results_from_dicts(raw.get("results") or [])
        if not results:
            return None
        return LiveCacheEntry(
            question=str(raw.get("question") or fallback_label),
            fetched_at=fetched_at,
            results=results,
        )

    # -- exact-question lookup (fast path for literal repeats) ------------- #
    def get(self, question: str) -> LiveCacheEntry | None:
        key = _normalize_question(question)
        if not key:
            return None
        return self._entry_from_raw(self._load().get("by_question", {}).get(key), question)

    def put(self, question: str, results: list[WebSearchResult]) -> None:
        key = _normalize_question(question)
        if not key or not results:
            return
        data = self._load()
        data["by_question"][key] = {
            "question": question,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results": _results_to_dicts(results),
        }
        self._save(data)

    # -- entity lookup (reused across any question naming the entity) ------ #
    def get_entity(self, entity_key: str) -> LiveCacheEntry | None:
        key = _normalize_question(entity_key)
        if not key:
            return None
        return self._entry_from_raw(self._load().get("by_entity", {}).get(key), entity_key)

    def put_entity(self, entity_key: str, results: list[WebSearchResult]) -> None:
        key = _normalize_question(entity_key)
        if not key or not results:
            return
        data = self._load()
        data["by_entity"][key] = {
            "question": entity_key,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
            "results": _results_to_dicts(results),
        }
        self._save(data)

    def find_entity_in_question(self, question: str) -> LiveCacheEntry | None:
        """Return the cached entity whose name is the longest match inside
        ``question`` (case/punctuation-insensitive substring), or None.

        This is what lets a *new* question ("When was she born?" rewritten to
        "When was Sanae Takaichi born?") reuse an entity learned from a
        *previous, differently-worded* question, instead of hitting the
        network again.
        """
        norm_q = _normalize_question(question)
        if not norm_q:
            return None
        by_entity = self._load().get("by_entity", {})
        best_key: str | None = None
        for key in by_entity:
            # Require >=5 chars so a short/common name (e.g. "fox") can't
            # collide with an unrelated question that merely contains it.
            if len(key) < 5:
                continue
            if norm_q == key or f" {key} " in f" {norm_q} ":
                if best_key is None or len(key) > len(best_key):
                    best_key = key
        if best_key is None:
            return None
        return self._entry_from_raw(by_entity.get(best_key), best_key)
