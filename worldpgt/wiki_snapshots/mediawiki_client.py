"""Controlled MediaWiki Action API client for allowlisted snapshots."""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from typing import Iterable

from worldpgt.wiki_snapshots.types import PageSnapshot

DEFAULT_API_ENDPOINT = "https://en.wikipedia.org/w/api.php"
LICENSE_NOTE = (
    "Wikipedia text is available under the Creative Commons Attribution-ShareAlike "
    "License unless otherwise noted; this local file is an untrusted source snapshot."
)
GENERIC_USER_AGENTS = {"", "python-urllib", "python-requests", "urllib", "requests", "bot"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def source_url_for_title(title: str) -> str:
    normalized = title.strip().replace(" ", "_")
    return "https://en.wikipedia.org/wiki/" + urllib.parse.quote(normalized, safe="()_,.%")


def is_meaningful_user_agent(user_agent: str | None) -> bool:
    ua = (user_agent or "").strip()
    if not ua:
        return False
    lowered = ua.lower()
    if lowered in GENERIC_USER_AGENTS:
        return False
    if "python-urllib" in lowered or "python-requests" in lowered:
        return False
    return "/" in ua and ("contact:" in lowered or "local research" in lowered)


class MediaWikiClient:
    def __init__(
        self,
        allowed_titles: Iterable[str],
        allow_network: bool = False,
        user_agent: str | None = None,
        endpoint: str = DEFAULT_API_ENDPOINT,
        timeout_sec: float = 20.0,
        delay_sec: float = 0.5,
        retries: int = 2,
    ) -> None:
        self.allowed_titles = set(allowed_titles)
        self.allow_network = allow_network
        self.user_agent = user_agent or ""
        self.endpoint = endpoint
        self.timeout_sec = timeout_sec
        self.delay_sec = delay_sec
        self.retries = retries
        self.network_calls = 0
        if self.allow_network and not is_meaningful_user_agent(self.user_agent):
            raise ValueError("network fetch requires a meaningful User-Agent with contact context")

    def build_api_url(self, title: str) -> str:
        if title not in self.allowed_titles:
            raise ValueError(f"title is not allowlisted: {title}")
        params = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "prop": "extracts|revisions|info",
            "explaintext": "1",
            "exsectionformat": "plain",
            "rvprop": "ids|timestamp",
            "inprop": "url",
            "redirects": "1",
            "titles": title,
        }
        return self.endpoint + "?" + urllib.parse.urlencode(params)

    def fetch_page(self, title: str) -> PageSnapshot:
        if not self.allow_network:
            raise PermissionError("network fetch refused without allow_network=True")
        if not is_meaningful_user_agent(self.user_agent):
            raise ValueError("network fetch requires a meaningful User-Agent")
        if title not in self.allowed_titles:
            raise ValueError(f"title is not allowlisted: {title}")

        api_url = self.build_api_url(title)
        last_error = ""
        for attempt in range(self.retries + 1):
            if self.network_calls > 0 or attempt > 0:
                time.sleep(self.delay_sec + (0.25 * attempt))
            try:
                request = urllib.request.Request(api_url, headers={"User-Agent": self.user_agent})
                self.network_calls += 1
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    body = response.read().decode("utf-8")
                return self._snapshot_from_response(title, api_url, body)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = str(exc)
                if attempt >= self.retries:
                    break
        return self._error_snapshot(title, api_url, last_error)

    def _snapshot_from_response(self, requested_title: str, api_url: str, body: str) -> PageSnapshot:
        retrieved_at = utc_now_iso()
        try:
            data = json.loads(body)
            pages = data.get("query", {}).get("pages", [])
            page = pages[0] if pages else {}
            normalized_title = str(page.get("title") or requested_title)
            missing = bool(page.get("missing"))
            raw_text = "" if missing else str(page.get("extract") or "")
            revisions = page.get("revisions") or []
            revision = revisions[0] if revisions else {}
            revision_id = revision.get("revid")
            timestamp = str(revision.get("timestamp") or "")
            status = "missing" if missing else "success"
            error = "page_missing" if missing else ""
            sha = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if raw_text else ""
            source_url = str(page.get("fullurl") or source_url_for_title(normalized_title))
            return PageSnapshot(
                title=requested_title,
                normalized_title=normalized_title,
                pageid=page.get("pageid"),
                revision_id=revision_id,
                timestamp=timestamp,
                source_url=source_url,
                api_url=api_url,
                retrieved_at=retrieved_at,
                raw_text=raw_text,
                raw_text_sha256=sha,
                license_note=LICENSE_NOTE,
                fetch_status=status,
                error=error,
            )
        except (json.JSONDecodeError, TypeError, KeyError) as exc:
            return self._error_snapshot(requested_title, api_url, f"parse_error:{exc}")

    def _error_snapshot(self, title: str, api_url: str, error: str) -> PageSnapshot:
        return PageSnapshot(
            title=title,
            normalized_title=title,
            pageid=None,
            revision_id=None,
            timestamp="",
            source_url=source_url_for_title(title),
            api_url=api_url,
            retrieved_at=utc_now_iso(),
            raw_text="",
            raw_text_sha256="",
            license_note=LICENSE_NOTE,
            fetch_status="error",
            error=error,
        )

