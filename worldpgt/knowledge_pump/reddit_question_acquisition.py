"""Live, proposal-only collection of AI question phrasing from Reddit.

This module is intentionally an acquisition lane, not factual learning.  It
collects public post titles that look like questions and writes reviewable
candidates.  No collected text can alter accepted memory or the runtime
relation-input graph.
"""

from __future__ import annotations

from datetime import datetime, timezone
import base64
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from worldpgt.reasoning.relation_input_graph import default_relation_input_graph, question_frame_tokens


_QUESTION_PREFIX = re.compile(r"^(?:what|which|who|whom|where|when|why|how|can|could|does|do|is|are)\b", re.IGNORECASE)
_USER_AGENT = "MicroWorldRelationInputPump/1.0 (proposal-only research collector)"
_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
_OAUTH_BASE_URL = "https://oauth.reddit.com"


def question_candidates_from_listing(payload: dict[str, Any], subreddit: str) -> list[dict[str, Any]]:
    """Turn one Reddit listing response into deduplicable review candidates."""

    graph = default_relation_input_graph()
    candidates: list[dict[str, Any]] = []
    for child in (payload.get("data") or {}).get("children") or []:
        data = child.get("data") if isinstance(child, dict) else None
        if not isinstance(data, dict):
            continue
        title = " ".join(str(data.get("title") or "").split())
        if not title or not ("?" in title or _QUESTION_PREFIX.match(title)):
            continue
        candidates.append({
            "candidate_id": "reddit:" + str(data.get("id") or ""),
            "source_system": "reddit",
            "subreddit": subreddit,
            "url": "https://www.reddit.com" + str(data.get("permalink") or ""),
            "question": title,
            "frame_tokens": list(question_frame_tokens(title)),
            "existing_graph_predicate": graph.resolve(title),
            "score": data.get("score"),
            "created_utc": data.get("created_utc"),
            "proposal_only": True,
            "factual_support_allowed": False,
            "review_status": "unreviewed",
        })
    return candidates


def official_reddit_fetcher_from_env() -> Callable[[str], dict[str, Any]]:
    """Create an application-only OAuth reader from explicit environment vars.

    ``REDDIT_CLIENT_ID`` and ``REDDIT_CLIENT_SECRET`` are intentionally read
    only at invocation time and are never placed into artifacts, logs, or
    command-line arguments.
    """

    client_id = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    user_agent = os.environ.get("REDDIT_USER_AGENT", _USER_AGENT)
    if not client_id or not client_secret:
        raise RuntimeError("Reddit OAuth requires REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
    credentials = base64.b64encode(f"{client_id}:{client_secret}".encode("utf-8")).decode("ascii")
    token_request = Request(
        _TOKEN_URL,
        data=urlencode({"grant_type": "client_credentials"}).encode("ascii"),
        headers={
            "Authorization": "Basic " + credentials,
            "User-Agent": user_agent,
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    with urlopen(token_request, timeout=20) as response:  # nosec B310: fixed official HTTPS origin
        token_payload = json.loads(response.read().decode("utf-8"))
    token = token_payload.get("access_token")
    if not isinstance(token, str) or not token:
        raise RuntimeError("Reddit OAuth token response did not include access_token")

    def fetch(subreddit: str) -> dict[str, Any]:
        url = f"{_OAUTH_BASE_URL}/r/{subreddit}/new?limit=100&raw_json=1"
        request = Request(url, headers={
            "Authorization": "bearer " + token,
            "User-Agent": user_agent,
            "Accept": "application/json",
        })
        with urlopen(request, timeout=20) as response:  # nosec B310: fixed official HTTPS origin
            return json.loads(response.read().decode("utf-8"))

    return fetch


def run_live_question_acquisition(
    output_dir: str | Path,
    *,
    duration_seconds: float,
    subreddits: Iterable[str],
    poll_seconds: float = 60.0,
    fetch: Callable[[str], dict[str, Any]] | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Collect a bounded live sample and persist candidates plus audit report."""

    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    fetch = fetch or official_reddit_fetcher_from_env()
    targets = tuple(dict.fromkeys(subreddit for subreddit in subreddits if subreddit))
    deadline = clock() + duration_seconds
    candidates: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, str]] = []
    requests = 0
    while clock() < deadline:
        for subreddit in targets:
            if clock() >= deadline:
                break
            try:
                payload = fetch(subreddit)
                requests += 1
                for candidate in question_candidates_from_listing(payload, subreddit):
                    candidates.setdefault(candidate["candidate_id"], candidate)
            except Exception as exc:  # keep an auditable partial collection
                errors.append({"subreddit": subreddit, "error": repr(exc)})
            sleep(min(2.0, max(0.0, deadline - clock())))
        remaining = deadline - clock()
        if remaining > 0:
            sleep(min(poll_seconds, remaining))
    ordered = [candidates[key] for key in sorted(candidates)]
    (root / "reddit_ai_question_candidates.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in ordered), encoding="utf-8"
    )
    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration_seconds,
        "subreddits": list(targets),
        "requests": requests,
        "candidate_count": len(ordered),
        "known_graph_matches": sum(item["existing_graph_predicate"] is not None for item in ordered),
        "errors": errors,
        "proposal_only": True,
        "accepted_memory_modified": False,
        "runtime_graph_modified": False,
        "factual_support_allowed": False,
    }
    (root / "reddit_ai_question_acquisition_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
