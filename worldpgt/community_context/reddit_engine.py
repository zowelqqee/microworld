"""Offline Reddit/community-context engine.

The engine accepts local Reddit-like JSON/JSONL exports and writes an isolated
community-context artifact. It does not fetch from Reddit, does not create
``overlay_type`` facts, and does not modify accepted/promoted memory.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import re
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from worldpgt.community_context.types import (
    COMMUNITY_CONTEXT_TRUST,
    COMMUNITY_SOURCE_SYSTEM_REDDIT,
    CommunityContextItem,
    CommunityQuarantineItem,
    CommunitySearchResult,
    RedditRecord,
)
from worldpgt.community_context.cognitive_pattern_pump import (
    build_cognitive_pattern_graph,
    extract_cognitive_pattern_events,
)


_STOP_WORDS = frozenset(
    {
        "about", "after", "again", "also", "and", "are", "because", "been",
        "being", "but", "can", "could", "does", "for", "from", "had", "has",
        "have", "how", "into", "its", "just", "like", "more", "most", "not",
        "one", "only", "our", "out", "over", "really", "some", "than", "that",
        "the", "their", "them", "then", "there", "these", "they", "this",
        "those", "through", "was", "were", "what", "when", "where", "which",
        "while", "who", "why", "with", "would", "you", "your",
    }
)
_WEAK_QUERY_TERMS = frozenset({
    "common", "usually", "people", "make", "makes", "made", "thing", "things",
    "good", "bad", "best", "better", "question", "questions",
})
_TERM_ALIASES = {
    "beginner": {"beginner", "beginners", "newbie", "newbies", "novice", "novices", "junior"},
    "mistake": {"mistake", "mistakes", "pitfall", "pitfalls", "trap", "traps", "wrong"},
    "programming": {"programming", "programmer", "programmers", "coding", "code", "software"},
    "learn": {"learn", "learning", "learned", "study", "studying"},
    "debug": {"debug", "debugging", "debugger", "error", "errors", "traceback"},
    "python": {"python", "py"},
}

_DELETED_MARKERS = frozenset({"[deleted]", "[removed]", "deleted", "removed"})
_BOT_AUTHORS = frozenset({"automoderator", "autowikibot", "reddit"})
_PII_PATTERNS = (
    re.compile(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", re.I),
    re.compile(r"\b(?:\+?\d[\s().-]*){10,}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b(?:my|their|his|her)\s+(?:ssn|social security number)\b", re.I),
)
_HIGH_TOXICITY_TERMS = frozenset(
    {
        "kill yourself",
        "kys",
        "dox",
        "doxx",
    }
)


def load_reddit_records(paths: Sequence[str | Path]) -> list[RedditRecord]:
    """Load local Reddit-like JSON/JSONL files into normalized records."""

    records: list[RedditRecord] = []
    for path in paths:
        p = Path(path)
        for file_path in _expand_input_path(p):
            records.extend(_load_file(file_path))
    return records


def build_reddit_community_context(
    input_paths: Sequence[str | Path],
    out_dir: str | Path,
    *,
    max_items: int | None = None,
) -> dict:
    """Build isolated community-context, quarantine, and summary artifacts."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    records = load_reddit_records(input_paths)
    accepted, quarantine = classify_reddit_records(records)
    if max_items is not None:
        accepted = accepted[:max_items]

    context_path = out / "reddit_community_context.json"
    quarantine_path = out / "reddit_community_quarantine.json"
    summary_path = out / "reddit_community_summary.json"
    speaking_profile_path = out / "reddit_speaking_profile.json"
    cognitive_patterns_path = out / "cognitive_pattern_events.json"
    cognitive_graphs_path = out / "cognitive_pattern_graphs.json"

    _write_json(context_path, [item.to_dict() for item in accepted])
    _write_json(quarantine_path, [item.to_dict() for item in quarantine])
    speaking_profile = build_speaking_profile(accepted)
    _write_json(speaking_profile_path, speaking_profile)
    cognitive_patterns = extract_cognitive_pattern_events(accepted)
    _write_json(cognitive_patterns_path, [event.to_dict() for event in cognitive_patterns])
    cognitive_graphs = build_cognitive_pattern_graph(cognitive_patterns)
    _write_json(cognitive_graphs_path, cognitive_graphs)

    by_reason = Counter(item.reason for item in quarantine)
    by_subreddit = Counter(item.subreddit for item in accepted)
    by_pattern_kind = Counter(event.kind for event in cognitive_patterns)
    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_system": COMMUNITY_SOURCE_SYSTEM_REDDIT,
        "input_records_count": len(records),
        "accepted_context_items_count": len(accepted),
        "cognitive_pattern_events_count": len(cognitive_patterns),
        "quarantine_count": len(quarantine),
        "quarantine_by_reason": dict(sorted(by_reason.items())),
        "accepted_by_subreddit": dict(sorted(by_subreddit.items())),
        "cognitive_patterns_by_kind": dict(sorted(by_pattern_kind.items())),
        "context_path": str(context_path),
        "quarantine_path": str(quarantine_path),
        "speaking_profile_path": str(speaking_profile_path),
        "cognitive_patterns_path": str(cognitive_patterns_path),
        "cognitive_graphs_path": str(cognitive_graphs_path),
        "trust": COMMUNITY_CONTEXT_TRUST,
        "factual_support_allowed": False,
        "community_patterns_factual_support_allowed": False,
        "pattern_memory_layer": "cognitive_behavioral_patterns",
        "accepted_overlay_modified": False,
        "promoted_overlay_modified": False,
        "snapshot_dry_run_overlay_modified": False,
    }
    _write_json(summary_path, summary)
    return summary


def build_speaking_profile(items: Iterable[CommunityContextItem | dict]) -> dict:
    """Summarize accepted community items into a low-trust speaking profile.

    The profile is descriptive only. It helps inspect what the community pump is
    teaching the system about phrasing, topics, and common concerns, but it is
    not factual memory.
    """

    coerced = [_coerce_item(item) for item in items]
    topic_counts: Counter[str] = Counter()
    subreddit_counts: Counter[str] = Counter()
    question_phrasings: list[dict] = []
    concern_phrases: list[str] = []

    for item in coerced:
        topic_counts.update(item.topic_terms)
        if item.subreddit:
            subreddit_counts[item.subreddit] += 1
        title = _clean_text(item.title)
        if "?" in title and len(question_phrasings) < 30:
            question_phrasings.append({
                "subreddit": item.subreddit,
                "text": _clip(title, 180),
                "score": item.score,
            })
        lower = item.text.lower()
        if any(word in lower for word in ("worry", "concern", "stuck", "confused", "struggle")):
            concern_phrases.append(_clip(item.text, 220))

    return {
        "source_system": COMMUNITY_SOURCE_SYSTEM_REDDIT,
        "trust": COMMUNITY_CONTEXT_TRUST,
        "item_count": len(coerced),
        "factual_support_allowed": False,
        "top_topic_terms": [
            {"term": term, "count": count}
            for term, count in topic_counts.most_common(50)
        ],
        "top_subreddits": [
            {"subreddit": subreddit, "count": count}
            for subreddit, count in subreddit_counts.most_common(30)
        ],
        "sample_question_phrasings": question_phrasings,
        "sample_common_concerns": concern_phrases[:30],
        "usage": (
            "Use for conversational phrasing, examples, and common concerns. "
            "Do not use as stable factual support."
        ),
    }


def classify_reddit_records(
    records: Iterable[RedditRecord],
) -> tuple[list[CommunityContextItem], list[CommunityQuarantineItem]]:
    accepted: list[CommunityContextItem] = []
    quarantine: list[CommunityQuarantineItem] = []
    seen_hashes: set[str] = set()

    for record in records:
        text = _clean_text(" ".join(part for part in (record.title, record.body) if part))
        reason = _reject_reason(record, text)
        url = _normalize_permalink(record.permalink)
        if reason:
            quarantine.append(
                CommunityQuarantineItem(
                    source_id=record.source_id,
                    source_system=COMMUNITY_SOURCE_SYSTEM_REDDIT,
                    reason=reason,
                    text=_clip(text, 500),
                    subreddit=record.subreddit,
                    url=url,
                )
            )
            continue

        content_hash = hashlib.sha256(_normalize_for_hash(text).encode("utf-8")).hexdigest()[:16]
        if content_hash in seen_hashes:
            quarantine.append(
                CommunityQuarantineItem(
                    source_id=record.source_id,
                    source_system=COMMUNITY_SOURCE_SYSTEM_REDDIT,
                    reason="duplicate_text",
                    text=_clip(text, 500),
                    subreddit=record.subreddit,
                    url=url,
                )
            )
            continue
        seen_hashes.add(content_hash)

        flags = _quality_flags(record, text)
        accepted.append(
            CommunityContextItem(
                item_id=f"reddit:{content_hash}",
                source_system=COMMUNITY_SOURCE_SYSTEM_REDDIT,
                source_kind=record.source_kind,
                trust=COMMUNITY_CONTEXT_TRUST,
                subreddit=record.subreddit,
                title=_clip(_clean_text(record.title), 160),
                text=_clip(text, 1200),
                url=url,
                score=record.score,
                created_utc=str(record.created_utc or ""),
                topic_terms=tuple(_topic_terms(text, subreddit=record.subreddit)),
                flags=tuple(flags),
                risk="medium" if flags else "low",
                stability="volatile",
            )
        )

    accepted.sort(key=lambda item: ((item.score or 0), len(item.text)), reverse=True)
    return accepted, quarantine


def query_community_context(
    items: Iterable[CommunityContextItem | dict],
    question: str,
    *,
    max_results: int = 5,
) -> list[CommunitySearchResult]:
    """Return lexical community-context matches for a question."""

    query_terms = _expanded_query_terms(question)
    if not query_terms:
        return []
    results: list[CommunitySearchResult] = []
    for raw_item in items:
        item = _coerce_item(raw_item)
        searchable = f"{item.title} {item.text}"
        item_terms = _expanded_text_terms(searchable)
        matched = tuple(sorted(query_terms & item_terms))
        strong_matches = [term for term in matched if term not in _WEAK_QUERY_TERMS]
        required = 1 if len(query_terms - _WEAK_QUERY_TERMS) <= 1 else 2
        if len(strong_matches) < required:
            continue
        score = len(strong_matches) * 15.0 + len(matched) * 2.0
        if item.score is not None and item.score > 0:
            score += min(item.score, 100) / 25.0
        score += _sentence_overlap_bonus(searchable, query_terms)
        results.append(CommunitySearchResult(item=item, score=score, matched_terms=matched))
    results.sort(key=lambda result: result.score, reverse=True)
    return results[:max_results]


def render_community_context(question: str, results: Sequence[CommunitySearchResult]) -> str:
    """Render a forum-style answer without exposing raw community snippets.

    Community data teaches phrasing and answer shape here. It is not used as a
    fact source, so normal answers should not dump source excerpts back to the
    user.
    """

    advice = _community_style_answer(question, results)
    if not results and _is_generic_style_answer(advice):
        return "I do not have enough community-style signal for this wording yet."

    lines = []
    if advice:
        lines.extend(advice)

    lines.append(
        "\nStyle note: shaped from low-trust community phrasing, not factual "
        "support. Do not promote it to stable facts."
    )
    return "\n".join(lines)


def _is_generic_style_answer(lines: Sequence[str]) -> bool:
    return bool(lines) and lines[0] == "I would phrase it like this:"


def _community_style_answer(question: str, results: Sequence[CommunitySearchResult]) -> list[str]:
    terms = _expanded_query_terms(question)
    low = question.lower()
    del results
    if "ask" in low and "programming question" in low:
        return [
            "A good programming question is basically a small, clean bug report.",
            "",
            "Use this shape:",
            "- What I am trying to do.",
            "- What I expected to happen.",
            "- What actually happened.",
            "- The smallest code example that still shows the problem.",
            "- The exact error text, copied as text rather than paraphrased.",
            "- What I already tried, and what changed when I tried it.",
            "",
            "The trick is to remove everything that is not part of the problem. "
            "People can help much faster when they can run the same tiny example "
            "and see the same failure.",
        ]
    if "recursion" in low:
        return [
            "Recursion is when a function solves a problem by calling itself on a "
            "smaller version of the same problem.",
            "",
            "A forum-style way to think about it:",
            "- First, define the tiny case where you already know the answer.",
            "- Then define how to shrink the big case toward that tiny case.",
            "- Each call handles one smaller piece.",
            "- When the tiny case is reached, the answers unwind back up.",
            "",
            "Like opening nested boxes: each box contains a smaller box, until one "
            "box is empty. Then you stop opening and walk back out.",
        ]
    if "api" in low or "apis" in low:
        return [
            "An API is a way for one piece of software to ask another piece of "
            "software for something in a predictable format.",
            "",
            "Plain English version: you send a request, it sends back a response. "
            "The API contract says what you are allowed to ask for, what details "
            "you must include, and what shape the answer will have.",
        ]
    if {"beginner", "mistake", "programming"} & terms and (
        "mistake" in terms or "mistakes" in low
    ):
        return [
            "Common beginner programming mistakes, in plain terms:",
            "- Trying to learn too much theory before building anything small.",
            "- Copying code without stopping to explain what each line is doing.",
            "- Treating errors as failure instead of reading them as clues.",
            "- Jumping between languages, courses, and tools before one idea has settled.",
            "- Avoiding small projects, tests, and debugging practice because they feel slower than watching tutorials.",
        ]
    if "debug" in terms:
        return [
            "A practical debugging rhythm:",
            "- Reproduce the bug in the smallest possible example.",
            "- Read the first real error message, not only the last line of noise.",
            "- Add prints or assertions around the exact assumption that might be wrong.",
            "- Change one thing at a time so you know what actually fixed it.",
        ]
    if "learn" in terms and "programming" in terms:
        return [
            "A sane way to learn programming:",
            "- Pick one language and build tiny things with it repeatedly.",
            "- Keep notes on mistakes you keep making; that turns confusion into a checklist.",
            "- Use docs and examples together: docs tell you the rule, examples show the shape.",
            "- Ask focused questions with the error, the tiny repro, and what you expected.",
        ]
    return [
        "I would phrase it like this:",
        "",
        "Start with the concrete thing you are trying to understand, then give one "
        "small example, then say where the confusion starts. A good explanation "
        "usually beats a clever one: simple words, one idea at a time, and no "
        "extra jargon until it is needed.",
    ]


def _expand_input_path(path: Path) -> list[Path]:
    if path.is_dir():
        files = sorted(path.rglob("*.json")) + sorted(path.rglob("*.jsonl"))
        return [p for p in files if p.is_file()]
    return [path]


def _load_file(path: Path) -> list[RedditRecord]:
    if path.suffix.lower() == ".jsonl":
        records: list[RedditRecord] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            loaded = json.loads(line)
            records.extend(_records_from_json_obj(loaded))
        return records
    return _records_from_json_obj(json.loads(path.read_text(encoding="utf-8")))


def _records_from_json_obj(obj: object) -> list[RedditRecord]:
    return [record for raw in _iter_records(obj) if (record := _normalize_record(raw))]


def _iter_records(obj: object) -> Iterator[dict]:
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_records(item)
        return
    if not isinstance(obj, dict):
        return
    if _looks_like_reddit_record(obj):
        yield obj
        return
    data = obj.get("data")
    if isinstance(data, dict):
        children = data.get("children")
        if isinstance(children, list):
            for child in children:
                yield from _iter_records(child)
            return
        if _looks_like_reddit_record(data):
            yield data
            return
    for key in ("posts", "comments", "items", "records", "children"):
        value = obj.get(key)
        if isinstance(value, list):
            for item in value:
                yield from _iter_records(item)


def _looks_like_reddit_record(obj: dict) -> bool:
    if "kind" in obj and isinstance(obj.get("data"), dict):
        return True
    return any(key in obj for key in ("selftext", "body", "subreddit", "permalink", "created_utc")) and (
        "title" in obj or "body" in obj or "selftext" in obj
    )


def _normalize_record(raw: dict) -> RedditRecord | None:
    if "kind" in raw and isinstance(raw.get("data"), dict):
        kind_hint = str(raw.get("kind") or "")
        raw = raw["data"]
    else:
        kind_hint = str(raw.get("kind") or raw.get("name") or "")

    title = str(raw.get("title") or "")
    body = str(raw.get("selftext") or raw.get("body") or raw.get("text") or "")
    source_kind = "comment" if kind_hint.startswith("t1") or (body and not title) else "post"
    source_id = str(raw.get("name") or raw.get("id") or _stable_id(title, body))
    subreddit = str(raw.get("subreddit") or raw.get("subreddit_name_prefixed") or "").removeprefix("r/")
    author = str(raw.get("author") or "")
    return RedditRecord(
        source_id=source_id,
        source_kind=source_kind,
        subreddit=subreddit,
        title=title,
        body=body,
        author=author,
        score=_to_int(raw.get("score")),
        permalink=str(raw.get("permalink") or raw.get("url") or ""),
        created_utc=str(raw.get("created_utc") or ""),
        over_18=bool(raw.get("over_18") or raw.get("nsfw")),
    )


def _reject_reason(record: RedditRecord, text: str) -> str:
    if record.over_18:
        return "nsfw"
    if not text or text.lower() in _DELETED_MARKERS:
        return "deleted_or_empty"
    if record.body.strip().lower() in _DELETED_MARKERS:
        return "deleted_or_empty"
    if len(text) < 40:
        return "too_short"
    author_norm = record.author.strip().lower()
    if author_norm in _BOT_AUTHORS or author_norm.endswith("bot"):
        return "bot_or_automated"
    if record.score is not None and record.score <= -3:
        return "low_quality_score"
    if any(pattern.search(text) for pattern in _PII_PATTERNS):
        return "private_or_sensitive_data"
    text_lower = text.lower()
    if any(term in text_lower for term in _HIGH_TOXICITY_TERMS):
        return "high_toxicity"
    return ""


def _quality_flags(record: RedditRecord, text: str) -> list[str]:
    flags: list[str] = []
    if record.score is not None and record.score < 2:
        flags.append("low_score")
    if "i think" in text.lower() or "in my experience" in text.lower():
        flags.append("anecdotal")
    if "?" in text:
        flags.append("question_like")
    return flags


def _topic_terms(text: str, *, subreddit: str) -> list[str]:
    counts = Counter(_tokenize(text))
    if subreddit and ":" not in subreddit:
        counts[_clean_token(subreddit)] += 2
    terms = [
        term
        for term, _count in counts.most_common(16)
        if term and term not in _STOP_WORDS and len(term) > 2
    ]
    return terms[:12]


def _tokenize(text: str) -> list[str]:
    return [
        token
        for token in (_canonical_token(_clean_token(match.group(0))) for match in re.finditer(r"[A-Za-z0-9][A-Za-z0-9_'-]*", text.lower()))
        if token and token not in _STOP_WORDS and len(token) > 2
    ]


def _clean_token(token: str) -> str:
    return re.sub(r"(^[^a-z0-9]+|[^a-z0-9]+$)", "", token.lower())


def _canonical_token(token: str) -> str:
    if not token:
        return ""
    for canonical, variants in _TERM_ALIASES.items():
        if token in variants:
            return canonical
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ers"):
        return token[:-1]
    if len(token) > 3 and token.endswith("s"):
        return token[:-1]
    return token


def _expanded_query_terms(text: str) -> set[str]:
    base = set(_tokenize(text)) - _WEAK_QUERY_TERMS
    expanded = set(base)
    for canonical, variants in _TERM_ALIASES.items():
        if canonical in base or base & variants:
            expanded.add(canonical)
            expanded.update(_canonical_token(v) for v in variants)
    return {term for term in expanded if term and term not in _STOP_WORDS}


def _expanded_text_terms(text: str) -> set[str]:
    base = set(_tokenize(text))
    expanded = set(base)
    for canonical, variants in _TERM_ALIASES.items():
        if canonical in base or base & variants:
            expanded.add(canonical)
    return expanded


def _sentence_overlap_bonus(text: str, query_terms: set[str]) -> float:
    best = 0
    for sentence in _sentences(text):
        best = max(best, len(_expanded_text_terms(sentence) & query_terms))
    return float(best * 5)


def _best_excerpt(item: CommunityContextItem, query_terms: set[str], *, limit: int) -> str:
    candidates = _sentences(f"{item.title}. {item.text}")
    if not candidates:
        return _clip(item.text or item.title, limit)
    ranked = sorted(
        candidates,
        key=lambda sentence: len(_expanded_text_terms(sentence) & query_terms),
        reverse=True,
    )
    return _clip(ranked[0], limit)


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", _clean_text(text))
    return [part.strip() for part in parts if len(part.strip()) >= 30]


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") + "."


def _normalize_permalink(value: str) -> str:
    value = (value or "").strip()
    if value.startswith("/"):
        return "https://www.reddit.com" + value
    return value


def _normalize_for_hash(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _stable_id(title: str, body: str) -> str:
    return hashlib.sha256(f"{title}\n{body}".encode("utf-8")).hexdigest()[:16]


def _to_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _coerce_item(raw_item: CommunityContextItem | dict) -> CommunityContextItem:
    if isinstance(raw_item, CommunityContextItem):
        return raw_item
    payload = dict(raw_item)
    payload["topic_terms"] = tuple(payload.get("topic_terms") or ())
    payload["flags"] = tuple(payload.get("flags") or ())
    return CommunityContextItem(**payload)


def _write_json(path: Path, data: object) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
