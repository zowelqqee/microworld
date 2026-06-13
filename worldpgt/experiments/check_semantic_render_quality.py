"""Flag awkward continuations among Microworld *continued* rows (semantic quality).

This is a measurement tool, not a gate. It scans rows whose decision is
``continue`` and flags continuations that look awkward: too short, repeating an
action/noun from the prompt, mismatching the connector, showing story-drift
markers (dialogue, unrelated family/story words, newlines), or empty.

It is intentionally lenient about clean short continuations: a short phrase is
only flagged when it adds no new content word at all.

Usage:
    python3 -m worldpgt.experiments.check_semantic_render_quality \
        --input worldpgt/experiments/microworld_continuation_v1_2_outputs.csv \
        --output worldpgt/experiments/microworld_continuation_v1_2_semantic_quality.json
"""

from __future__ import annotations

import argparse
import csv
import json
import re

from worldpgt.continuation.realization import (
    ENDING_AND,
    ENDING_AS,
    ENDING_INFINITIVE,
    classify_prompt_ending,
)

_TOKEN_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = {
    "the", "a", "an", "to", "and", "as", "it", "its", "his", "her", "he", "she",
    "they", "them", "their", "was", "were", "is", "are", "be", "of", "in", "on",
    "at", "with", "for", "by", "from", "into", "below", "above", "near", "over",
    "under", "down", "up", "out", "this", "that", "then", "only", "so", "but",
    "or", "if", "after", "before", "until", "when", "while", "him", "had", "has",
    "have", "about", "back", "another", "more", "other", "same",
}
_SOFTENERS = {"another", "more", "other", "same", "fresh", "second"}

_DRIFT_WORDS = {
    "said", "asked", "told", "replied", "shouted", "whispered",
    "mother", "father", "boyfriend", "girlfriend", "pregnancy",
    "wife", "husband", "daughter", "son",
}
_DRIFT_CHARS = ('"', "'", "“", "”", "‘", "’", "\n")

_PROMPT_TAIL_BAD_PATTERNS = {
    "could_and_searched": re.compile(r"\bcould\s+and\s+searched\b", re.IGNORECASE),
    "before_and_hit": re.compile(r"\bbefore\s+and\s+hit\b", re.IGNORECASE),
    "motioned_for_and_completed": re.compile(r"\bmotioned\s+for\s+and\s+completed\b", re.IGNORECASE),
    "while_tourists_and_swam": re.compile(r"\bwhile\s+tourists\s+and\s+swam\b", re.IGNORECASE),
    "turned_toward_and_carried": re.compile(r"\bturned\s+toward\s+and\s+carried\b", re.IGNORECASE),
    "as_the_hook_the_operator": re.compile(r"\bas\s+the\s+hook\s+the\s+operator\b", re.IGNORECASE),
    "made_everyone_and_brought": re.compile(r"\bmade\s+everyone\s+and\s+brought\b", re.IGNORECASE),
    "after_and_filled": re.compile(r"\bafter\s+and\s+filled\b", re.IGNORECASE),
    "would_get_louder_after_and": re.compile(r"\bwould\s+get\s+louder\s+after\s+and\b", re.IGNORECASE),
    "before_player_swung_comma_no_and": re.compile(
        r"\bbefore\s+the\s+player\s+swung,\s+he\s+steadied\b",
        re.IGNORECASE,
    ),
}


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _phrase_part(prompt: str, continuation: str) -> str:
    base = prompt.rstrip()
    if continuation.lower().startswith(base.lower()):
        return continuation[len(base):].strip()
    return continuation.strip()


def _repeated_content(prompt: str, phrase_part: str, term: str) -> list[str]:
    recent = _tokens(prompt)[-6:]
    recent_content = {t for t in recent if t not in _STOPWORDS and t != term}
    phrase_tokens = _tokens(phrase_part)
    repeated = []
    for idx, token in enumerate(phrase_tokens):
        if token in _STOPWORDS or token == term:
            continue
        if token in recent_content:
            prev = phrase_tokens[idx - 1] if idx > 0 else ""
            if prev in _SOFTENERS:
                continue
            repeated.append(token)
    return repeated


def _connector_mismatch(prompt: str, phrase_part: str) -> bool:
    ending = classify_prompt_ending(prompt)
    low = phrase_part.lower()
    if ending == ENDING_INFINITIVE:
        # After "to", a bare past-tense verb start reads wrong (e.g. "to swam").
        return bool(re.match(r"^(swam|closed|lifted|brought|hardened)\b", low))
    if ending in {ENDING_AND, ENDING_AS} and low.startswith(("and ", "as ")):
        return True
    return False


def _flag_row(row: dict) -> list[str]:
    prompt = row.get("prompt", "")
    continuation = row.get("continuation", "")
    term = (row.get("ambiguous_term", "") or "").lower()

    if not continuation.strip():
        return ["empty_continuation"]

    phrase_part = _phrase_part(prompt, continuation)
    content_tokens = [t for t in _tokens(phrase_part) if t not in _STOPWORDS]

    flags: list[str] = []
    if len(content_tokens) < 1:
        flags.append("too_short")

    repeated = _repeated_content(prompt, phrase_part, term)
    if repeated:
        flags.append("repeated_from_prompt:" + ",".join(sorted(set(repeated))))

    if _connector_mismatch(prompt, phrase_part):
        flags.append("connector_mismatch")

    if any(ch in phrase_part for ch in _DRIFT_CHARS):
        flags.append("story_drift:dialogue_or_newline")
    drift = sorted({t for t in _tokens(phrase_part) if t in _DRIFT_WORDS})
    if drift:
        flags.append("story_drift:" + ",".join(drift))

    prompt_tail = [
        name for name, pattern in _PROMPT_TAIL_BAD_PATTERNS.items() if pattern.search(continuation)
    ]
    if prompt_tail:
        flags.append("prompt_tail:" + ",".join(sorted(prompt_tail)))

    return flags


def check_rows(rows: list[dict]) -> dict:
    flagged_rows = []
    total_continued = 0
    for row in rows:
        if row.get("decision") != "continue":
            continue
        total_continued += 1
        flags = _flag_row(row)
        if flags:
            flagged_rows.append(
                {
                    "id": row.get("id", ""),
                    "prompt": row.get("prompt", ""),
                    "selected_sense": row.get("selected_sense", ""),
                    "flags": flags,
                    "continuation": row.get("continuation", ""),
                }
            )
    flagged_count = len(flagged_rows)
    return {
        "total_continued": total_continued,
        "flagged_count": flagged_count,
        "flagged_rate": round(flagged_count / total_continued, 4) if total_continued else 0.0,
        "flagged_rows": flagged_rows,
    }


def check_file(input_path: str) -> dict:
    with open(input_path, "r", newline="", encoding="utf-8") as handle:
        return check_rows(list(csv.DictReader(handle)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check semantic continuation quality.")
    parser.add_argument("--input", required=True, help="Input controlled continuation output CSV")
    parser.add_argument("--output", required=True, help="Output JSON quality report")
    args = parser.parse_args()

    summary = check_file(args.input)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
