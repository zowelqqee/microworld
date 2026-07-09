"""Backfill proposal-overlay definitions from fetched Wikipedia lead text.

This is intentionally proposal-only: it reads normalized local snapshot docs
and appends low-risk ``overlay_definition`` candidates to the pump dry-run
overlay. It does not fetch network data and does not touch accepted memory.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from worldpgt.knowledge_pump.safe_delta_merger import overlay_key
from worldpgt.relation_extraction_v2.sentence_splitter import (
    extract_full_body,
    split_paragraphs,
    split_sentences,
)

_COPULA_RE = re.compile(r"\b(?:is|are|was|were)\s+(?P<article>a|an|the|any)\s+", re.IGNORECASE)
_LASTED_APPROX_RE = re.compile(
    r"^In the history of (?P<context>[^,]+),\s+.+?\s+lasted approximately\s+(?P<period>from\s+.+?)(?:,\s+|\.|$)",
    re.IGNORECASE,
)
_INCLUDES_BOTH_RE = re.compile(
    r"^(?:Present-day\s+)?(?P<subject>.+?)\s+includes both\s+(?P<body>.+?)\.$",
    re.IGNORECASE,
)
_LEADING_ARTICLE_RE = re.compile(r"^(?:a|an|the)\s+", re.IGNORECASE)
_CLAUSE_STOP_RE = re.compile(
    r"\s+(?:which|who|that|while|although|but|including|covering|comprising|succeeding)\b",
    re.IGNORECASE,
)
_BAD_DEFINITION_HEADS = {
    "one",
    "same",
    "some",
    "many",
    "several",
    "former",
    "current",
    "latest",
}
_BAD_SUBJECT_MARKERS = (
    " election",
    " elections",
    " congress",
    " convention",
    " census",
    " primaries",
    " caucuses",
    " debates",
    " referendum",
)
_BAD_DEFINITION_PHRASES = (
    "meeting of the legislative branch",
    "election for the united states",
    "series of electoral contests",
    "series of primary elections",
    "series of debates",
    "involved in the various aspects",
    "political event to select",
)
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_RECENT_YEAR_RE = re.compile(r"\b202[3-9]\b")
_BAD_DEFINITION_ENDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}
_INCOMPLETE_COMMA_ENDS = {
    "existence",
    "production",
    "widespread",
}
_BACKFILL_SOURCE = "lead_definition_backfill"


def _norm(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return [item for item in data if isinstance(item, dict)]


def _doc_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback.replace("_", " ").strip()


def _lead_sentence(text: str) -> str | None:
    body = extract_full_body(text)
    paragraphs = split_paragraphs(body)
    if not paragraphs:
        return None
    sentences = split_sentences(paragraphs[0])
    return sentences[0] if sentences else None


def _lead_sentence_candidates(text: str) -> list[str]:
    body = extract_full_body(text)
    paragraphs = split_paragraphs(body)
    if not paragraphs:
        return []
    sentences = split_sentences(paragraphs[0])
    candidates: list[str] = []
    if len(sentences) >= 2:
        first = sentences[0]
        second = sentences[1]
        if first.count("(") > first.count(")") or second.lstrip().startswith(('"', "'")):
            candidates.append(f"{first} {second}")
    candidates.extend(sentences[:3])
    return candidates


def _truncate_definition(text: str) -> str:
    text = text.strip().rstrip(" .")
    comma = text.find(",")
    if comma >= 16:
        before_comma = text[:comma].strip().rstrip(" ,;:")
        last_word = before_comma.split()[-1].lower().rstrip(".,;:") if before_comma.split() else ""
        if last_word not in _INCOMPLETE_COMMA_ENDS:
            text = before_comma
    clause = _CLAUSE_STOP_RE.search(text)
    if clause and clause.start() >= 16:
        text = text[: clause.start()].strip().rstrip(" ,;:")
    if len(text) > 180:
        cut = text[:180].rsplit(" ", 1)[0].strip()
        text = cut.rstrip(" ,;:")
    return _LEADING_ARTICLE_RE.sub("", text).strip().rstrip(" ,;:")


def extract_lead_definition(title: str, sentence: str) -> str | None:
    """Return a short noun-phrase definition from a Wikipedia lead sentence."""

    if not sentence:
        return None
    title_norm = _norm(title)
    if any(marker in f" {title_norm}" for marker in _BAD_SUBJECT_MARKERS):
        return None
    if _RECENT_YEAR_RE.search(title):
        return None
    lasted = _LASTED_APPROX_RE.search(sentence)
    if lasted:
        definition = _truncate_definition(
            f"period in the history of {lasted.group('context')} lasting approximately {lasted.group('period')}"
        )
        if definition and len(definition.split()) >= 2:
            return definition
    includes = _INCLUDES_BOTH_RE.search(sentence)
    if includes and _norm(includes.group("subject")).endswith(title_norm):
        definition = (
            f"present-day change that includes both {includes.group('body')}"
        ).strip().rstrip(" .")
        if len(definition) > 180:
            definition = definition[:180].rsplit(" ", 1)[0].strip().rstrip(" ,;:")
        if 2 <= len(definition.split()) <= 30:
            return definition
    match = _COPULA_RE.search(sentence)
    if not match or match.start() > 220:
        return None
    definition = _truncate_definition(sentence[match.end() :])
    words = definition.split()
    if len(words) < 2:
        return None
    if words[0].lower() in _BAD_DEFINITION_HEADS:
        return None
    definition_norm = _norm(definition)
    if _CONTROL_CHAR_RE.search(definition):
        return None
    if _RECENT_YEAR_RE.search(definition):
        return None
    if any(phrase in definition_norm for phrase in _BAD_DEFINITION_PHRASES):
        return None
    if words[-1].lower().rstrip(".,;:") in _BAD_DEFINITION_ENDS:
        return None
    if len(words) > 30:
        return None
    if title_norm and definition_norm.startswith(title_norm):
        return None
    return definition


def make_definition_item(title: str, definition: str, sentence: str) -> dict[str, Any]:
    return {
        "overlay_type": "overlay_definition",
        "subject": title,
        "definition": definition,
        "predicate": "is_a",
        "source_page": title,
        "evidence_text": sentence,
        "evidence_span": sentence,
        "trust": "overlay_candidate",
        "risk": "low",
        "stability": "stable",
        "candidate_source": _BACKFILL_SOURCE,
        "extraction_pattern": "lead_definition_backfill_v1",
        "v2_pattern_id": "lead_definition_backfill_v1",
        "confidence_label": "lead_sentence",
        "pump_source_kind": "broad_seed_backfill",
    }


def build_definition_candidates(docs_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for doc_path in sorted(docs_dir.glob("*.md")):
        text = doc_path.read_text(encoding="utf-8")
        title = _doc_title(text, doc_path.stem)
        lead_candidates = _lead_sentence_candidates(text)
        if not lead_candidates:
            skipped.append({"title": title, "reason": "lead_sentence_missing"})
            continue
        definition = None
        sentence = ""
        for candidate in lead_candidates:
            definition = extract_lead_definition(title, candidate)
            if definition:
                sentence = candidate
                break
        if not definition:
            skipped.append({"title": title, "reason": "lead_definition_missing"})
            continue
        candidates.append(make_definition_item(title, definition, sentence))
    return candidates, {
        "docs_seen": len(list(docs_dir.glob("*.md"))),
        "candidate_count": len(candidates),
        "skipped_count": len(skipped),
        "skipped_sample": skipped[:30],
    }


def backfill_overlay_definitions(
    *,
    overlay_json: Path,
    docs_dir: Path,
    report_json: Path | None = None,
) -> dict[str, Any]:
    original_overlay_items = _read_json_list(overlay_json)
    overlay_items = [
        item
        for item in original_overlay_items
        if item.get("candidate_source") != _BACKFILL_SOURCE
    ]
    previous_removed_count = len(original_overlay_items) - len(overlay_items)
    candidates, candidate_summary = build_definition_candidates(docs_dir)
    seen = {overlay_key(item) for item in overlay_items}

    added: list[dict[str, Any]] = []
    duplicates: list[dict[str, Any]] = []
    for item in candidates:
        key = overlay_key(item)
        if key in seen:
            duplicates.append(item)
            continue
        seen.add(key)
        overlay_items.append(item)
        added.append(item)

    overlay_json.parent.mkdir(parents=True, exist_ok=True)
    overlay_json.write_text(json.dumps(overlay_items, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    report = {
        "overlay_json": str(overlay_json),
        "docs_dir": str(docs_dir),
        **candidate_summary,
        "previous_backfill_removed_count": previous_removed_count,
        "added_count": len(added),
        "duplicate_count": len(duplicates),
        "overlay_items_count": len(overlay_items),
        "added_sample": [
            {"subject": item.get("subject"), "definition": item.get("definition")}
            for item in added[:30]
        ],
    }
    if report_json is not None:
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Backfill pump dry-run overlay definitions from local normalized Wikipedia docs"
    )
    parser.add_argument("--overlay-json", required=True)
    parser.add_argument("--docs-dir", required=True)
    parser.add_argument("--report-json")
    args = parser.parse_args(argv)

    report = backfill_overlay_definitions(
        overlay_json=Path(args.overlay_json),
        docs_dir=Path(args.docs_dir),
        report_json=Path(args.report_json) if args.report_json else None,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
