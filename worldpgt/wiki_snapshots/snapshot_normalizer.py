"""Normalize raw Wikipedia snapshots into local wiki-like documents."""

from __future__ import annotations

import re
import urllib.parse
from pathlib import Path

from worldpgt.wiki_snapshots.types import PageSnapshot

_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_REF_RE = re.compile(r"<ref[^>]*>.*?</ref>|<ref[^>]*/>", re.IGNORECASE | re.DOTALL)
_HTML_RE = re.compile(r"<[^>]+>")


def safe_title_filename(title: str) -> str:
    normalized = "_".join(title.strip().split())
    encoded = urllib.parse.quote(normalized, safe="._-()")
    return encoded or "untitled"


def clean_article_text(raw_text: str) -> str:
    text = raw_text or ""
    text = _REF_RE.sub("", text)
    text = _TEMPLATE_RE.sub("", text)
    text = _HTML_RE.sub("", text)
    text = text.replace("'''", "").replace("''", "")
    cleaned_lines: list[str] = []
    previous_blank = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith(("[[Category:", "{{", "|", "}}")):
            continue
        if not line:
            if not previous_blank and cleaned_lines:
                cleaned_lines.append("")
            previous_blank = True
            continue
        cleaned_lines.append(line)
        previous_blank = False
    while cleaned_lines and cleaned_lines[-1] == "":
        cleaned_lines.pop()
    return "\n".join(cleaned_lines)


def build_normalized_doc(snapshot: PageSnapshot) -> str:
    revision_id = "" if snapshot.revision_id is None else str(snapshot.revision_id)
    header = [
        f"# {snapshot.normalized_title or snapshot.title}",
        "",
        f"Source: {snapshot.source_url}",
        f"Retrieved at: {snapshot.retrieved_at}",
        f"Revision ID: {revision_id}",
        f"Raw text SHA256: {snapshot.raw_text_sha256}",
        "Status: LOCAL_WIKIPEDIA_SNAPSHOT",
        "Safe for accepted memory: false",
        "Requires ingestion/quarantine/promotion/regression: true",
        "",
    ]
    body = clean_article_text(snapshot.raw_text)
    return "\n".join(header) + body + ("\n" if body else "")


def write_normalized_doc(snapshot: PageSnapshot, output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{safe_title_filename(snapshot.normalized_title or snapshot.title)}.md"
    path.write_text(build_normalized_doc(snapshot), encoding="utf-8")
    return path

