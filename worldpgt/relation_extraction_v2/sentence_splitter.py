"""Deterministic sentence splitter for Wikipedia snapshot text.

Handles common Wikipedia abbreviations (Inc., Corp., L.P., U.S., e.g., etc.)
without external NLP libraries. Good enough for Wikipedia-style prose paragraphs.
No ML, no network.
"""

from __future__ import annotations

import re

# Abbreviations that contain a period but do NOT end a sentence.
_ABBREV = re.compile(
    r"\b("
    r"Inc|Corp|Ltd|L\.P|L\.L\.C|LLC|LLP|PLC|S\.A|GmbH|B\.V"
    r"|U\.S|U\.K|U\.N|E\.U|U\.A\.E|Ph\.D|Dr|Mr|Mrs|Ms|St|Ave|Blvd|Dept|Govt|Est"
    r"|approx|vs|etc|e\.g|i\.e|cf|ca|al|vol|no|fig|ed|rev|eq"
    r"|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
    r")\.",
    re.IGNORECASE,
)

# Pattern for sentence boundary: ./?/! followed by whitespace + capital letter.
_BOUNDARY_RE = re.compile(r"([.!?])\s+(?=[A-Z\"])")

# Minimum sentence length (characters) — ignore tiny fragments.
_MIN_SENTENCE_LEN = 15


def _placeholder_abbrevs(text: str) -> tuple[str, dict[str, str]]:
    """Replace each known abbreviation (period included) with an opaque
    placeholder so the boundary splitter can't break on its internal period.
    The whole match is swapped out and restored verbatim -- see ``_restore``.
    """

    placeholders: dict[str, str] = {}
    idx = [0]

    def replace(m: re.Match) -> str:
        key = f"\x00AB{idx[0]}\x00"
        idx[0] += 1
        placeholders[key] = m.group(0)
        return key

    replaced = _ABBREV.sub(replace, text)
    return replaced, placeholders


def _restore(text: str, placeholders: dict[str, str]) -> str:
    # Replace the FULL placeholder key with the FULL original match. The old
    # implementation sliced both key and value (``key[:-1]`` / ``val[:-1]``),
    # which left the trailing NUL of the key in the text and duplicated the
    # abbreviation -- turning "U.S." into the infamous "U.SU.S\x00" garbage
    # that then corrupted every entity/relation extracted from that sentence.
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return text


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences. Handles common abbreviations."""

    if not text:
        return []

    text, placeholders = _placeholder_abbrevs(text)
    parts: list[str] = []
    last = 0
    for m in _BOUNDARY_RE.finditer(text):
        end = m.start() + 1  # include the punctuation
        chunk = text[last:end].strip()
        if chunk:
            parts.append(chunk)
        last = m.end() - len(m.group(0)) + 1  # start after the punctuation char

    tail = text[last:].strip()
    if tail:
        parts.append(tail)

    result: list[str] = []
    for p in parts:
        restored = _restore(p, placeholders)
        if len(restored) >= _MIN_SENTENCE_LEN:
            result.append(restored)
    return result


def split_paragraphs(text: str) -> list[str]:
    """Split body text into non-empty paragraphs (separated by blank lines)."""

    paras: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Skip markdown section headers.
        if stripped.startswith("#"):
            if current:
                paras.append(" ".join(current))
                current = []
            continue
        if not stripped:
            if current:
                paras.append(" ".join(current))
                current = []
        else:
            current.append(stripped)
    if current:
        paras.append(" ".join(current))
    return [p for p in paras if p]


def extract_full_body(raw_doc_text: str) -> str:
    """Extract the full body of a normalized snapshot doc, including intro text.

    The existing ``split_snapshot_doc`` helper sets the body start at the first
    blank line whose successor is not a metadata key-value pair. For some pages
    this skips the intro paragraphs that appear BEFORE the first section header
    (e.g. SpaceX.md has 3 rich intro paragraphs before "History"). This function
    recovers all body text after the metadata block.
    """

    # Metadata lines in the normalized docs use "Key: value" format and appear
    # right after the "# Title" line. We skip lines until we see the first
    # non-metadata, non-blank, non-heading line.
    _META_RE = re.compile(r"^[A-Z][A-Za-z0-9 /-]+:\s*.*$", re.IGNORECASE)

    lines = raw_doc_text.splitlines()
    in_meta = True
    body_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        # Skip the title line.
        if stripped.startswith("# ") and in_meta:
            continue
        if in_meta:
            if not stripped:
                continue  # blank between meta lines
            if _META_RE.match(stripped):
                continue  # metadata key: value
            # First non-meta, non-blank line: body starts here.
            in_meta = False
        body_lines.append(line)

    return "\n".join(body_lines).strip()
