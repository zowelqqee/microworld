"""Extract local-only frontier titles for Knowledge Pump v1."""

from __future__ import annotations

import json
import re
from pathlib import Path

from worldpgt.knowledge_pump.types import FrontierTitle

_PHRASE_RE = re.compile(
    r"\b(?:[A-Z][A-Za-z0-9&.'-]+|[A-Z]{2,})(?:[ \t]+(?:[A-Z][A-Za-z0-9&.'-]+|[A-Z]{2,})){0,5}\b"
)
_MARKDOWN_LINK_RE = re.compile(r"\[\[([^]|#]+)(?:\|[^]]+)?\]\]|\[([^]]+)\]\((?:/wiki/)?([^)#]+)\)")

_GENERIC = {
    "Source", "Retrieved", "Revision ID", "Raw text SHA256", "Status",
    "Safe", "Requires", "Wikipedia", "History", "References", "External",
    "United States", "New York", "The", "This", "It", "He", "She",
    "According", "After", "Although", "They", "What", "When", "With",
    "Therefore", "During", "Following", "From", "Since", "However",
    "Some", "Many", "Most", "More", "Both", "Each", "Other", "Another",
    "Company", "Corporation", "Corp", "Business", "Market", "Model",
    "Technology", "Development", "Coverage", "List", "SHA256", "SHA-2",
}
_MONTHS = {
    "January", "February", "March", "April", "May", "June", "July",
    "August", "September", "October", "November", "December",
}
_FRAGMENT_STARTERS = {
    "In", "On", "At", "From", "When", "What", "While", "With", "After",
    "Before", "During", "Following", "Therefore", "Although", "However",
    "Since", "Does", "Is", "Are", "Was", "Were", "They", "This", "That",
}
# Words that are never legitimate title-terminal words — they signal a prose continuation.
_PROSE_TERMINAL_WORDS = frozenset({
    "In", "It", "The", "This", "A", "An", "During", "After", "Before",
    "However", "Beginning", "Starting", "When", "Where", "Despite",
    "He", "She", "They", "And", "But", "From", "On", "At", "Of", "To",
    "Which", "Who", "What", "That", "Addressing", "Conventional",
})
# Short multi-letter abbreviations that may legitimately precede ". " in a title.
_SAFE_ABBREVS_BEFORE_PERIOD_SPACE = frozenset({
    "St", "Mt", "Mr", "Mrs", "Ms", "Dr", "Jr", "Sr", "Lt", "Capt", "vs",
    "Fig", "No", "Vol", "Rev",
})
_COMMON_SINGLE_WORDS = _GENERIC | _MONTHS | {
    "American", "British", "Chinese", "European", "French", "German",
    "Italian", "Russian", "Japanese", "Canadian", "India", "Japan",
    "China", "Russia", "France", "Germany", "Europe", "California",
    "Florida", "Earth", "Moon", "Mars", "Space", "Internet", "Energy",
    "Battery", "Solar", "Renewable", "Engineering", "Journalism",
    "Analytics", "Businessperson", "Aerospace", "Adposition",
}
_ALLOWLISTED_SINGLE = {
    "SpaceX", "Starlink", "Neuralink", "OpenAI", "SolarCity", "NASA",
    "Tesla", "PayPal", "Zip2", "LVMH", "Forbes", "Bloomberg",
    "Arianespace", "Airbus", "Boeing", "Microsoft", "Reuters",
}
_TRUSTED_SOURCES = {
    "snapshot_link", "relation_v2", "knowledge_requests", "overlay_entity",
    "overlay_alias", "overlay_relation", "weak_context_link",
}


def _clean_title(title: str) -> str:
    title = re.sub(r"\s+", " ", title.replace("_", " ")).strip(" .,:;()[]{}")
    return title


def _has_prose_period_break(title: str) -> bool:
    """Return True if title contains a sentence-boundary period (not an initial or abbreviation).

    "SpaceX. It" → True (prose fragment).
    "John F. Kennedy" → False (initial before period).
    "B.C. Forbes" → False (abbreviation with internal dots).
    "U.S. Army" → False (two single-letter initials).
    """
    parts = title.split(". ")
    if len(parts) < 2:
        return False
    for part in parts[:-1]:
        atoms = part.split()
        if not atoms:
            continue
        last_atom = atoms[-1]
        core = last_atom.rstrip(".")
        if len(core) <= 1:
            continue
        if "." in core:
            continue
        if core in _SAFE_ABBREVS_BEFORE_PERIOD_SPACE:
            continue
        return True
    return False


def _usable(title: str, source: str) -> bool:
    if not title or title in _GENERIC:
        return False
    if "sha256" in title.lower() or title.lower().startswith(("raw text", "retrieved at", "revision id")):
        return False
    if title in _MONTHS:
        return False
    if "'s" in title or title.endswith("'"):
        return False
    if len(title) < 4 or len(title) > 90:
        return False
    if title.lower().startswith(("list of ", "category:", "template:")):
        return False
    words = title.split()
    if words and words[0] in _FRAGMENT_STARTERS:
        return False
    if len(words) == 1:
        if title not in _ALLOWLISTED_SINGLE:
            return False
        return True
    if len(words) == 2 and words[0] in _FRAGMENT_STARTERS:
        return False
    if any(w in _MONTHS for w in words) and words[0] in {"In", "On", "By", "Since"}:
        return False
    if words[-1] in _PROSE_TERMINAL_WORDS:
        return False
    if _has_prose_period_break(title):
        return False
    if source == "snapshot_phrase":
        if words[0] in _COMMON_SINGLE_WORDS:
            return False
        if title.endswith(" usually") or title.endswith(" typically"):
            return False
        # Require a title-like phrase: at least one non-generic proper token,
        # and avoid sentence fragments such as "Therefore turbocharged engines".
        if not any(w[:1].isupper() and w not in _COMMON_SINGLE_WORDS for w in words):
            return False
    if re.fullmatch(r"\d+|\d{4}", title):
        return False
    return True


def _add(out: dict[str, FrontierTitle], title: str, source: str, reason: str, weight: int) -> None:
    clean = _clean_title(title)
    if not _usable(clean, source):
        return
    key = clean.casefold()
    if key in out:
        out[key].weight += weight
        if source not in out[key].source:
            out[key].source += f"|{source}"
        return
    out[key] = FrontierTitle(clean, source, reason, weight)


def _load_json(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def extract_frontier_titles(
    snapshots_dir: str | Path,
    relation_candidates_path: str | Path | None = None,
    knowledge_requests_path: str | Path | None = None,
    overlay_paths: list[str | Path] | None = None,
) -> list[FrontierTitle]:
    out: dict[str, FrontierTitle] = {}
    snapshots = Path(snapshots_dir)

    for doc_path in sorted((snapshots / "normalized_docs").glob("*.md")):
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        for match in _MARKDOWN_LINK_RE.finditer(text):
            title = match.group(1) or match.group(2) or match.group(3) or ""
            _add(out, title, "snapshot_link", f"linked from {doc_path.name}", 8)
        for match in _PHRASE_RE.finditer(text[:12000]):
            phrase = match.group(0)
            _add(out, phrase, "snapshot_phrase", f"capitalized phrase in {doc_path.name}", 2)
            words = phrase.split()
            for width in (2, 3):
                if len(words) <= width:
                    continue
                for start in range(0, len(words) - width + 1):
                    _add(
                        out,
                        " ".join(words[start:start + width]),
                        "snapshot_phrase",
                        f"capitalized subphrase in {doc_path.name}",
                        1,
                    )

    if relation_candidates_path:
        rows = _load_json(Path(relation_candidates_path)) or []
        for row in rows:
            _add(out, str(row.get("subject", "")), "relation_v2", "relation candidate subject", 10)
            _add(out, str(row.get("object", "")), "relation_v2", "relation candidate object", 10)

    if knowledge_requests_path:
        rows = _load_json(Path(knowledge_requests_path)) or []
        if isinstance(rows, dict):
            rows = rows.get("requests") or rows.get("items") or []
        for row in rows:
            if isinstance(row, dict):
                for key in ("title", "entity", "subject", "target_title", "source_title"):
                    _add(out, str(row.get(key, "")), "knowledge_requests", f"knowledge request {key}", 7)
                text = " ".join(str(v) for v in row.values())
                for match in _PHRASE_RE.finditer(text):
                    _add(out, match.group(0), "knowledge_requests", "knowledge request phrase", 3)

    for overlay_path in overlay_paths or []:
        rows = _load_json(Path(overlay_path)) or []
        for item in rows:
            if not isinstance(item, dict):
                continue
            if item.get("overlay_type") == "overlay_entity":
                _add(out, str(item.get("label", "")), "overlay_entity", "existing overlay entity", 6)
                for alias in item.get("aliases") or []:
                    _add(out, str(alias), "overlay_alias", "existing overlay alias", 2)
            if item.get("overlay_type") == "overlay_relation":
                _add(out, str(item.get("subject", "")), "overlay_relation", "relation subject", 5)
                _add(out, str(item.get("object", "")), "overlay_relation", "relation object", 5)
            if item.get("overlay_type") == "overlay_context_link":
                _add(out, str(item.get("target", "")), "weak_context_link", "weak context target", 4)

    return sorted(out.values(), key=lambda x: (-x.weight, x.title.casefold()))
