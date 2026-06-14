"""Source manifest loader for Wikipedia Self-Ingestion v1.

Loads and validates a local sources manifest. Rejects any source that is a URL /
network source, has an unknown source_type or trust_tier, or whose local file is
missing. No network access is ever performed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from worldpgt.self_ingestion.types import (
    SOURCE_TYPES,
    TRUST_TIERS,
    ManifestLoadResult,
    SourceEntry,
)

_URL_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")  # http://, https://, ftp://, ...
_NETWORKY = ("http://", "https://", "ftp://", "ftps://", "ssh://", "www.")


def _looks_like_url(path: str) -> bool:
    p = (path or "").strip().lower()
    if _URL_RE.match(p):
        return True
    return any(p.startswith(n) or n in p for n in _NETWORKY)


def load_manifest(path: str | Path) -> ManifestLoadResult:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("sources manifest must be a JSON list of source entries")

    result = ManifestLoadResult()
    for entry in raw:
        sid = entry.get("source_id", "")
        spath = entry.get("path", "")
        stype = entry.get("source_type", "")
        tier = entry.get("trust_tier", "")

        if _looks_like_url(spath):
            result.errors.append({"source_id": sid, "reason": "url_or_network_source_rejected"})
            continue
        if stype not in SOURCE_TYPES:
            result.errors.append({"source_id": sid, "reason": f"unknown_source_type:{stype}"})
            continue
        if tier not in TRUST_TIERS:
            result.errors.append({"source_id": sid, "reason": f"unknown_trust_tier:{tier}"})
            continue
        if not spath or not Path(spath).is_file():
            result.errors.append({"source_id": sid, "reason": "source_file_missing"})
            continue

        result.sources.append(SourceEntry(
            source_id=sid,
            path=spath,
            source_type=stype,
            trust_tier=tier,
            expected_domain=entry.get("expected_domain", ""),
            allow_auto_accept=bool(entry.get("allow_auto_accept", False)),
            notes=entry.get("notes", ""),
        ))
    return result
