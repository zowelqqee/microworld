"""Readiness rules for local Wikipedia snapshots."""

from __future__ import annotations

from pathlib import Path

from worldpgt.wiki_snapshots.types import PageSnapshot, ReadinessResult

MIN_RAW_TEXT_CHARS = 500


def looks_like_disambiguation(snapshot: PageSnapshot) -> bool:
    title = (snapshot.normalized_title or snapshot.title).lower()
    head = (snapshot.raw_text or "")[:1000].lower()
    if "disambiguation" in title:
        return True
    markers = (
        "may refer to:",
        "may also refer to:",
        "may refer to",
        "{{disambiguation",
        "__disambig__",
        "wikipedia disambiguation",
    )
    return any(marker in head for marker in markers)


def evaluate_snapshot_readiness(
    snapshot: PageSnapshot,
    normalized_doc_path: str | Path | None,
    minimum_chars: int = MIN_RAW_TEXT_CHARS,
) -> ReadinessResult:
    reasons: list[str] = []
    if snapshot.fetch_status != "success":
        reasons.append(f"fetch_status:{snapshot.fetch_status}")
    if not snapshot.raw_text or len(snapshot.raw_text) <= minimum_chars:
        reasons.append("raw_text_missing_or_short")
    if normalized_doc_path is None or not Path(normalized_doc_path).is_file():
        reasons.append("normalized_doc_missing")
    if not snapshot.source_url:
        reasons.append("source_url_missing")
    if not snapshot.retrieved_at:
        reasons.append("retrieved_at_missing")
    if not snapshot.raw_text_sha256:
        reasons.append("raw_text_sha256_missing")
    if snapshot.fetch_status in {"missing", "error"} or snapshot.error:
        reasons.append("missing_or_error_page")
    if looks_like_disambiguation(snapshot):
        reasons.append("disambiguation_like_page")

    return ReadinessResult(
        title=snapshot.normalized_title or snapshot.title,
        ready_for_self_ingestion=not reasons,
        requires_quarantine=True,
        safe_for_general_runtime=False,
        reasons=reasons,
    )

