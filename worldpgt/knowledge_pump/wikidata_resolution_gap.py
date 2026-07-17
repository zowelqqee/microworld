"""Deterministic sample selection for a read-only Wikidata resolution audit."""

from __future__ import annotations

import random
from typing import Any, Iterable


def seeded_original_failure_sample(rows: Iterable[dict[str, Any]], *, size: int, seed: int) -> list[dict[str, Any]]:
    """Choose a reproducible sample from original subjects lacking an exact QID."""

    failures = sorted(
        (
            dict(row) for row in rows
            if "original_331" in (row.get("cohorts") or ()) and not row.get("canonical_qid")
        ),
        key=lambda row: str(row.get("subject") or "").casefold(),
    )
    if len(failures) < size:
        raise ValueError(f"need {size} unresolved original subjects, got {len(failures)}")
    return sorted(random.Random(seed).sample(failures, size), key=lambda row: str(row["subject"]).casefold())


def classify_manual_sample(
    sample: Iterable[dict[str, Any]],
    classifications: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach manually reviewed, evidence-bearing classifications without inference."""

    rows: list[dict[str, Any]] = []
    for source in sample:
        subject = str(source.get("subject") or "")
        review = classifications.get(subject)
        if not isinstance(review, dict):
            raise ValueError(f"missing manual classification for {subject!r}")
        verdict = str(review.get("verdict") or "")
        if verdict not in {"genuinely_absent", "matching_gap"}:
            raise ValueError(f"invalid verdict for {subject!r}: {verdict!r}")
        if not str(review.get("rationale") or "").strip():
            raise ValueError(f"manual classification needs rationale for {subject!r}")
        rows.append({
            "subject": subject,
            "prior_resolution_status": source.get("canonical_resolution_status"),
            "verdict": verdict,
            "failure_type": review.get("failure_type"),
            "correct_wikidata_qid": review.get("correct_wikidata_qid"),
            "correct_wikidata_label": review.get("correct_wikidata_label"),
            "rationale": review["rationale"],
        })
    return rows
