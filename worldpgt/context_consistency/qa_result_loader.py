"""QA result loader for the Context-Pack QA Consistency Gate v1.

Discovers QA output CSVs under the experiments directory (tolerant of renamed
files via glob patterns), normalizes rows into :class:`LoadedQAResult`, and
reads the matching summaries when present. Missing optional outputs produce
warnings, not crashes.

Read-only. No network.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

from .types import LoadedQAResult

# Suite -> (preferred output filename, glob pattern, summary filename).
SUITE_SPECS: List[Tuple[str, str, str, str]] = [
    (
        "entity_qa_v1",
        "entity_qa_v1_outputs.csv",
        "*entity*v1_outputs*.csv",
        "entity_qa_v1_summary.json",
    ),
    (
        "entity_qa_expansion_v1",
        "entity_qa_expansion_v1_outputs.csv",
        "*entity*expansion*outputs*.csv",
        "entity_qa_expansion_v1_summary.json",
    ),
    (
        "entity_qa_adversarial_v1",
        "entity_qa_adversarial_v1_outputs.csv",
        "*adversarial*outputs*.csv",
        "entity_qa_adversarial_v1_summary.json",
    ),
    (
        "cross_page_qa_v1",
        "cross_page_qa_v1_outputs.csv",
        "*cross_page*outputs*.csv",
        "cross_page_qa_v1_summary.json",
    ),
]


def _to_bool(value: str):
    v = (value or "").strip().lower()
    if v in ("true", "yes", "1"):
        return True
    if v in ("false", "no", "0"):
        return False
    return None


@dataclass
class LoadedQA:
    rows: List[LoadedQAResult] = field(default_factory=list)
    summaries: Dict[str, dict] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_artifacts_read: List[str] = field(default_factory=list)


def _normalize_rows(suite: str, csv_rows: List[dict]) -> List[LoadedQAResult]:
    out: List[LoadedQAResult] = []
    for row in csv_rows:
        question = (row.get("question") or "").strip()
        if not question:
            continue
        out.append(
            LoadedQAResult(
                id=row.get("row_id", row.get("id", "")),
                source_suite=suite,
                question=question,
                qa_decision=(row.get("decision") or "").strip(),
                qa_intent=(row.get("detected_intent") or row.get("intent") or "").strip(),
                qa_answer=row.get("answer", "") or "",
                qa_correct=_to_bool(row.get("is_correct", "")),
                expected_decision=(row.get("expected_decision") or "").strip(),
            )
        )
    return out


def load_qa_results(experiments_dir: Path) -> LoadedQA:
    experiments_dir = Path(experiments_dir)
    out = LoadedQA()

    for suite, preferred, glob_pat, summary_name in SUITE_SPECS:
        # Locate the outputs CSV: preferred name first, else glob discovery.
        path = experiments_dir / preferred
        if not path.is_file():
            candidates = sorted(experiments_dir.glob(glob_pat))
            # Exclude files already claimed by a more specific suite (adversarial
            # / expansion) from the generic entity pattern.
            if suite == "entity_qa_v1":
                candidates = [
                    c
                    for c in candidates
                    if "adversarial" not in c.name and "expansion" not in c.name
                ]
            path = candidates[0] if candidates else None

        if path is None or not path.is_file():
            out.warnings.append(f"QA outputs missing for suite {suite} ({glob_pat})")
        else:
            try:
                with path.open("r", encoding="utf-8", newline="") as fh:
                    csv_rows = list(csv.DictReader(fh))
                out.rows.extend(_normalize_rows(suite, csv_rows))
                out.source_artifacts_read.append(path.name)
            except (OSError, ValueError) as exc:
                out.errors.append(f"failed reading {path.name}: {exc}")

        # Summary (optional).
        summary_path = experiments_dir / summary_name
        if summary_path.is_file():
            try:
                out.summaries[suite] = json.loads(
                    summary_path.read_text(encoding="utf-8")
                )
                out.source_artifacts_read.append(summary_name)
            except (OSError, ValueError) as exc:
                out.warnings.append(f"failed reading {summary_name}: {exc}")
        else:
            out.warnings.append(f"QA summary missing for suite {suite} ({summary_name})")

    return out
