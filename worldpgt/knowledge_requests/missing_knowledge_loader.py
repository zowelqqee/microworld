"""Loader for Candidate Knowledge Request v1.

Reads the Self-Learning Loop v1 opportunities and the Curriculum Regression
Gate v1 outputs, and normalizes opportunities into :class:`MissingKnowledgeInput`
objects. The regression outputs are used to confirm each opportunity's *current*
decision/behavior (it should currently audit).

- Required: learning_opportunities.json. Missing -> fail loudly.
- Optional: regression outputs/summary, self-ingestion / promotion artifacts.
  Missing -> warnings, not crashes.

Read-only. No network.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from .types import MissingKnowledgeInput

MISSING_CATEGORY = "missing_stable_knowledge"


class MissingRequiredInput(FileNotFoundError):
    """Raised when a required input artifact is absent."""


@dataclass
class LoadedInputs:
    opportunities: List[dict] = field(default_factory=list)
    opportunity_summary: Optional[dict] = None
    regression_outputs: List[dict] = field(default_factory=list)
    regression_summary: Optional[dict] = None
    quarantine: Optional[list] = None
    self_ingestion_report: Optional[dict] = None
    promotion_report: Optional[dict] = None
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_artifacts_read: List[str] = field(default_factory=list)


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class MissingKnowledgeLoader:
    def __init__(self, experiments_dir: Path) -> None:
        self.experiments = Path(experiments_dir)
        self.self_learning = self.experiments / "self_learning_v1"
        self.regression = self.experiments / "curriculum_regression_v1"
        self.self_ingestion = self.experiments / "self_ingestion_v1"

    def load(self) -> LoadedInputs:
        out = LoadedInputs()

        opp_path = self.self_learning / "learning_opportunities.json"
        if not opp_path.is_file():
            raise MissingRequiredInput(
                f"required artifact missing: {opp_path}"
            )
        out.opportunities = _read_json(opp_path)
        out.source_artifacts_read.append("self_learning_v1/learning_opportunities.json")

        self._opt_json(
            out,
            self.self_learning / "learning_opportunity_summary.json",
            "opportunity_summary",
            "self_learning_v1/learning_opportunity_summary.json",
        )
        self._opt_csv(
            out,
            self.regression / "curriculum_regression_outputs.csv",
            "regression_outputs",
            "curriculum_regression_v1/curriculum_regression_outputs.csv",
        )
        self._opt_json(
            out,
            self.regression / "curriculum_regression_summary.json",
            "regression_summary",
            "curriculum_regression_v1/curriculum_regression_summary.json",
        )
        self._opt_json(
            out,
            self.self_ingestion / "quarantine.json",
            "quarantine",
            "self_ingestion_v1/quarantine.json",
        )
        self._opt_json(
            out,
            self.self_ingestion / "self_ingestion_report.json",
            "self_ingestion_report",
            "self_ingestion_v1/self_ingestion_report.json",
        )
        self._opt_json(
            out,
            self.self_ingestion / "promotion" / "promotion_report.json",
            "promotion_report",
            "self_ingestion_v1/promotion/promotion_report.json",
        )

        return out

    def _opt_json(self, out: LoadedInputs, path: Path, attr: str, rel: str) -> None:
        if not path.is_file():
            out.warnings.append(f"optional artifact missing: {rel}")
            return
        try:
            setattr(out, attr, _read_json(path))
            out.source_artifacts_read.append(rel)
        except (ValueError, OSError) as exc:
            out.warnings.append(f"optional artifact unreadable: {rel}: {exc}")

    def _opt_csv(self, out: LoadedInputs, path: Path, attr: str, rel: str) -> None:
        if not path.is_file():
            out.warnings.append(f"optional artifact missing: {rel}")
            return
        try:
            setattr(out, attr, _read_csv(path))
            out.source_artifacts_read.append(rel)
        except (ValueError, OSError) as exc:
            out.warnings.append(f"optional artifact unreadable: {rel}: {exc}")


def normalize_missing_inputs(loaded: LoadedInputs) -> List[MissingKnowledgeInput]:
    """Build MissingKnowledgeInput objects for missing_stable_knowledge only."""
    # Map source_opportunity_id -> observed decision from regression outputs.
    observed_by_opp: Dict[str, str] = {}
    for row in loaded.regression_outputs or []:
        opp_id = row.get("source_opportunity_id")
        if opp_id:
            observed_by_opp[opp_id] = (row.get("observed_decision") or "").strip()

    out: List[MissingKnowledgeInput] = []
    for opp in loaded.opportunities or []:
        if opp.get("category") != MISSING_CATEGORY:
            continue
        opp_id = opp.get("id", "")
        observed = observed_by_opp.get(opp_id) or opp.get("observed_decision") or "audit"
        current_behavior = (
            "audits safely until explicit stable evidence is ingested and promoted"
            if observed == "audit"
            else f"current decision observed as '{observed}'"
        )
        out.append(
            MissingKnowledgeInput(
                id=opp_id,
                category=opp.get("category", ""),
                question=opp.get("question_or_claim", ""),
                current_decision=observed or "audit",
                current_behavior=current_behavior,
                source_artifact=opp.get("source_artifact", ""),
                priority=opp.get("priority"),
                observed_decision=opp.get("observed_decision"),
                expected_policy=opp.get("expected_policy"),
                reason=opp.get("reason"),
                suggested_next_step=opp.get("suggested_next_step"),
            )
        )
    return out
