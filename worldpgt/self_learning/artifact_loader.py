"""Artifact loader for the Self-Learning Loop v1.

Safely loads the EXISTING QA / regression / quarantine / ingestion / promotion
artifacts. Required core artifacts must exist (we fail loudly otherwise);
optional artifacts (chiefly the ``*_outputs.csv`` files and a few secondary
JSON reports) are allowed to be missing and only produce warnings.

No network. No writes. Read-only over the experiments directory.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class MissingRequiredArtifact(FileNotFoundError):
    """Raised when a required core artifact is absent."""


# Logical name -> path relative to the experiments directory.
REQUIRED_JSON: Dict[str, str] = {
    "answer_planner_summary": "answer_planner_v1_summary.json",
    "qa_generalization_summary": "qa_generalization_test_v1_summary.json",
    "entity_qa_summary": "entity_qa_v1_summary.json",
    "entity_qa_expansion_summary": "entity_qa_expansion_v1_summary.json",
    "entity_qa_adversarial_summary": "entity_qa_adversarial_v1_summary.json",
    "cross_page_qa_summary": "cross_page_qa_v1_summary.json",
    "quarantine": "self_ingestion_v1/quarantine.json",
    "self_ingestion_report": "self_ingestion_v1/self_ingestion_report.json",
    "promotion_report": "self_ingestion_v1/promotion/promotion_report.json",
}

OPTIONAL_JSON: Dict[str, str] = {
    "manifest_errors": "self_ingestion_v1/manifest_errors.json",
    "overlay_delta_proposal": "self_ingestion_v1/overlay_delta_proposal.json",
    "promotion_validation": "self_ingestion_v1/promotion/promotion_validation.json",
    "promotion_regression_summary": (
        "self_ingestion_v1/promotion/promotion_regression_summary.json"
    ),
}

# Optional QA / regression output CSVs. Missing ones become warnings.
OPTIONAL_CSV: Dict[str, str] = {
    "answer_planner_outputs": "answer_planner_v1_outputs.csv",
    "qa_generalization_outputs": "qa_generalization_test_v1_outputs.csv",
    "entity_qa_outputs": "entity_qa_v1_outputs.csv",
    "entity_qa_expansion_outputs": "entity_qa_expansion_v1_outputs.csv",
    "entity_qa_adversarial_outputs": "entity_qa_adversarial_v1_outputs.csv",
    "cross_page_qa_outputs": "cross_page_qa_v1_outputs.csv",
}


@dataclass
class LoadedArtifacts:
    """Container for everything the loader managed to read."""

    json_data: Dict[str, Any] = field(default_factory=dict)
    csv_data: Dict[str, List[Dict[str, str]]] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    source_artifacts_read: List[str] = field(default_factory=list)


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


class ArtifactLoader:
    """Loads self-learning input artifacts from an experiments directory."""

    def __init__(self, experiments_dir: Path) -> None:
        self.experiments_dir = Path(experiments_dir)

    def load(self) -> LoadedArtifacts:
        result = LoadedArtifacts()

        # Required JSON: fail loudly if any is missing or unparsable.
        for name, rel in REQUIRED_JSON.items():
            path = self.experiments_dir / rel
            if not path.is_file():
                raise MissingRequiredArtifact(
                    f"required artifact missing: {rel} (expected at {path})"
                )
            try:
                result.json_data[name] = _load_json(path)
            except (ValueError, OSError) as exc:
                raise MissingRequiredArtifact(
                    f"required artifact unreadable: {rel}: {exc}"
                ) from exc
            result.source_artifacts_read.append(rel)

        # Optional JSON: warn, do not crash.
        for name, rel in OPTIONAL_JSON.items():
            path = self.experiments_dir / rel
            if not path.is_file():
                result.warnings.append(f"optional artifact missing: {rel}")
                continue
            try:
                result.json_data[name] = _load_json(path)
                result.source_artifacts_read.append(rel)
            except (ValueError, OSError) as exc:
                result.warnings.append(f"optional artifact unreadable: {rel}: {exc}")

        # Optional CSV: warn, do not crash.
        for name, rel in OPTIONAL_CSV.items():
            path = self.experiments_dir / rel
            if not path.is_file():
                result.warnings.append(f"optional outputs CSV missing: {rel}")
                continue
            try:
                result.csv_data[name] = _load_csv(path)
                result.source_artifacts_read.append(rel)
            except (ValueError, OSError) as exc:
                result.warnings.append(f"optional outputs CSV unreadable: {rel}: {exc}")

        return result
