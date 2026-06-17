"""Loader for the Curriculum Regression Gate v1.

Reads the proposal-only artifacts produced by the Self-Learning Loop v1.

- The two proposed CSVs (curriculum + adversarial) are required: missing ones
  fail loudly.
- The proposal JSONs (extraction patterns, analyzer rules, self-learning
  report) are optional: missing ones produce warnings, not crashes.

Read-only. No network.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .types import AdversarialCase, CurriculumCase, LoadedProposals


class MissingProposalArtifact(FileNotFoundError):
    """Raised when a required proposal CSV is absent."""


CURRICULUM_CSV = "proposed_curriculum_v1.csv"
ADVERSARIAL_CSV = "proposed_adversarial_cases_v1.csv"
EXTRACTION_PATTERNS_JSON = "proposed_extraction_patterns_v1.json"
ANALYZER_RULES_JSON = "proposed_analyzer_rules_v1.json"
SELF_LEARNING_REPORT_JSON = "self_learning_report_v1.json"


def _read_csv(path: Path) -> list:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class CurriculumCaseLoader:
    def __init__(self, self_learning_dir: Path) -> None:
        self.dir = Path(self_learning_dir)

    def load(self) -> LoadedProposals:
        out = LoadedProposals()

        curriculum_path = self.dir / CURRICULUM_CSV
        adversarial_path = self.dir / ADVERSARIAL_CSV
        for required in (curriculum_path, adversarial_path):
            if not required.is_file():
                raise MissingProposalArtifact(
                    f"required proposal artifact missing: {required}"
                )

        for row in _read_csv(curriculum_path):
            out.curriculum_cases.append(
                CurriculumCase(
                    id=row.get("id", ""),
                    category=row.get("category", ""),
                    question=row.get("question", ""),
                    expected_decision=(row.get("expected_decision", "") or "").strip(),
                    expected_answer_contains=row.get("expected_answer_contains", ""),
                    expected_audit_reason=row.get("expected_audit_reason", ""),
                    source_opportunity_id=row.get("source_opportunity_id", ""),
                    activation_status=row.get("activation_status", ""),
                )
            )
        out.source_artifacts_read.append(CURRICULUM_CSV)

        for row in _read_csv(adversarial_path):
            out.adversarial_cases.append(
                AdversarialCase(
                    id=row.get("id", ""),
                    attack_category=row.get("attack_category", ""),
                    question=row.get("question", ""),
                    expected_decision=(row.get("expected_decision", "") or "").strip(),
                    expected_safety_reason=row.get("expected_safety_reason", ""),
                    source_opportunity_id=row.get("source_opportunity_id", ""),
                )
            )
        out.source_artifacts_read.append(ADVERSARIAL_CSV)

        # Optional proposal JSONs.
        self._load_optional_json(out, EXTRACTION_PATTERNS_JSON, "extraction_patterns")
        self._load_optional_json(out, ANALYZER_RULES_JSON, "analyzer_rules")
        self._load_optional_json(out, SELF_LEARNING_REPORT_JSON, "self_learning_report")

        return out

    def _load_optional_json(self, out: LoadedProposals, fname: str, attr: str) -> None:
        path = self.dir / fname
        if not path.is_file():
            out.warnings.append(f"optional proposal artifact missing: {fname}")
            return
        try:
            data = _read_json(path)
        except (ValueError, OSError) as exc:
            out.warnings.append(f"optional proposal artifact unreadable: {fname}: {exc}")
            return
        setattr(out, attr, data)
        out.source_artifacts_read.append(fname)
