"""Proposal report builder for the Self-Learning Loop v1.

Assembles the top-level :class:`SelfLearningReport`. The report explicitly
records that nothing was applied: trusted memory, accepted overlay, and promoted
overlay are all unmodified, and the proposal layer is not safe for general
runtime.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .artifact_loader import LoadedArtifacts
from .types import (
    AnalyzerRuleProposal,
    AdversarialCase,
    CurriculumRow,
    LearningOpportunity,
    PatternProposal,
    SelfLearningReport,
)


def _regression_status(loaded: LoadedArtifacts) -> Dict[str, Any]:
    status = loaded.json_data.get("promotion_regression_summary")
    if isinstance(status, dict) and status:
        source = status
    else:
        report = loaded.json_data.get("self_ingestion_report") or {}
        source = report.get("qa_regression_status", {}) if isinstance(report, dict) else {}
    observed: Dict[str, Any] = {}
    for suite, info in source.items():
        if isinstance(info, dict):
            observed[suite] = info.get("status")
    return observed


def _next_actions(opportunities: List[LearningOpportunity]) -> List[str]:
    cats = {opp.category for opp in opportunities}
    actions = [
        "Treat all artifacts as proposals only; do not auto-apply, auto-promote, "
        "or auto-activate anything.",
    ]
    if "missing_stable_knowledge" in cats:
        actions.append(
            "Route missing_stable_knowledge items through a local source document "
            "+ ingestion + quarantine + promotion before any answer is allowed."
        )
    if "volatile_review_candidate" in cats:
        actions.append(
            "Hold volatile_review_candidate items for human review; never auto-accept "
            "as stable facts."
        )
    if cats & {
        "adversarial_test_candidate",
        "weak_link_safety_case",
        "relation_inversion_guard_case",
        "unsupported_universal_guard_case",
        "current_claim_guard_case",
    }:
        actions.append(
            "Add proposed adversarial / guard cases to a regression gate "
            "(expecting audit/reject) before considering any rule change."
        )
    actions.append(
        "Review proposed extraction patterns and analyzer rules manually; keep "
        "their activation_status proposal_only until tests are written and pass."
    )
    return actions


def build_report(
    loaded: LoadedArtifacts,
    opportunities: List[LearningOpportunity],
    curriculum_rows: List[CurriculumRow],
    adversarial_cases: List[AdversarialCase],
    pattern_proposals: List[PatternProposal],
    analyzer_rule_proposals: List[AnalyzerRuleProposal],
) -> SelfLearningReport:
    by_category = Counter(opp.category for opp in opportunities)
    by_priority = Counter(opp.priority for opp in opportunities)
    return SelfLearningReport(
        safe_for_general_runtime=False,
        trusted_memory_modified=False,
        accepted_overlay_modified=False,
        promoted_overlay_modified=False,
        proposal_only=True,
        opportunities_total=len(opportunities),
        opportunities_by_category=dict(sorted(by_category.items())),
        opportunities_by_priority=dict(sorted(by_priority.items())),
        curriculum_rows_total=len(curriculum_rows),
        adversarial_rows_total=len(adversarial_cases),
        pattern_proposals_total=len(pattern_proposals),
        analyzer_rule_proposals_total=len(analyzer_rule_proposals),
        warnings=list(loaded.warnings),
        errors=list(loaded.errors),
        source_artifacts_read=list(loaded.source_artifacts_read),
        regression_status_observed=_regression_status(loaded),
        next_recommended_actions=_next_actions(opportunities),
    )
