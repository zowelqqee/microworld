"""Path safety validator for Multi-hop QA v1.

Validates that a found 2-hop path is safe to answer, given the question type.

Blocks:
  - direct_only questions (Who founded X? / Who owns X?)
  - unsupported chain types
  - paths with volatile stability
  - paths with high risk
  - paths using current-sensitive predicates (see relation_policy.CURRENT_SENSITIVE_PREDICATES)

All blocking reasons are returned as a structured string so callers can
categorize the audit: "volatile_relation:pred", "high_risk_relation:pred",
"current_sensitive_relation:pred", etc.
"""
from __future__ import annotations

from typing import Optional

from worldpgt.multihop_qa.types import HopEdge, MultihopPath, MultihopQuestion
from worldpgt.relation_extraction_v2.relation_policy import CURRENT_SENSITIVE_PREDICATES
from worldpgt.knowledge.temporal_classification import requires_as_of


def chain_label(pred1: str, pred2: str) -> str:
    """Canonical chain label from two predicate strings."""
    return f"{pred1}_then_{pred2}"


def validate_hop_safety(hop: HopEdge) -> tuple[bool, Optional[str]]:
    """Validate one explicit relation hop against the shared path safety policy."""
    if requires_as_of(hop.temporal_class) and not hop.as_of:
        return False, f"{hop.temporal_class}_missing_as_of:{hop.predicate}"
    if hop.stability == "volatile":
        return False, f"volatile_relation:{hop.predicate}"
    if hop.risk == "high":
        return False, f"high_risk_relation:{hop.predicate}"
    if hop.predicate in CURRENT_SENSITIVE_PREDICATES and not (
        hop.temporal_class == "snapshot" and hop.as_of
    ):
        return False, f"current_sensitive_relation:{hop.predicate}"
    return True, None


def validate_path(
    path: MultihopPath,
    question: MultihopQuestion,
) -> tuple[bool, Optional[str]]:
    """Returns (is_valid, reason_if_invalid).

    Reason prefix conventions used by callers for metric categorisation:
      "direct_relation_query_blocked"
      "unsupported_chain_type"
      "volatile_relation:<pred>"
      "high_risk_relation:<pred>"
      "current_sensitive_relation:<pred>"
    """
    if question.is_direct_relation or question.chain_type == "direct_only":
        return False, "direct_relation_query_blocked"

    if question.chain_type == "unsupported":
        return False, "unsupported_chain_type"

    for hop in (path.hop1, path.hop2):
        valid, reason = validate_hop_safety(hop)
        if not valid:
            return False, reason

    return True, None
