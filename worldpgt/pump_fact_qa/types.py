"""Dataclasses for Pump Fact QA Benchmark v1."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Prompt categories.
CATEGORY_POSITIVE = "positive"
CATEGORY_ADVERSARIAL = "adversarial"
CATEGORY_CURRENT = "current"

# Fact kinds carried through the benchmark.
FACT_KIND_RELATION = "relation"
FACT_KIND_DEFINITION = "definition"


@dataclass
class PumpFact:
    """A single precision-filtered pump fact (relation or definition)."""

    kind: str  # FACT_KIND_RELATION | FACT_KIND_DEFINITION
    subject: str
    predicate: str
    obj: str  # object for relations, definition text for definitions
    source_page: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAPrompt:
    """A generated benchmark question with its expectation."""

    prompt_id: str
    question: str
    category: str
    expected_decision: str  # "answer" | "audit"
    predicate: str = ""
    subject: str = ""
    obj: str = ""
    expected_tokens: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class QAResult:
    """Outcome of running one prompt through the assistant surface."""

    prompt: QAPrompt
    decision: str
    support_kind: str
    supported_by_context: bool
    overlay_mode: str
    answer_text: str
    risk_flags: list[str]
    valid: bool
    classification: str
    is_critical_failure: bool

    def to_dict(self) -> dict[str, Any]:
        data = {"prompt": self.prompt.to_dict()}
        data.update(
            {
                "decision": self.decision,
                "support_kind": self.support_kind,
                "supported_by_context": self.supported_by_context,
                "overlay_mode": self.overlay_mode,
                "answer_text": self.answer_text,
                "risk_flags": list(self.risk_flags),
                "valid": self.valid,
                "classification": self.classification,
                "is_critical_failure": self.is_critical_failure,
            }
        )
        return data


# Failure / classification taxonomy.
CLASS_OK = "ok"
CLASS_PLANNER_GAP = "planner_gap"
CLASS_SOURCE_OR_VOLATILITY_SENSITIVE = "source_or_volatility_sensitive"
CLASS_ENTITY_MATCH_GAP = "entity_match_gap"
CLASS_PREDICATE_MAPPING_GAP = "predicate_mapping_gap"
CLASS_RENDERER_SURFACE_ISSUE = "renderer_surface_issue"
CLASS_UNSUPPORTED_ANSWER = "unsupported_answer"
CLASS_WRONG_ANSWER = "wrong_answer"
CLASS_UNSAFE_CURRENT_ANSWER = "unsafe_current_answer"
CLASS_INVERSION_FALSE_ANSWER = "inversion_false_answer"
CLASS_EXPECTED_AUDIT_BUT_ANSWERED = "expected_audit_but_answered"
CLASS_EXPECTED_ANSWER_BUT_AUDITED = "expected_answer_but_audited"
CLASS_MISSING_SUPPORT_TRACE = "missing_support_trace"
CLASS_OVERLAY_ADAPTER_ERROR = "overlay_adapter_error"

# Classifications that count as critical (must be zero for a clean benchmark).
CRITICAL_CLASSIFICATIONS = frozenset(
    {
        CLASS_WRONG_ANSWER,
        CLASS_UNSUPPORTED_ANSWER,
        CLASS_UNSAFE_CURRENT_ANSWER,
        CLASS_INVERSION_FALSE_ANSWER,
        CLASS_OVERLAY_ADAPTER_ERROR,
    }
)
