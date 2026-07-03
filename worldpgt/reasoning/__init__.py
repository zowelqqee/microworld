"""Reasoning layer: explanatory chains, graph patterns, counterfactual traces.

Three additive capabilities over the same verified fact graph:

  explanation_builder — why a fact exists / makes sense (Part 1);
  pattern_discovery   — unnamed structural regularities with evidence (Part 2);
  counterfactual      — what in the model rests on a given fact (Part 3).

Everything is deterministic and structural: no fact is ever invented, every
link is a verified overlay fact, an explicit inference rule, or a measured
observation about the graph.
"""

from worldpgt.reasoning.counterfactual import analyze_counterfactual
from worldpgt.reasoning.explanation_builder import explain_fact
from worldpgt.reasoning.pattern_discovery import (
    discover_patterns,
    relevant_patterns,
    render_pattern_note,
)
from worldpgt.reasoning.pattern_store import load_patterns, save_patterns
from worldpgt.reasoning.reasoning_adapter import (
    AssistantReasoningResult,
    is_reasoning_compatible,
    try_answer_reasoning,
)
from worldpgt.reasoning.types import (
    CounterfactualTrace,
    ExplanationChain,
    ExplanationStep,
    GraphPattern,
)

__all__ = [
    "AssistantReasoningResult",
    "CounterfactualTrace",
    "ExplanationChain",
    "ExplanationStep",
    "GraphPattern",
    "analyze_counterfactual",
    "discover_patterns",
    "explain_fact",
    "is_reasoning_compatible",
    "load_patterns",
    "relevant_patterns",
    "render_pattern_note",
    "save_patterns",
    "try_answer_reasoning",
]
