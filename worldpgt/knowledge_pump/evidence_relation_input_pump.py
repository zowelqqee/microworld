"""Proposal-only relation-input training from local evidence-bearing graph edges.

Each overlay relation already pairs an evidence sentence with a canonical
predicate.  This pump obtains candidate input phrasing from the sentence's
verb phrases, aggregates only repeated non-conflicting observations, and
delegates graph construction to the proposal-only relation-input pump.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable

from worldpgt.knowledge_pump.relation_input_pump import build_relation_input_proposal, write_relation_input_proposal
from worldpgt.relation_extraction_v2.relation_policy import relation_intent_from_text


def _default_extract_phrases(text: str) -> list[str]:
    from worldpgt.entity_qa.semantic_question_parser import extract_verb_phrases
    return extract_verb_phrases(text)


def evidence_examples(
    items: Iterable[dict[str, Any]],
    *,
    extract_phrases: Callable[[str], list[str]] = _default_extract_phrases,
    validate_phrase: Callable[[str, str], bool] | None = None,
) -> list[dict[str, Any]]:
    """Make high-precision supervised examples from evidence-bearing edges.

    A dependency parser may misread a proper noun or a number as a verb.  The
    default independent lexical check therefore requires the extracted phrase
    to map back to the relation's same canonical predicate before it can become
    training data.  Novel, unverified phrasing stays out of the proposal graph.
    """

    examples: list[dict[str, Any]] = []
    validate = validate_phrase or (lambda phrase, predicate: relation_intent_from_text(phrase) == predicate)
    for ordinal, item in enumerate(items):
        if not isinstance(item, dict) or item.get("overlay_type") != "overlay_relation":
            continue
        predicate = item.get("predicate")
        evidence = item.get("evidence_text")
        if not isinstance(predicate, str) or not predicate or not isinstance(evidence, str) or not evidence.strip():
            continue
        edge_id = "evidence:" + str(item.get("relation_id") or ordinal)
        for phrase in extract_phrases(evidence):
            if not validate(phrase, predicate):
                continue
            examples.append({
                "id": edge_id + ":" + phrase,
                "question": phrase,
                "expected_predicate": [predicate],
                "training_source": "local_graph_evidence",
                "evidence_text": evidence,
            })
    return examples


def build_evidence_relation_input_proposal(
    items: Iterable[dict[str, Any]],
    *,
    min_support: int = 3,
    extract_phrases: Callable[[str], list[str]] = _default_extract_phrases,
    validate_phrase: Callable[[str, str], bool] | None = None,
) -> dict[str, Any]:
    examples = evidence_examples(items, extract_phrases=extract_phrases, validate_phrase=validate_phrase)
    result = build_relation_input_proposal(examples, min_support=min_support)
    result["training_lane"] = "local_graph_evidence"
    result["evidence_examples_seen"] = len(examples)
    return result


def write_evidence_relation_input_proposal(
    items: Iterable[dict[str, Any]],
    output_dir: str,
    *,
    min_support: int = 3,
) -> dict[str, Any]:
    """Write proposal artifacts from local evidence; runtime data stays untouched."""

    examples = evidence_examples(items)
    result = write_relation_input_proposal(examples, output_dir, min_support=min_support)
    result["training_lane"] = "local_graph_evidence"
    result["evidence_examples_seen"] = len(examples)
    return result
