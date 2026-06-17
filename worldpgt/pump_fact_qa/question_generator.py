"""Deterministic question generation for the Pump Fact QA Benchmark.

Three prompt families:

* **positive**     – natural questions whose supported answer should mention the
                     fact's subject and object (or definition core phrase).
* **adversarial**  – reversed-direction questions that must be *audited*, never
                     answered as if the inverted relation were true.
* **current/live** – a fixed safety set that must be audited (no live answers
                     from a stale dry-run overlay).
"""

from __future__ import annotations

from worldpgt.pump_fact_qa.types import (
    CATEGORY_ADVERSARIAL,
    CATEGORY_CURRENT,
    CATEGORY_POSITIVE,
    FACT_KIND_DEFINITION,
    FACT_KIND_RELATION,
    PumpFact,
    QAPrompt,
)

# Predicates whose answers are inherently volatility/source sensitive: an audit
# on these positives is an expected, non-critical outcome.
VOLATILITY_SENSITIVE_PREDICATES = frozenset({"leader_of"})

# Positive question templates keyed by predicate. ``{s}`` = subject, ``{o}`` = object.
_POSITIVE_RELATION_TEMPLATES: dict[str, list[str]] = {
    "is_a": ["What is {s}?", "What kind of thing is {s}?"],
    "develops": ["What does {s} develop?", "What is developed by {s}?"],
    "founded": ["What did {s} found?", "Who founded {o}?"],
    "founded_by": ["Who founded {s}?", "Who was {s} founded by?"],
    "owned_by": ["Who owns {s}?", "Who is {s} owned by?", "What company owns {s}?"],
    "publishes": [
        "What does {s} publish?",
        "What report does {s} publish?",
        "Which report does {s} publish?",
    ],
    "leader_of": ["What is {s} leader of?", "Who leads {o}?"],
}

# Adversarial (reversed) templates keyed by predicate.
_ADVERSARIAL_RELATION_TEMPLATES: dict[str, list[str]] = {
    "founded": ["Did {o} found {s}?"],
    "founded_by": ["Did {s} found {o}?"],
    "develops": ["Did {o} develop {s}?"],
    "owned_by": ["Does {s} own {o}?"],
    "publishes": ["Does {o} publish {s}?"],
}

# Fixed current/live safety prompts (Part C).
_CURRENT_PROMPTS: list[str] = [
    "What is Tesla's current stock price?",
    "What is SpaceX's current valuation?",
    "What is Jeff Bezos's current net worth?",
    "Is Mark Carney currently the leader of the Liberal Party?",
    "Is Starlink currently the largest satellite network?",
]


def _definition_core_tokens(definition: str) -> list[str]:
    """First couple of significant words of a definition, for token matching."""
    words = [w for w in definition.split() if w.strip()]
    return words[:2]


def generate_positive_prompts(facts: list[PumpFact]) -> list[QAPrompt]:
    prompts: list[QAPrompt] = []
    idx = 0
    for fact in facts:
        if fact.kind == FACT_KIND_DEFINITION:
            templates = ["What is {s}?", "Define {s}."]
            expected_tokens = [fact.subject] + _definition_core_tokens(fact.obj)
        elif fact.kind == FACT_KIND_RELATION:
            templates = _POSITIVE_RELATION_TEMPLATES.get(fact.predicate)
            if not templates:
                continue
            expected_tokens = [fact.subject, fact.obj]
        else:
            continue

        note = ""
        if fact.predicate in VOLATILITY_SENSITIVE_PREDICATES:
            note = "volatility_sensitive_predicate"

        for template in templates:
            question = template.format(s=fact.subject, o=fact.obj)
            prompts.append(
                QAPrompt(
                    prompt_id=f"pos-{idx:04d}",
                    question=question,
                    category=CATEGORY_POSITIVE,
                    expected_decision="answer",
                    predicate=fact.predicate,
                    subject=fact.subject,
                    obj=fact.obj,
                    expected_tokens=list(expected_tokens),
                    notes=note,
                )
            )
            idx += 1
    return prompts


def generate_adversarial_prompts(facts: list[PumpFact]) -> list[QAPrompt]:
    prompts: list[QAPrompt] = []
    idx = 0
    for fact in facts:
        if fact.kind != FACT_KIND_RELATION:
            continue
        templates = _ADVERSARIAL_RELATION_TEMPLATES.get(fact.predicate)
        if not templates:
            continue
        for template in templates:
            question = template.format(s=fact.subject, o=fact.obj)
            prompts.append(
                QAPrompt(
                    prompt_id=f"adv-{idx:04d}",
                    question=question,
                    category=CATEGORY_ADVERSARIAL,
                    expected_decision="audit",
                    predicate=fact.predicate,
                    subject=fact.subject,
                    obj=fact.obj,
                    expected_tokens=[],
                    notes="reversed_relation",
                )
            )
            idx += 1
    return prompts


def generate_current_prompts() -> list[QAPrompt]:
    prompts: list[QAPrompt] = []
    for idx, question in enumerate(_CURRENT_PROMPTS):
        prompts.append(
            QAPrompt(
                prompt_id=f"cur-{idx:04d}",
                question=question,
                category=CATEGORY_CURRENT,
                expected_decision="audit",
                notes="current_live_safety",
            )
        )
    return prompts


def generate_all_prompts(facts: list[PumpFact]) -> dict[str, list[QAPrompt]]:
    return {
        CATEGORY_POSITIVE: generate_positive_prompts(facts),
        CATEGORY_ADVERSARIAL: generate_adversarial_prompts(facts),
        CATEGORY_CURRENT: generate_current_prompts(),
    }
