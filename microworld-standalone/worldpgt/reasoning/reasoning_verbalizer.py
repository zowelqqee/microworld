"""Natural-language surface over explanation chains and counterfactual traces.

Mirrors ``worldpgt.cognition.verbalization_engine``: the structured artifacts
(``ExplanationChain``, ``CounterfactualTrace``) already carry every fact,
rule, and pattern involved — this module only assembles connected English
sentences out of fields that are already there. Deterministic: sentence
order follows the chain's own step order (or the trace's already-sorted
lists), so the same artifact always verbalizes to the same text. No fact is
ever added, softened, or invented; only the wording changes from a tagged
list to flowing prose.

``explanation_renderer.render`` / ``counterfactual.render`` remain the
structured, list-form output (used directly by callers that want the literal
step-by-step trace); this module is the plain-language alternative used by
the reasoning adapter for user-facing answers.
"""

from __future__ import annotations

from worldpgt.cognition.verbalization_engine import (
    _article_phrase,
    _clause,
    _join,
    confidence_word,
)
from worldpgt.reasoning.fact_graph import norm
from worldpgt.reasoning.types import CounterfactualTrace, ExplanationChain

# Inference-rule predicates read awkwardly through the generic subject/verb/
# object template (``_clause``) — these get a hand-phrased override so
# derived relations like "share_founder" read as English, not snake_case.
_RELATION_PHRASE_OVERRIDE: dict[str, str] = {
    "share_founder": "shares a founder with",
    "share_leader": "shares a leader with",
    "associated_with_expertise": "is associated with the expertise of",
    "competes_with": "competes with",
    "indirectly_requires": "indirectly requires",
}

# Past-tense form for the verbs the counterfactual question parser accepts
# (see ``reasoning_adapter._CF_VERB_PREDICATES``), so "develops" renders as
# "had not developed" rather than the ungrammatical "had not develops".
_PAST_TENSE: dict[str, str] = {
    "founded": "founded",
    "develops": "developed",
    "produces": "produced",
    "manufactures": "manufactured",
    "operates": "operated",
    "publishes": "published",
    "owns": "owned",
    "uses": "used",
    "provides": "provided",
    "leads": "led",
}


def _reasoning_clause(subject: str, predicate: str, obj: str) -> str:
    # "is_a" needs an a/an that agrees with the object ("an aerospace
    # manufacturer", not "a aerospace manufacturer") — _clause's generic
    # "is a" prefix can't see the object, so this predicate gets its own path.
    if predicate == "is_a":
        return f"{subject} is {_article_phrase(obj)}"
    override = _RELATION_PHRASE_OVERRIDE.get(predicate)
    if override:
        return f"{subject} {override} {obj}"
    return _clause(subject, predicate, obj)


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def _past(predicate: str) -> str:
    return _PAST_TENSE.get(predicate, predicate.replace("_", " "))


def _target_key(chain: ExplanationChain) -> tuple[str, str, str]:
    return (norm(chain.subject), norm(chain.predicate), norm(chain.object))


# ---------------------------------------------------------------------------
# Part 1 — explanation chains
# ---------------------------------------------------------------------------

def verbalize_explanation(chain: ExplanationChain) -> str:
    """Render an ``ExplanationChain`` as connected English prose."""
    target = _reasoning_clause(chain.subject, chain.predicate, chain.object)

    if chain.decision == "audit":
        reason = chain.audit_reason or "that isn't a fact in my knowledge base"
        return f"I can't explain why {target} — {reason}."

    target_key = _target_key(chain)
    context_facts = [
        s
        for s in chain.steps
        if s.kind == "fact"
        and (norm(s.subject), norm(s.predicate), norm(s.object)) != target_key
    ]
    pattern_steps = [s for s in chain.steps if s.kind == "pattern"]
    rule_steps = [s for s in chain.steps if s.kind == "rule"]

    if chain.fact_status == "inferred":
        because = _join([_reasoning_clause(s.subject, s.predicate, s.object) for s in context_facts])
        sentence = _cap(target)
        if because:
            sentence += f" — I infer this because {because}"
        sentence += "."
        if rule_steps:
            sentence += f" This carries {confidence_word(rule_steps[0].confidence)} confidence."
        return sentence

    sentences = [f"{_cap(target)}."]
    if context_facts:
        reason_clause = _join(
            [_reasoning_clause(s.subject, s.predicate, s.object) for s in context_facts]
        )
        sentences.append(f"This follows because {reason_clause}.")
    for pattern_step in pattern_steps:
        sentences.append(
            f"This also matches a pattern I've noticed in my graph: {pattern_step.text}."
        )
    if rule_steps:
        sentences.append("An inference rule in my graph independently reaches the same conclusion.")

    if chain.decision == "partial":
        if chain.frontier:
            sentences.append(
                f"I can trace the connection that far, but my knowledge base doesn't "
                f"fully close the loop back to {chain.subject} — the trail runs out at "
                f"{_join(chain.frontier)}."
            )
        elif chain.audit_reason:
            sentences.append(f"However, {chain.audit_reason}.")
        else:
            sentences.append(
                f"However, I can't fully close the loop back to {chain.subject}."
            )
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Part 3 — counterfactual traces
# ---------------------------------------------------------------------------

_MAX_VERBALIZED_INFERENCES = 3
_MAX_VERBALIZED_PATTERNS = 3


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or f"{singular}s")


def _pattern_sentence(pattern) -> str:
    """One plain-language sentence for a single affected pattern.

    Avoids code-ish jargon ("discovery thresholds", "pattern(s)") and the
    confusing "drop from 80% to 80%" case where confidence rounds to the same
    number even though the underlying evidence shrank — support counts make
    that visible either way.
    """
    if pattern.new_confidence is None:
        return (
            f"I'd noticed that {pattern.description}, but with that evidence gone "
            f"it's no longer a pattern I'd stand behind."
        )
    if round(pattern.new_confidence, 2) == round(pattern.old_confidence, 2):
        return (
            f"I'd also have less evidence for the pattern that {pattern.description} "
            f"({pattern.old_support} → {pattern.new_support} examples, "
            f"confidence about the same)."
        )
    return (
        f"I'd also be less confident that {pattern.description} "
        f"({pattern.old_confidence:.0%} → {pattern.new_confidence:.0%} confidence, "
        f"{pattern.old_support} → {pattern.new_support} examples)."
    )


def verbalize_counterfactual(trace: CounterfactualTrace) -> str:
    """Render a ``CounterfactualTrace`` as connected, plain-language English."""
    if trace.decision == "audit":
        reason = trace.audit_reason or "that fact isn't in my knowledge base"
        return f"I can't analyze that counterfactual — {reason}."

    if trace.target_predicate and trace.target_object:
        hypothesis = (
            f"If {trace.target_subject} had not "
            f"{_past(trace.target_predicate)} {trace.target_object}"
        )
    else:
        hypothesis = f"If {trace.target_subject} did not exist"

    paragraphs = [f"{hypothesis}, here's what would change in what I currently believe."]

    if trace.lost_inferences:
        shown = trace.lost_inferences[:_MAX_VERBALIZED_INFERENCES]
        remaining = len(trace.lost_inferences) - len(shown)
        lines = [f"I'd no longer be able to say that {_reasoning_clause(f.subject, f.predicate, f.object)}." for f in shown]
        if remaining > 0:
            lines.append(
                f"That's on top of {remaining} other {_plural(remaining, 'conclusion')} "
                f"that would stop holding too."
            )
        paragraphs.append(" ".join(lines))
    else:
        paragraphs.append("No inferred facts in my graph depend on it.")

    if trace.affected_patterns:
        shown = trace.affected_patterns[:_MAX_VERBALIZED_PATTERNS]
        remaining = len(trace.affected_patterns) - len(shown)
        lines = [_pattern_sentence(p) for p in shown]
        if remaining > 0:
            lines.append(
                f"{remaining} more {_plural(remaining, 'pattern')} would weaken as well."
            )
        paragraphs.append(" ".join(lines))

    return "\n\n".join(paragraphs)
