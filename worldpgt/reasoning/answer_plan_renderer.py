"""Surface realizer for :class:`~worldpgt.reasoning.answer_behavior.AnswerPlan`.

The renderer receives a *finished* plan and only turns it into connected text.
All content decisions — which claims appear, in what order, whether to expand
or stop — were already made by the answer-behavior layer; nothing here queries
memory, reorders blocks, or adds a claim.  Every sentence realizes exactly one
content block, so answer text stays traceable block-by-block to graph edges.

Deterministic. No ML. No network.
"""

from __future__ import annotations

import re

from worldpgt.reasoning.answer_behavior import AnswerPlan, ContentBlock
from worldpgt.reasoning.answer_behavior import _tokens as _behavior_tokens

# Discourse connectives keyed by *structural* block kind (never by predicate,
# entity, or question shape).  Cycled per kind so repeats stay readable.
_CONNECTIVES: dict[str, tuple[str, ...]] = {
    "direct_claim": ("",),
    "explanatory_continuation": ("In turn,", "From there,", "Following that link,"),
    "sibling_elaboration": ("Alongside that,", "On another supported side,"),
}


def render_answer_plan(plan: AnswerPlan) -> str:
    sentences: list[str] = []
    kind_counts: dict[str, int] = {}
    quoted_spans: set[str] = set()
    for block in plan.blocks:
        occurrence = kind_counts.get(block.kind, 0)
        kind_counts[block.kind] = occurrence + 1
        sentence = _sentence_for(block, occurrence, quoted_spans)
        if sentence:
            sentences.append(sentence)
    if not sentences:
        return ""
    provenance = _provenance_note(plan)
    if provenance:
        sentences.append(provenance)
    return " ".join(sentences)


def _sentence_for(block: ContentBlock, occurrence: int, quoted_spans: set[str]) -> str:
    edge = block.step.edge
    clause = _grounded_clause(edge, quoted_spans)
    if block.kind == "uncertainty_note":
        alternative_objects = ", ".join(
            f"'{_trim(alternative.object)}'" for alternative in block.alternatives
        )
        return _capitalize(
            f"the evidence diverges on {_trim(edge.subject)}: one cited span "
            f"supports '{_trim(edge.object)}', while other cited evidence "
            f"instead points to {alternative_objects}; neither reading is "
            "treated as settled."
        )
    connectives = _CONNECTIVES.get(block.kind, ("",))
    connective = connectives[occurrence % len(connectives)]
    if connective:
        # Subject casing is kept as the graph stores it (it may be a proper
        # name); only the connective governs sentence-initial capitalization.
        return f"{connective} {clause}"
    return clause


# An evidence span is preferred over a reconstructed triple clause when it is a
# compact sentence that actually carries the edge's content — the span *is*
# the grounded phrasing, so quoting it maximizes traceability.
_MAX_QUOTED_SPAN_CHARS = 240
_MIN_SPAN_CONTENT_COVERAGE = 0.6


def _grounded_clause(edge, quoted_spans: set[str]) -> str:
    span = _strip_attribution_preamble(_trim(edge.evidence_text), edge.subject)
    if span and len(span) <= _MAX_QUOTED_SPAN_CHARS:
        span_norm = " ".join(span.casefold().split())
        subject_norm = " ".join(str(edge.subject or "").casefold().split())
        content = _behavior_tokens(f"{edge.subject} {edge.predicate} {edge.object}")
        covered = _behavior_tokens(span)
        subject_present = bool(subject_norm) and subject_norm in span_norm
        if (
            subject_present
            # Two blocks may cite the same source sentence (a relation and a
            # descriptor extracted from one span); quote it once, then fall
            # back to the reconstructed clause so the text never repeats.
            and span_norm not in quoted_spans
            and content
            and len(content & covered) / len(content) >= _MIN_SPAN_CONTENT_COVERAGE
        ):
            quoted_spans.add(span_norm)
            return _capitalize(f"{span}.")
    return _clause(edge.subject, edge.predicate, edge.object)


# Author-authored framing that precedes the actual claim in a paper abstract
# ("This paper introduces X …", "We present X …"). Surface-only: the raw span
# is kept verbatim on the plan/trace for provenance; only the rendered prose
# drops the preamble so the sentence leads with the subject. Stripping fires
# only when the subject still appears in what remains, so content is never
# distorted.
_ATTRIBUTION_PREAMBLE_RE = re.compile(
    r"^(?:"
    r"in\s+this\s+(?:paper|work|study|article|letter)\s*,?\s+(?:we\s+\w+|the\s+\w+\s+\w+s)"
    r"|this\s+(?:paper|work|study|article|letter)\s+\w+s"
    r"|here\s*,?\s+we\s+\w+"
    r"|we\s+(?:present|propose|introduce|describe|develop|consider|report|show|study)"
    r")\s+",
    re.IGNORECASE,
)


def _strip_attribution_preamble(span: str, subject: str) -> str:
    if not span:
        return span
    stripped = _ATTRIBUTION_PREAMBLE_RE.sub("", span, count=1)
    if stripped == span:
        return span
    subject_norm = " ".join(str(subject or "").casefold().split())
    stripped = stripped.lstrip()
    if subject_norm and subject_norm not in " ".join(stripped.casefold().split()):
        return span  # preamble removal would drop the subject — keep original
    return _capitalize(_appositive_to_copula(stripped, subject))


# After a preamble is removed the span may lead with an appositive fragment
# ("TRIAGE, a two-pronged approach that …"). Restoring the elided copula
# ("TRIAGE is a …") turns it back into a well-formed, meaning-preserving
# sentence. Only fires for the exact "SUBJECT, a/an/the/one …" shape.
def _appositive_to_copula(span: str, subject: str) -> str:
    subject_text = _trim(subject)
    if not subject_text:
        return span
    prefix = f"{subject_text}, "
    if not span.lower().startswith(prefix.lower()):
        return span
    remainder = span[len(prefix):]
    if remainder.split(" ", 1)[0].lower() in {"a", "an", "the", "one"}:
        return f"{span[:len(subject_text)]} is {remainder}"
    return span


def _provenance_note(plan: AnswerPlan) -> str:
    single_source = sum(
        1 for block in plan.blocks if "single_source" in block.cautions
    )
    if not single_source:
        return ""
    total = len(plan.blocks)
    step_word = "step" if total == 1 else "steps"
    if single_source == total:
        coverage = f"each of the {total} {step_word}" if total > 1 else "this step"
    else:
        coverage = f"{single_source} of the {total} {step_word}"
    return (
        f"(Every statement above is tied to a cited evidence span; "
        f"{coverage} rests on a single source and remains "
        "proposal-level knowledge, not verified memory.)"
    )


def _clause(subject: str, predicate: str, obj: str) -> str:
    return _capitalize(_fix_articles(
        f"{_trim(subject)} {_predicate_text(predicate)} {_trim(obj)}."
    ))


# Words that start with a vowel letter but a consonant sound, so "a" stays
# (mirrors semantic_speech_planner._article_for).
_A_BEFORE_VOWEL_EXCEPTIONS = ("uni", "use", "user", "euro", "eu", "one")
_ARTICLE_RE = re.compile(r"\ba (\w+)")


def _fix_articles(text: str) -> str:
    """Surface grammar only: "a agentic framework" -> "an agentic framework"."""

    def _swap(match):
        word = match.group(1)
        if word[:1].lower() in "aeiou" and not word.lower().startswith(
            _A_BEFORE_VOWEL_EXCEPTIONS
        ):
            return f"an {word}"
        return match.group(0)

    return _ARTICLE_RE.sub(_swap, text)


def _predicate_text(predicate: str) -> str:
    return " ".join(str(predicate or "").replace("_", " ").split())


def _trim(text: str) -> str:
    return " ".join(str(text or "").split()).rstrip(".,;:")


def _capitalize(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    return text[0].upper() + text[1:]


