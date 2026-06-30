"""Verbal reasoning — turn reasoning artifacts into plain-language narration.

The cognition stack produces several *typed* reasoning artifacts:

  - :class:`ReasoningTrace`      — bounded cognition over a speech plan.
  - :class:`ThoughtLoopTrace`    — an iterative hypothesis/gap loop.
  - :class:`InferredFact`        — a fact derived through an inference rule chain.
  - :class:`SynthesisAnswer`     — every fact known about one entity, tiered.

A :class:`MultihopResult` (from ``worldpgt.multihop_qa``) is also accepted as a
duck-typed multi-hop chain.

This module is the *language surface* over those artifacts: instead of dumping a
technical log, it explains — in connected English — what was found and how the
system reached its conclusion. It is deterministic and offline: no ML, no
network, no templates pulled from anywhere external. Every sentence is backed by
a field on the artifact it is handed; this layer never invents a fact.

The public entry point mirroring the task spec is :func:`verbalize_reasoning`.
Sibling helpers (:func:`verbalize_multihop`, :func:`verbalize_inferred_fact`,
:func:`verbalize_audit`, :func:`verbalize_synthesis`) narrate the other artifact
kinds, since each is a distinct typed input rather than a ``ReasoningTrace``.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from worldpgt.cognition.inference_engine import InferredFact
from worldpgt.cognition.types import ReasoningTrace, ThoughtLoopTrace

# ---------------------------------------------------------------------------- #
# Predicate phrasing
# ---------------------------------------------------------------------------- #
# Map overlay predicates to a natural clause. Two phrasings per predicate:
#   "subject" form keeps the explicit subject ("Starlink is owned by SpaceX").
#   "pronoun" form assumes the entity is already the topic ("It is owned by …").

_SUBJECT_PHRASE: dict[str, str] = {
    "founded_by": "was founded by",
    "owned_by": "is owned by",
    "operated_by": "is operated by",
    "created_by": "was created by",
    "designed_by": "was designed by",
    "developed_by": "was developed by",
    "manufactured_by": "is manufactured by",
    "led_by": "is led by",
    "headquartered_in": "is headquartered in",
    "located_in": "is located in",
    "part_of": "is part of",
    "subsidiary_of": "is a subsidiary of",
    "develops": "develops",
    "produces": "produces",
    "manufactures": "manufactures",
    "operates": "operates",
    "provides": "provides",
    "publishes": "publishes",
    "owns": "owns",
    "founded": "founded",
    "is_a": "is a",
    "known_for": "is known for",
    "used_for": "is used for",
    "enables": "enables",
    "uses": "uses",
}

# Phrasing when the entity is the relation *object* (inverse direction): the
# clause is spoken from the entity's perspective, so the predicate is inverted.
_INVERSE_PHRASE: dict[str, str] = {
    "founded": "was founded by",
    "founded_by": "founded",
    "owned_by": "owns",
    "owns": "is owned by",
    "operated_by": "operates",
    "operates": "is operated by",
    "developed_by": "develops",
    "develops": "is developed by",
    "manufactured_by": "manufactures",
    "produces": "is produced by",
    "uses": "is used by",
    "enables": "is enabled by",
    "provides": "is provided by",
    "part_of": "includes",
    "subsidiary_of": "owns",
}


def _phrase(predicate: str) -> str:
    """A readable verb phrase for *predicate* (subject-form)."""
    pred = (predicate or "").strip()
    if pred in _SUBJECT_PHRASE:
        return _SUBJECT_PHRASE[pred]
    return pred.replace("_", " ") or "is related to"


def _inverse_phrase(predicate: str) -> str:
    """A readable verb phrase when the entity is the relation *object*."""
    pred = (predicate or "").strip()
    if pred in _INVERSE_PHRASE:
        return _INVERSE_PHRASE[pred]
    return f"is linked (via {pred.replace('_', ' ')}) to" if pred else "is linked to"


def _clause(subject: str, predicate: str, obj: str) -> str:
    """One readable clause, e.g. ``Starlink is owned by SpaceX``."""
    return f"{subject.strip()} {_phrase(predicate)} {obj.strip()}".strip()


def _join(items: Sequence[str]) -> str:
    """Oxford-comma join: ``a``, ``a and b``, ``a, b, and c``."""
    items = [i for i in items if i]
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def confidence_word(value: float | str) -> str:
    """Map a confidence score or label to a plain word: high / medium / low."""
    if isinstance(value, str):
        return {
            "grounded": "high",
            "thin": "medium",
            "gap_heavy": "low",
        }.get(value, "medium")
    if value >= 0.85:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------- #
# 1. ReasoningTrace
# ---------------------------------------------------------------------------- #

def verbalize_reasoning(trace: ReasoningTrace) -> str:
    """Narrate a bounded :class:`ReasoningTrace` in connected English.

    Dispatches on the primary conclusion and the chosen action so a direct
    answer reads differently from a gap-limited one.
    """
    subject = trace.workspace.subject or trace.task.subject or "this topic"
    conclusion = trace.primary_conclusion
    action = trace.action.next_action if trace.action else "answer"

    if conclusion is None or action == "ask_clarification":
        return _verbalize_clarification(trace, subject)

    kind = conclusion.kind
    if kind == "direct_answer":
        text = _strip_period(conclusion.text) or subject
        return (
            f"I found that {text}. "
            "This is a verified historical fact from my knowledge base."
        )
    if kind == "profile_summary":
        body = conclusion.text.strip()
        return f"Here's what I gathered about {subject}: {body}".strip()
    if kind in ("mechanism_gap", "thin_profile") or action == "answer_with_gap":
        return _verbalize_reasoning_gap(trace, subject, conclusion)
    # Fallback: speak the conclusion plainly.
    return _strip_period(conclusion.text) + "." if conclusion.text else (
        f"I considered {subject} but reached no confident conclusion."
    )


def _verbalize_reasoning_gap(trace: ReasoningTrace, subject: str, conclusion) -> str:
    known = conclusion.text.strip()
    missing = [m.reason for m in trace.missing_evidence if m.reason]
    parts: list[str] = []
    if known:
        parts.append(known if known.endswith(".") else known + ".")
    if missing:
        parts.append(
            "However, I don't have verified information about "
            f"{_join(missing)}."
        )
    else:
        parts.append(
            f"However, I don't have enough verified detail to fully answer "
            f"about {subject}."
        )
    questions = [q for m in trace.missing_evidence for q in m.next_questions]
    if questions:
        parts.append(
            "To answer this properly, I would need facts about: "
            f"{_join(questions)}."
        )
    return "\n".join(parts)


def _verbalize_clarification(trace: ReasoningTrace, subject: str) -> str:
    questions = [
        q for m in trace.missing_evidence for q in m.next_questions
    ] or list(trace.action.next_questions) if trace.action else []
    if questions:
        return (
            f"I need a little more to go on about {subject}. "
            f"Could you clarify: {_join(questions)}?"
        )
    return f"I need more detail before I can reason about {subject}."


# ---------------------------------------------------------------------------- #
# 2. ThoughtLoopTrace
# ---------------------------------------------------------------------------- #

def verbalize_thought_loop(loop: ThoughtLoopTrace) -> str:
    """Narrate an iterative :class:`ThoughtLoopTrace`."""
    subject = loop.subject or "this topic"
    if loop.accepted_hypothesis is not None:
        return (
            f"I worked through {subject} step by step and settled on: "
            f"{_strip_period(loop.accepted_hypothesis.text)}. "
            f"I stopped because I had a supported conclusion."
        )
    if loop.stop_reason == "blocked_missing_evidence":
        missing = [q for q in loop.next_questions]
        tail = (
            f" I would still need: {_join(missing)}." if missing else ""
        )
        return (
            f"I worked through {subject} but couldn't confirm a conclusion — "
            f"the evidence I needed wasn't available.{tail}"
        )
    return (
        f"I explored {subject} for several steps without reaching a "
        f"confident conclusion."
    )


# ---------------------------------------------------------------------------- #
# 3. Multi-hop chain
# ---------------------------------------------------------------------------- #

def verbalize_multihop(result) -> str:
    """Narrate a 2-hop chain from a ``MultihopResult``-shaped object.

    Reads ``hop1_detail`` / ``hop2_detail`` (dicts with subject/predicate/object)
    when present, otherwise falls back to the ``hop1`` / ``hop2`` display strings
    (``"subject | predicate | object"``).
    """
    hop1 = hop_triple(getattr(result, "hop1_detail", None) or getattr(result, "hop1", None))
    hop2 = hop_triple(getattr(result, "hop2_detail", None) or getattr(result, "hop2", None))

    if hop1 is None or hop2 is None:
        # No usable chain — defer to whatever answer text exists.
        return (getattr(result, "answer_text", "") or "").strip()

    (s1, _p1, o1) = hop1
    (s2, _p2, o2) = hop2
    via = o1 if o1 == s2 else s2
    start, end = s1, o2

    lines = [
        "I reasoned through two steps: "
        f"{_clause(*hop1)}, and {_clause(*hop2)}."
    ]
    if via and start and end:
        lines.append(
            f"Therefore {start} is connected to {end} through {via}."
        )
    return " ".join(lines)


def hop_triple(hop) -> tuple[str, str, str] | None:
    """Normalize a hop (dict or ``"subj | pred | obj"`` string) to a triple."""
    if isinstance(hop, dict):
        s = str(hop.get("subject") or "").strip()
        p = str(hop.get("predicate") or "").strip()
        o = str(hop.get("object") or "").strip()
        if s and p and o:
            return (s, p, o)
        return None
    if hop:
        parts = [part.strip() for part in str(hop).split("|")]
        if len(parts) == 3 and all(parts):
            return (parts[0], parts[1], parts[2])
    return None


def hop_sentence(hop) -> str:
    """A readable clause for one hop, e.g. ``Starlink is owned by SpaceX``."""
    triple = hop_triple(hop)
    return _clause(*triple) if triple else ""


# ---------------------------------------------------------------------------- #
# 4. Inferred fact
# ---------------------------------------------------------------------------- #

def verbalize_inferred_fact(fact: InferredFact) -> str:
    """Narrate an :class:`InferredFact`, spelling out its proof chain."""
    headline = _clause(fact.subject, fact.predicate, fact.object)
    because = [_clause(s, p, o) for (s, p, o) in fact.chain]
    word = confidence_word(fact.confidence)

    parts = [f"I inferred that {headline} — not as a direct fact"]
    if because:
        parts.append(f", but because {_join(because)}")
    parts.append(".")
    sentence = "".join(parts)
    return f"{sentence} This conclusion carries {word} confidence."


# ---------------------------------------------------------------------------- #
# 5. Audit / gap
# ---------------------------------------------------------------------------- #

def verbalize_audit(
    question: str,
    *,
    reason: str | None = None,
    missing_topics: Iterable[str] = (),
    known: str | None = None,
    needs: Iterable[str] = (),
) -> str:
    """Narrate an honest audit: what was looked for and what is missing.

    ``known``  — anything the system *does* know about the subject (optional).
    ``needs``  — concrete facts that would be required to answer.
    """
    topic = _looked_for_phrase(question)
    parts: list[str] = []
    if known:
        parts.append(known if known.endswith(".") else known + ".")
        parts.append(
            f"However, I don't have verified information about {topic}."
        )
    else:
        lead = f"I looked for information about {topic}"
        if reason:
            parts.append(f"{lead}, but {reason}.")
        else:
            parts.append(
                f"{lead}, but I don't have a verified answer for it."
            )
        parts.append("I can't answer this reliably.")
    needs = [n for n in needs if n]
    if needs:
        parts.append(
            f"To answer this properly, I would need facts about: {_join(needs)}."
        )
    return " ".join(parts)


def _looked_for_phrase(question: str) -> str:
    q = (question or "").strip().rstrip("?.! ")
    if not q:
        return "this question"
    low = q.lower()
    for prefix in (
        "how does ", "how do ", "how is ", "how are ", "what is ",
        "what are ", "what does ", "who is ", "who are ", "who ", "when ",
        "where ", "why ", "which ",
    ):
        if low.startswith(prefix):
            return q[len(prefix):].strip() or "this question"
    return q


# ---------------------------------------------------------------------------- #
# 6. Synthesis (tiered)
# ---------------------------------------------------------------------------- #

def verbalize_synthesis(answer, inferred_facts: Sequence[InferredFact] = ()) -> str:
    """Narrate a :class:`SynthesisAnswer` as a tiered, tagged bullet list.

    Each line is tagged ``[verified]``, ``[snapshot — may be outdated]`` or
    ``[inferred]`` so the reader can see exactly how trustworthy each claim is.
    Optional *inferred_facts* are appended as the INFERRED tier.
    """
    if not getattr(answer, "matched", False):
        subject = getattr(answer, "subject", None) or "that"
        return f"I don't have anything verified about {subject} in my knowledge base."

    subject = answer.subject or "this"
    reference = _synthesis_reference(getattr(answer, "entity_type", None))
    possessive = _synthesis_possessive(getattr(answer, "entity_type", None))
    lines: list[str] = [f"Here's what I know about {subject}:"]

    if answer.definition:
        lines.append(
            f"- {reference} {_copula(reference)} {_article_phrase(answer.definition)}. [verified]"
        )

    for group in answer.groups:
        if group.tier == "SNAPSHOT":
            lines.append("- " + _snapshot_line(group, possessive=possessive))
        elif group.tier == "INFERRED":
            clause = _group_clause(group, reference=reference)
            if clause:
                confidence = getattr(group, "confidence", None)
                conf = confidence_word(confidence) if confidence is not None else "medium"
                lines.append(f"- {clause} [inferred — {conf} confidence]")
        else:
            clause = _group_clause(group, reference=reference)
            if clause:
                lines.append(f"- {clause} [verified]")

    for fact in inferred_facts:
        lines.append(
            f"- {reference} {_phrase_for_reference(reference, fact.predicate)} {fact.object}. "
            f"[inferred — {confidence_word(fact.confidence)} confidence]"
        )

    return "\n".join(lines)


def _group_clause(group, *, reference: str = "It") -> str:
    objs = _join(list(group.objects))
    if not objs:
        return ""
    if group.kind == "inverse_relation":
        # The entity is the relation object — speak it from the entity's side.
        return f"{reference} {_inverse_phrase_for_reference(reference, group.predicate)} {objs}."
    return f"{reference} {_phrase_for_reference(reference, group.predicate)} {objs}."


def _snapshot_line(group, *, possessive: str = "Its") -> str:
    pred = (group.predicate or "").replace("_", " ").strip()
    obj = _join(list(group.objects))
    sentence = f"{possessive} {pred} was {obj}"
    if group.as_of:
        sentence += f" as of {group.as_of}"
    if group.source_name:
        sentence += f", according to {group.source_name}"
    return sentence + ". [snapshot — may be outdated]"


# ---------------------------------------------------------------------------- #
# Shared helpers
# ---------------------------------------------------------------------------- #

def _strip_period(text: str) -> str:
    return (text or "").strip().rstrip(".").strip()


def _synthesis_reference(entity_type: str | None) -> str:
    return "They" if entity_type == "person" else "It"


def _synthesis_possessive(entity_type: str | None) -> str:
    return "Their" if entity_type == "person" else "Its"


def _copula(reference: str) -> str:
    return "are" if reference == "They" else "is"


def _article_phrase(text: str) -> str:
    stripped = _strip_period(text)
    if not stripped:
        return stripped
    lower = stripped.lower()
    if lower.startswith(("a ", "an ", "the ", "one of ", "part of ")):
        return stripped
    article = "an" if lower[:1] in "aeiou" else "a"
    return f"{article} {stripped}"


def _phrase_for_reference(reference: str, predicate: str) -> str:
    phrase = _phrase(predicate)
    return _agree_phrase(reference, phrase)


def _inverse_phrase_for_reference(reference: str, predicate: str) -> str:
    phrase = _inverse_phrase(predicate)
    return _agree_phrase(reference, phrase)


def _agree_phrase(reference: str, phrase: str) -> str:
    if reference != "They":
        return phrase
    if phrase.startswith("is "):
        return "are " + phrase.removeprefix("is ")
    if phrase.startswith("was "):
        return "were " + phrase.removeprefix("was ")
    return phrase
