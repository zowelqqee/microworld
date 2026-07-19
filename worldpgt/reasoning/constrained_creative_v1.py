"""Isolated constrained-creative experiment — grounded generation + a post-hoc
constraint gate.

The mode: "write a short piece using ONLY these N facts about subject X" — combine
controlled generation with explicit grounding, unlike (a) pure factual QA (no
creative freedom) and (b) the existing Creative mode (no grounding constraint).

The measurable claim is narrow (see ``artifacts/constrained_creative_v1/design.md``):
constraint adherence — fact inclusion, fact fidelity, non-hallucination — NOT
"who writes better prose". Fluency is a labelled proxy, expected to favour an LLM.

The heart of this module is the **post-hoc constraint verifier**: it scores any
text — ours or an LLM's — against the same constraint spec with the same
normalization, so an A/B comparison is symmetric. Our v1 generator is deliberately
conservative (template + connectives), which makes inclusion/fidelity high and
hallucination ~0 by construction; the interesting number is what the *same*
verifier reports for free LLM generation.

No API/server imports: callers pass the evidence slice.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

_WORD_RE = re.compile(r"[^\W_]+(?:-[^\W_]+)?", re.UNICODE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Function/closed-class words that are never counted as "extra content" when
# looking for hallucinated material, and never counted as fact tokens.
_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "was", "are", "were", "be", "been", "being",
    "that", "which", "who", "it", "its", "this", "these", "those", "also", "known",
    "based", "one", "might", "reason", "using", "only", "these", "facts", "about",
    "while", "both", "into", "over", "within", "such", "more", "most", "some",
    "has", "have", "had", "not", "no", "their", "they", "he", "she", "his", "her",
    "produces", "develops", "founded", "leads", "located", "owned", "part",
    # discourse connectives used by the generator's openers — closed-class markers,
    # not factual content, so they must not read as hallucination.
    "addition", "beyond", "furthermore", "true", "moreover", "additionally",
})

_PREDICATE_PHRASE = {
    "founded": "founded",
    "founded_by": "was founded by",
    "leader_of": "leads",
    "known_for": "is known for",
    "developed_by": "was developed by",
    "develops": "develops",
    "produces": "produces",
    "located_in": "is located in",
    "owned_by": "is owned by",
    "part_of": "is part of",
    "created_by": "was created by",
    "published_by": "was published by",
    "uses": "uses",
    "runs_on": "runs on",
}


def _norm(value: str) -> str:
    return " ".join((value or "").casefold().split())


def _content_tokens(text: str) -> list[str]:
    return [t for t in _WORD_RE.findall((text or "").casefold())
            if t not in _STOPWORDS and len(t) >= 3]


def _lex(predicate: str) -> str:
    return _PREDICATE_PHRASE.get(_norm(predicate), (predicate or "").replace("_", " "))


# --------------------------------------------------------------------------- #
# Constraint spec + fact selection
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Fact:
    predicate: str
    object: str


@dataclass(frozen=True)
class ConstraintSpec:
    subject: str
    facts: tuple[Fact, ...]

    @property
    def n(self) -> int:
        return len(self.facts)


def select_facts(overlay_items: Iterable[dict], subject: str, n: int = 3) -> ConstraintSpec:
    """Pull up to N accepted facts about ``subject`` from an overlay slice.

    Deterministic: preserves overlay order, dedupes on (predicate, object).
    """
    sub_norm = _norm(subject)
    facts: list[Fact] = []
    seen: set[tuple[str, str]] = set()
    for item in overlay_items or ():
        if not isinstance(item, dict) or item.get("overlay_type") != "overlay_relation":
            continue
        if _norm(str(item.get("subject") or "")) != sub_norm:
            continue
        predicate = str(item.get("predicate") or "").strip()
        obj = str(item.get("object") or "").strip()
        if not (predicate and obj):
            continue
        key = (_norm(predicate), _norm(obj))
        if key in seen:
            continue
        seen.add(key)
        facts.append(Fact(predicate, obj))
        if len(facts) >= n:
            break
    return ConstraintSpec(subject=subject, facts=tuple(facts))


# --------------------------------------------------------------------------- #
# Constrained generator (v1: template + connectives, not free recombination)
# --------------------------------------------------------------------------- #

_OPENERS = ("", "In addition, ", "Beyond that, ", "It is also true that ",
            "Furthermore, ")


def generate_constrained(spec: ConstraintSpec) -> str:
    """Realize exactly the spec's facts as connected prose. By construction it
    includes every fact and asserts nothing else — the creative freedom is only in
    connective choice and ordering, which v1 keeps deterministic."""
    if not spec.facts:
        return ""
    clauses: list[str] = []
    for i, fact in enumerate(spec.facts):
        opener = _OPENERS[i % len(_OPENERS)]
        subject_ref = spec.subject if i == 0 else _pronoun_or_subject(spec.subject, i)
        clause = f"{opener}{subject_ref} {_lex(fact.predicate)} {fact.object}."
        clauses.append(clause[0].upper() + clause[1:])
    return " ".join(clauses)


def _pronoun_or_subject(subject: str, i: int) -> str:
    # v1 keeps the explicit subject on alternating clauses for clarity; it never
    # introduces an entity not in the spec. Lowercase pronoun: it always follows a
    # discourse connective here, so it is mid-clause, not sentence-initial.
    return subject if i % 2 == 0 else "it"


# --------------------------------------------------------------------------- #
# Post-hoc constraint verifier — the shared measurement instrument
# --------------------------------------------------------------------------- #

@dataclass
class VerificationReport:
    n_required: int
    n_included: int
    n_faithful: int
    included: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    unfaithful: list[str] = field(default_factory=list)
    extra_content_tokens: list[str] = field(default_factory=list)

    @property
    def inclusion_rate(self) -> float:
        return round(self.n_included / self.n_required, 3) if self.n_required else 0.0

    @property
    def fidelity_rate(self) -> float:
        return round(self.n_faithful / self.n_required, 3) if self.n_required else 0.0

    @property
    def hallucination_token_rate(self) -> float:
        """Extra content tokens as a share of all content tokens. A PROXY for the
        QA unsupported-claim rate, moved post-hoc onto generated text — imperfect
        (surface-token based), so reported as a proxy, not ground truth."""
        total = self._total_content
        return round(len(self.extra_content_tokens) / total, 3) if total else 0.0

    _total_content: int = 0

    def to_dict(self) -> dict:
        return {
            "n_required": self.n_required,
            "n_included": self.n_included,
            "n_faithful": self.n_faithful,
            "inclusion_rate": self.inclusion_rate,
            "fidelity_rate": self.fidelity_rate,
            "hallucination_token_rate": self.hallucination_token_rate,
            "included": self.included,
            "missing": self.missing,
            "unfaithful": self.unfaithful,
            "extra_content_tokens": self.extra_content_tokens,
        }


def verify(text: str, spec: ConstraintSpec) -> VerificationReport:
    """Score ``text`` against ``spec``. Applied identically to our output and to an
    LLM's — this symmetry is the point."""
    text_norm = _norm(text)
    sentences = [_norm(s) for s in _SENT_SPLIT_RE.split(text or "") if s.strip()]
    sub_norm = _norm(spec.subject)

    included: list[str] = []
    missing: list[str] = []
    unfaithful: list[str] = []
    n_faithful = 0

    for fact in spec.facts:
        obj_norm = _norm(fact.object)
        label = f"{spec.subject} {fact.predicate} {fact.object}"
        if obj_norm and obj_norm in text_norm:
            included.append(label)
            # Fidelity: the object appears in a sentence that also carries the
            # subject (or the subject pronoun surrogate) — a right-attachment proxy.
            attached = any(
                obj_norm in s and (sub_norm in s or " it " in f" {s} " or s.startswith("it "))
                for s in sentences
            )
            if attached:
                n_faithful += 1
            else:
                unfaithful.append(label)
        else:
            missing.append(label)

    # Hallucination proxy: content tokens in the text that belong to neither the
    # subject, any required object, nor any predicate lexicalization.
    allowed: set[str] = set()
    allowed.update(_content_tokens(spec.subject))
    for fact in spec.facts:
        allowed.update(_content_tokens(fact.object))
        allowed.update(_content_tokens(_lex(fact.predicate)))
    text_content = _content_tokens(text)
    extra = sorted({t for t in text_content if t not in allowed})

    report = VerificationReport(
        n_required=spec.n,
        n_included=len(included),
        n_faithful=n_faithful,
        included=included,
        missing=missing,
        unfaithful=unfaithful,
        extra_content_tokens=extra,
    )
    report._total_content = len(text_content)
    return report


# --------------------------------------------------------------------------- #
# Proxy fluency (labelled proxy, not human-validated quality)
# --------------------------------------------------------------------------- #

def proxy_fluency(text: str, attested_trigrams: set[tuple[str, str, str]]) -> float | None:
    """Fraction of 3-word windows attested in a corpus trigram set (poetry_lab's
    3-word-window grammaticality spirit). Returns None if no windows / no corpus.
    Expected to be LOW for template output over graph vocabulary — that is the
    designed trade-off (constraint adherence up, proxy fluency down), not a bug."""
    if not attested_trigrams:
        return None
    toks = _WORD_RE.findall((text or "").casefold())
    windows = list(zip(toks, toks[1:], toks[2:]))
    if not windows:
        return None
    hit = sum(1 for w in windows if w in attested_trigrams)
    return round(hit / len(windows), 3)
