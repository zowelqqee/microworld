"""Symbolic text generation over verified speech plans.

This module is intentionally not a fact source. It receives a ``SpeechPlan``
that was already built from supported memory and generates the next allowed
speech unit from explicit state: question style, answer style, previous units,
and the remaining fact buckets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from hashlib import sha256
from typing import Mapping

from worldpgt.cognition.decision_surface import (
    clarification_candidates,
    decision_intro_candidates,
    decision_intro_sentence,
    decision_next_question_candidates,
    decision_next_question_sentence,
)
from worldpgt.cognition.types import ActionPlan, AnswerSelectionTrace, AnswerVariant

from worldpgt.entity_qa.semantic_speech_planner import (
    SpeechPlan,
    _brief_clause,
    _compact_objects,
    _dedupe,
    _join_clauses,
    _join_list,
    _simple_clauses,
)


_CONTENT_BUCKETS = (
    "mechanism",
    "purpose",
    "activity",
    "origin",
    "ownership",
    "classification",
    "recognition",
    "other",
)

_BUCKET_PRIORS: Mapping[str, tuple[str, ...]] = {
    "how": (
        "mechanism",
        "purpose",
        "classification",
        "ownership",
        "activity",
        "origin",
        "recognition",
        "other",
    ),
    "knowledge": (
        "activity",
        "origin",
        "classification",
        "ownership",
        "recognition",
        "purpose",
        "mechanism",
        "other",
    ),
    "overview": (
        "activity",
        "origin",
        "ownership",
        "classification",
        "recognition",
        "purpose",
        "mechanism",
        "other",
    ),
}


@dataclass(frozen=True)
class SpeechUnit:
    kind: str
    text: str


@dataclass(frozen=True)
class GenerationState:
    used_kinds: frozenset[str] = frozenset()
    emitted_units: int = 0


def generate_text(plan: SpeechPlan, action: ActionPlan | None = None) -> str:
    """Generate answer text from a speech plan without adding facts."""

    return generate_text_with_selection_trace(plan, action=action).final_text


def generate_text_with_selection_trace(
    plan: SpeechPlan,
    action: ActionPlan | None = None,
) -> AnswerSelectionTrace:
    """Generate answer text and expose the surface-variant selection trace."""

    from worldpgt.cognition.mini_reasoner import build_mini_thought
    from worldpgt.cognition.surface_selection import select_answer_variant

    if action is not None and action.next_action == "ask_clarification":
        questions = tuple(_clean_sentence(text) for text in clarification_candidates(plan, action))
        return select_answer_variant(
            tuple(
                AnswerVariant(name=name, text=text)
                for name, text in zip(
                    ("clarification", "clarification_direct", "clarification_minimal"),
                    questions,
                    strict=False,
                )
                if text
            ),
            plan,
        )

    units: list[SpeechUnit] = []
    if plan.intro:
        units.append(SpeechUnit("intro", _clean_sentence(plan.intro)))

    thought = build_mini_thought(plan)
    speech_action = action or thought.action
    decision_sentence = decision_intro_sentence(
        plan,
        speech_action,
        thought.trace.primary_conclusion if thought.trace is not None else None,
        thought.confidence,
    )
    decision_candidates = decision_intro_candidates(
        plan,
        speech_action,
        thought.trace.primary_conclusion if thought.trace is not None else None,
        thought.confidence,
    )
    conclusion = thought.trace.primary_conclusion if thought.trace is not None else None
    needs_gap_notice = (
        speech_action.next_action == "answer_with_gap"
        or thought.confidence == "thin"
        or (conclusion is not None and conclusion.kind in {"mechanism_gap", "thin_profile"})
    )
    if decision_sentence and (plan.answer_style != "brief" or needs_gap_notice):
        units.append(SpeechUnit("mini_thought", _clean_sentence(decision_sentence)))

    state = GenerationState()
    buckets = _bucket_clauses(plan)
    max_detail_units = _max_detail_units(plan, speech_action)
    while state.emitted_units < max_detail_units:
        candidates = next_speech_unit_candidates(plan, buckets, state, speech_action)
        if not candidates:
            break
        unit = _choose_next_unit(plan, candidates, state)
        if not unit.text:
            break
        if not _restates_prior_unit(unit.text, units):
            units.append(unit)
        state = GenerationState(
            used_kinds=state.used_kinds | frozenset({unit.kind}),
            emitted_units=state.emitted_units + 1,
        )

    if plan.answer_style != "brief":
        units.extend(SpeechUnit("snapshot", _clean_sentence(s)) for s in plan.snapshots)

    if plan.gaps and plan.answer_style != "brief":
        units.append(
            SpeechUnit(
                "gap",
                "I don't have verified information about "
                f"{_join_list(plan.gaps)} yet.",
            )
        )

    if (
        speech_action.next_action == "answer_with_gap"
        and speech_action.next_questions
        and plan.answer_style not in {"brief", "followup"}
    ):
        next_question_candidates = decision_next_question_candidates(speech_action)
        units.append(
            SpeechUnit(
                "next_question",
                _clean_sentence(decision_next_question_sentence(speech_action)),
            )
        )
    else:
        next_question_candidates = ()

    return select_answer_variant(
        _answer_variants(
            units,
            plan,
            decision_candidates=decision_candidates,
            next_question_candidates=next_question_candidates,
        ),
        plan,
    )


def next_speech_unit_candidates(
    plan: SpeechPlan,
    buckets: Mapping[str, list[str]] | None = None,
    state: GenerationState | None = None,
    action: ActionPlan | None = None,
) -> list[SpeechUnit]:
    """Return the next allowed speech units from explicit generation state."""

    buckets = buckets or _bucket_clauses(plan)
    state = state or GenerationState()
    candidates: list[SpeechUnit] = []
    for kind in _ordered_remaining_kinds(plan, buckets, state, action):
        clauses = _clauses_for_answer_style(plan, kind, buckets[kind])
        text = _sentence_for_bucket(plan, kind, clauses, state)
        if text:
            candidates.append(SpeechUnit(kind, text))
    return candidates


def _bucket_clauses(plan: SpeechPlan) -> dict[str, list[str]]:
    buckets: dict[str, list[str]] = {name: [] for name in _CONTENT_BUCKETS}
    if plan.classification:
        buckets["classification"].append(
            f"is classified as {_compact_objects(plan.classification)}"
        )
    if plan.activity:
        buckets["activity"].extend(plan.activity)
    if plan.origin:
        founder_names = [x for x in plan.origin if not x.startswith("founded ")]
        founded_objects = [x for x in plan.origin if x.startswith("founded ")]
        if founder_names:
            buckets["origin"].append(f"was founded by {_join_list(founder_names)}")
        buckets["origin"].extend(founded_objects)
    if plan.ownership:
        buckets["ownership"].extend(plan.ownership)
    if plan.recognition:
        buckets["recognition"].append(f"is known for {_join_list(plan.recognition)}")
    if plan.mechanism:
        buckets["mechanism"].extend(plan.mechanism)
    if plan.purpose:
        buckets["purpose"].extend(plan.purpose)
    if plan.other:
        buckets["other"].extend(plan.other)
    return {kind: _clean_clauses(clauses) for kind, clauses in buckets.items()}


def _ordered_remaining_kinds(
    plan: SpeechPlan,
    buckets: Mapping[str, list[str]],
    state: GenerationState,
    action: ActionPlan | None = None,
) -> list[str]:
    base_order = _BUCKET_PRIORS.get(plan.style, _BUCKET_PRIORS["overview"])
    if action and action.preferred_buckets:
        preferred = tuple(kind for kind in action.preferred_buckets if kind in base_order)
        order = preferred + tuple(kind for kind in base_order if kind not in preferred)
    else:
        order = base_order
    suppressed = set(action.suppressed_buckets) if action else set()
    kinds = [
        kind
        for kind in order
        if kind not in state.used_kinds and buckets.get(kind) and kind not in suppressed
    ]
    if "other" in kinds and plan.intro:
        kinds.remove("other")
    if "other" in kinds and any(kind != "other" for kind in kinds):
        kinds.remove("other")
        if plan.answer_style in {"followup", "brief", "simple"}:
            return kinds
        kinds.append("other")
    return kinds


def _choose_next_unit(
    plan: SpeechPlan,
    candidates: list[SpeechUnit],
    state: GenerationState,
) -> SpeechUnit:
    if not candidates:
        return SpeechUnit("", "")
    if len(candidates) == 1:
        return candidates[0]

    order = _BUCKET_PRIORS.get(plan.style, _BUCKET_PRIORS["overview"])
    rank = {kind: idx for idx, kind in enumerate(order)}
    best_rank = min(rank.get(unit.kind, len(order)) for unit in candidates)
    best = [unit for unit in candidates if rank.get(unit.kind, len(order)) == best_rank]
    if len(best) == 1:
        return best[0]
    index = _stable_index(plan.seed, f"next:{state.emitted_units}", len(best))
    return best[index]


def _sentence_for_bucket(
    plan: SpeechPlan,
    kind: str,
    clauses: list[str],
    state: GenerationState,
) -> str:
    clauses = _clean_clauses(clauses)
    if not clauses:
        return ""

    if plan.answer_style == "brief":
        clauses = [_brief_clause(clauses[0])]
    elif plan.answer_style == "followup":
        clauses = clauses[:2]
    elif plan.answer_style == "important":
        clauses = clauses[:1]
    elif plan.answer_style == "simple":
        clauses = _simple_clauses(clauses)[:2]

    reference = plan.reference or plan.subject or "It"
    clause_text = _join_clauses(clauses)
    if not clause_text:
        return ""

    if state.emitted_units == 0:
        return _clean_sentence(f"{reference} {clause_text}.")

    if kind in {"activity", "purpose", "mechanism", "ownership", "recognition"}:
        return _clean_sentence(f"{reference} {_followon_clause(clause_text)}.")

    return _clean_sentence(f"{reference} {clause_text}.")


def _clauses_for_answer_style(
    plan: SpeechPlan,
    kind: str,
    clauses: list[str],
) -> list[str]:
    if plan.answer_style == "brief":
        return clauses[:1]
    if plan.answer_style == "followup":
        return clauses[:2]
    if plan.answer_style == "important":
        return clauses[:1]
    if plan.answer_style == "simple":
        return clauses[:2]
    if kind == "other":
        return clauses[:1]
    return clauses[:3]


def _max_detail_units(plan: SpeechPlan, action: ActionPlan | None = None) -> int:
    if action and action.detail_unit_limit is not None:
        return max(0, action.detail_unit_limit)
    if plan.answer_style == "brief":
        return 1
    if plan.answer_style == "followup":
        return 1
    if plan.answer_style == "important":
        return 2
    if plan.answer_style == "simple":
        return 2
    if plan.style == "how":
        return 3
    return 3


def _answer_variants(
    units: list[SpeechUnit],
    plan: SpeechPlan,
    *,
    decision_candidates: tuple[str, ...] = (),
    next_question_candidates: tuple[str, ...] = (),
) -> tuple[AnswerVariant, ...]:
    baseline = _join_units(units)
    variants = [AnswerVariant(name="baseline", text=baseline)]

    concise_units = _concise_units(units, plan)
    concise = _join_units(concise_units)
    if concise and concise != baseline:
        variants.append(AnswerVariant(name="concise", text=concise))

    for index, candidate in enumerate(decision_candidates[1:], start=1):
        sampled = _replace_unit_text(units, "mini_thought", candidate)
        text = _join_units(sampled)
        if text and text != baseline:
            variants.append(AnswerVariant(name=f"decision_sample_{index}", text=text))

    for index, candidate in enumerate(next_question_candidates[1:], start=1):
        sampled = _replace_unit_text(units, "next_question", candidate)
        text = _join_units(sampled)
        if text and text != baseline:
            variants.append(AnswerVariant(name=f"next_question_sample_{index}", text=text))

    if len(decision_candidates) > 1 and len(next_question_candidates) > 1:
        sampled = _replace_unit_text(units, "mini_thought", decision_candidates[1])
        sampled = _replace_unit_text(sampled, "next_question", next_question_candidates[1])
        text = _join_units(sampled)
        if text and text != baseline:
            variants.append(AnswerVariant(name="combined_sample", text=text))

    variants = _dedupe_variants(variants)
    return tuple(variants)


def _concise_units(units: list[SpeechUnit], plan: SpeechPlan) -> list[SpeechUnit]:
    if plan.answer_style in {"brief", "followup"}:
        return units
    keep: list[SpeechUnit] = []
    for unit in units:
        if unit.kind in {"intro", "mini_thought", "gap", "next_question"}:
            keep.append(unit)
        elif unit.kind in {"mechanism", "purpose", "activity"} and not any(
            existing.kind in {"mechanism", "purpose", "activity"}
            for existing in keep
        ):
            keep.append(unit)
    return keep or units


def _join_units(units: list[SpeechUnit]) -> str:
    return " ".join(unit.text for unit in units if unit.text).strip()


def _replace_unit_text(
    units: list[SpeechUnit],
    kind: str,
    text: str,
) -> list[SpeechUnit]:
    replacement = _clean_sentence(text)
    return [
        SpeechUnit(unit.kind, replacement)
        if unit.kind == kind and replacement
        else unit
        for unit in units
    ]


def _dedupe_variants(variants: list[AnswerVariant]) -> list[AnswerVariant]:
    seen: set[str] = set()
    out: list[AnswerVariant] = []
    for variant in variants:
        if not variant.text or variant.text in seen:
            continue
        seen.add(variant.text)
        out.append(variant)
    return out


def _followon_clause(clause: str) -> str:
    if clause.startswith("is "):
        return clause
    if clause.startswith("was "):
        return clause
    return f"also {clause}"


def _clean_clauses(clauses: list[str]) -> list[str]:
    return _dedupe([_clean_fragment(clause) for clause in clauses if clause])


def _clean_sentence(text: str) -> str:
    text = _clean_fragment(text)
    if not text:
        return ""
    return text if text.endswith((".", "?", "!")) else f"{text}."


def _clean_fragment(text: str) -> str:
    text = _trim_noisy_including(str(text))
    return " ".join(text.strip().strip(" ,;").split())


def _trim_noisy_including(text: str) -> str:
    marker = ", including "
    if marker not in text:
        return text
    before, after = text.split(marker, 1)
    if ", the " in after or ", a " in after or ", an " in after:
        return before
    return text


def _stable_index(seed: str, node: str, size: int) -> int:
    if size <= 1:
        return 0
    value = int(sha256(f"{seed}:{node}".encode("utf-8")).hexdigest()[:12], 16)
    return value % size


_WORD_RE = re.compile(r"[a-z0-9]+")
_RESTATEMENT_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "by", "for", "from", "in", "is",
        "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
        "were", "with",
    }
)


def _restates_prior_unit(text: str, prior_units: list[SpeechUnit]) -> bool:
    """True when ``text`` mostly repeats the content words of an earlier unit.

    Catches cases where a decision/intro sentence anchors on the same clause
    that the bucket loop independently emits afterward (same underlying
    evidence, different sentence frame) — without hand-listing which
    sentence pairs can collide.
    """

    words = _content_words(text)
    if len(words) < 4:
        return False
    for unit in prior_units:
        prior_words = _content_words(unit.text)
        if len(prior_words) < 4:
            continue
        smaller = min(len(words), len(prior_words))
        overlap = len(words & prior_words) / smaller
        if overlap >= 0.7:
            return True
    return False


def _content_words(text: str) -> frozenset[str]:
    return frozenset(
        word for word in _WORD_RE.findall(text.lower()) if word not in _RESTATEMENT_STOPWORDS
    )
