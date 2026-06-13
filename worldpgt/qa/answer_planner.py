"""Answer planner for AnswerPlanner v1.

Retrieves explicit evidence from builtin sense memory and the accepted
knowledge memory provider, then builds an AnswerPlan.

Rule-based and deterministic only.  No ML libraries.
No learned model parameters.  No language-model inference.  No back-prop.

SAFETY CONTRACT:
- sense_memory.py is NOT modified.
- Thresholds are NOT lowered.
- Validators are NOT weakened.
- No generic fallback is added.
- No force-answer logic.
"""

from __future__ import annotations

from typing import Optional

from worldpgt.continuation.sense_memory import ExplicitSenseMemory
from worldpgt.knowledge.accepted_memory_provider import AcceptedKnowledgeMemoryProvider
from worldpgt.qa.types import AnalyzedQuestion, AnswerPlan, EvidenceBundle

_MIN_SCORE = 1.0
_MIN_MARGIN = 1.0

# Base descriptions for each sense — used to ground define_sense answers.
_SENSE_BASE: dict[tuple[str, str], str] = {
    ("bank", "financial_institution"): "a financial institution where people manage money",
    ("bank", "river_edge"): "the edge or slope of land beside a river or stream",
    ("bat", "animal"): "a nocturnal flying mammal",
    ("bat", "sports_equipment"): "a club used to hit a ball in sports such as baseball",
    ("seal", "animal"): "a marine mammal that lives near coasts and oceans",
    ("seal", "closure_stamp"): "a device or substance used to close or authenticate an object",
    ("crane", "bird"): "a large wading bird with long legs and a long neck",
    ("crane", "machine"): "a large machine used to lift and move heavy loads at construction sites",
    ("rock", "stone"): "a solid mineral mass found in nature",
    ("rock", "music"): "a genre of music typically featuring electric guitars and strong rhythms",
    ("spring", "season"): "the season that follows winter, associated with warming and blooming",
    ("spring", "coil"): "a coiled metal device that stores and releases mechanical energy",
}

_SENSE_LABEL: dict[str, str] = {
    "financial_institution": "financial bank",
    "river_edge": "river bank",
    "animal": "animal",
    "sports_equipment": "sports bat",
    "closure_stamp": "closure seal",
    "bird": "crane bird",
    "machine": "construction crane",
    "stone": "rock stone",
    "music": "rock music",
    "season": "spring season",
    "coil": "spring coil",
}


def _audit(analyzed: AnalyzedQuestion, reason: str, confidence: float = 0.0) -> AnswerPlan:
    return AnswerPlan(
        analyzed=analyzed,
        decision="audit",
        audit_reason=reason,
        evidence=EvidenceBundle(),
        render_template="audit",
        render_args={"reason": reason, "term": analyzed.term},
        confidence=confidence,
    )


def _builtin_cues(sense_mem: ExplicitSenseMemory, term: str, sense_id: str) -> list[str]:
    for entry in sense_mem.get_senses(term):
        if entry.sense_id == sense_id:
            return list(entry.cues)
    return []


def _plan_define_sense(
    analyzed: AnalyzedQuestion,
    provider: AcceptedKnowledgeMemoryProvider,
    sense_mem: ExplicitSenseMemory,
) -> AnswerPlan:
    term = analyzed.term
    sense_id = analyzed.target_sense
    if not term or not sense_id:
        return _audit(analyzed, "missing_term_or_sense")

    provider_items = provider.items_for_sense(term, sense_id)
    base_desc = _SENSE_BASE.get((term, sense_id))
    if not base_desc:
        return _audit(analyzed, "sense_not_in_memory")

    typical_actions = [i.value for i in provider_items if i.item_type == "typical_action"]
    typical_locations = [i.value for i in provider_items if i.item_type == "typical_location"]
    provider_cues = [i.value for i in provider_items if i.item_type == "positive_cue"]
    builtin = _builtin_cues(sense_mem, term, sense_id)
    all_cues = list(dict.fromkeys(provider_cues + builtin))[:6]

    fact_ids = [i.item_id for i in provider_items if i.item_type in (
        "positive_cue", "typical_action", "typical_location")]
    pattern_ids = [i.item_id for i in provider_items if "pattern" in i.item_type]

    evidence = EvidenceBundle(
        facts_used=fact_ids,
        patterns_used=pattern_ids,
        cues_used=all_cues[:4],
        provider_items_used=[i.item_id for i in provider_items],
    )
    return AnswerPlan(
        analyzed=analyzed,
        decision="answer",
        audit_reason=None,
        evidence=evidence,
        render_template="define_sense",
        render_args={
            "term": term,
            "sense_id": sense_id,
            "base_desc": base_desc,
            "cues": all_cues[:4],
            "typical_actions": typical_actions[:3],
            "typical_locations": typical_locations[:2],
        },
        confidence=1.0,
    )


def _plan_explain_cue(
    analyzed: AnalyzedQuestion,
    provider: AcceptedKnowledgeMemoryProvider,
    sense_mem: ExplicitSenseMemory,
) -> AnswerPlan:
    term = analyzed.term
    sense_id = analyzed.target_sense
    cues_in_q = analyzed.cues_in_question
    if not cues_in_q or not term or not sense_id:
        return _audit(analyzed, "missing_cue_or_sense")

    cue = cues_in_q[0]
    builtin = _builtin_cues(sense_mem, term, sense_id)
    provider_cue_vals = {
        i.value.strip().lower()
        for i in provider.items_for_sense(term, sense_id)
        if i.item_type == "positive_cue"
    }
    in_builtin = cue in builtin
    in_provider = cue in provider_cue_vals

    if not in_builtin and not in_provider:
        return _audit(analyzed, "cue_not_found")

    provider_items = provider.items_for_sense(term, sense_id)
    typical_actions = [i.value for i in provider_items if i.item_type == "typical_action"][:3]
    related_cues = [c for c in (builtin + list(provider_cue_vals)) if c != cue][:4]
    # Deduplicate while preserving order
    seen: set[str] = set()
    related_cues_dedup: list[str] = []
    for c in related_cues:
        if c not in seen:
            seen.add(c)
            related_cues_dedup.append(c)

    evidence = EvidenceBundle(
        facts_used=[i.item_id for i in provider_items if i.item_type == "positive_cue"],
        patterns_used=[i.item_id for i in provider_items if "pattern" in i.item_type],
        cues_used=[cue] + related_cues_dedup[:3],
        provider_items_used=[i.item_id for i in provider_items],
    )
    return AnswerPlan(
        analyzed=analyzed,
        decision="answer",
        audit_reason=None,
        evidence=evidence,
        render_template="explain_cue",
        render_args={
            "cue": cue,
            "term": term,
            "sense_id": sense_id,
            "related_cues": related_cues_dedup[:3],
            "typical_actions": typical_actions,
        },
        confidence=1.0,
    )


def _plan_classify_context(
    analyzed: AnalyzedQuestion,
    provider: AcceptedKnowledgeMemoryProvider,
    sense_mem: ExplicitSenseMemory,
    overlay_mem: ExplicitSenseMemory,
) -> AnswerPlan:
    inline_ctx = analyzed.inline_context
    term = analyzed.term
    if not inline_ctx or not term:
        return _audit(analyzed, "missing_context_or_term")

    evidence_obj = overlay_mem.score_senses_with_evidence(inline_ctx, term)
    scores = evidence_obj.adjusted_scores
    if not scores or all(s == 0.0 for s in scores.values()):
        return _audit(analyzed, "no_cue_evidence_in_context")

    ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    top_sense, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = top_score - second_score
    confidence = min(1.0, top_score / (top_score + second_score + 1.0))

    if top_score < _MIN_SCORE:
        return _audit(analyzed, "score_below_threshold", confidence)
    if margin < _MIN_MARGIN:
        return _audit(analyzed, "margin_below_threshold", confidence)
    if evidence_obj.conflict_detected and margin <= _MIN_MARGIN:
        return _audit(analyzed, "conflict_detected", confidence)

    matched_cues = evidence_obj.positive_cues.get(top_sense, [])

    evidence = EvidenceBundle(
        facts_used=[],
        patterns_used=[],
        cues_used=matched_cues,
        provider_items_used=[
            i.item_id
            for i in provider.items_for_sense(term, top_sense)
            if i.item_type == "positive_cue" and i.value.strip().lower() in matched_cues
        ],
    )
    return AnswerPlan(
        analyzed=analyzed,
        decision="answer",
        audit_reason=None,
        evidence=evidence,
        render_template="classify_context",
        render_args={
            "term": term,
            "sense_id": top_sense,
            "matched_cues": matched_cues,
        },
        confidence=confidence,
    )


def _plan_distinguish_senses(
    analyzed: AnalyzedQuestion,
    provider: AcceptedKnowledgeMemoryProvider,
    sense_mem: ExplicitSenseMemory,
) -> AnswerPlan:
    term = analyzed.term
    sense_a = analyzed.target_sense
    sense_b = analyzed.second_sense
    if not term or not sense_a or not sense_b:
        return _audit(analyzed, "missing_senses_for_distinction")

    def _sense_summary(sid: str) -> dict:
        items = provider.items_for_sense(term, sid)
        cues = _builtin_cues(sense_mem, term, sid)
        prov_cues = [i.value for i in items if i.item_type == "positive_cue"]
        actions = [i.value for i in items if i.item_type == "typical_action"]
        locations = [i.value for i in items if i.item_type == "typical_location"]
        all_cues = list(dict.fromkeys(prov_cues + cues))[:4]
        return {"cues": all_cues, "actions": actions[:3], "locations": locations[:2]}

    summary_a = _sense_summary(sense_a)
    summary_b = _sense_summary(sense_b)

    if not summary_a["cues"] or not summary_b["cues"]:
        return _audit(analyzed, "insufficient_evidence_for_distinction")

    items_a = provider.items_for_sense(term, sense_a)
    items_b = provider.items_for_sense(term, sense_b)

    evidence = EvidenceBundle(
        facts_used=[i.item_id for i in items_a + items_b if i.item_type in (
            "positive_cue", "typical_action", "typical_location")],
        patterns_used=[i.item_id for i in items_a + items_b if "pattern" in i.item_type],
        cues_used=summary_a["cues"] + summary_b["cues"],
        provider_items_used=[i.item_id for i in items_a + items_b],
    )
    return AnswerPlan(
        analyzed=analyzed,
        decision="answer",
        audit_reason=None,
        evidence=evidence,
        render_template="distinguish_senses",
        render_args={
            "term": term,
            "sense_a": sense_a,
            "sense_b": sense_b,
            "summary_a": summary_a,
            "summary_b": summary_b,
        },
        confidence=1.0,
    )


class AnswerPlanner:
    """Plans answers to QA questions using explicit memory only.

    Accepted memory provider is always used when available; default behavior
    (no provider) is not affected.
    """

    def __init__(
        self,
        provider: Optional[AcceptedKnowledgeMemoryProvider] = None,
        sense_mem: Optional[ExplicitSenseMemory] = None,
    ) -> None:
        self._provider = provider
        self._sense_mem = sense_mem if sense_mem is not None else ExplicitSenseMemory()
        # Build overlay memory once; reused across plan() calls.
        if provider is not None:
            self._overlay_mem: ExplicitSenseMemory = provider.build_overlay_memory()
        else:
            self._overlay_mem = self._sense_mem

    def plan(self, analyzed: AnalyzedQuestion) -> AnswerPlan:
        """Return an AnswerPlan for the analyzed question."""
        intent = analyzed.intent
        provider = self._provider or _NullProvider()

        if intent == "unknown_or_ambiguous" or analyzed.underconstrained:
            return _audit(analyzed, "underconstrained_question")

        if intent == "define_sense":
            return _plan_define_sense(analyzed, provider, self._sense_mem)

        if intent == "explain_cue":
            return _plan_explain_cue(analyzed, provider, self._sense_mem)

        if intent == "classify_context":
            return _plan_classify_context(
                analyzed, provider, self._sense_mem, self._overlay_mem
            )

        if intent == "distinguish_senses":
            return _plan_distinguish_senses(analyzed, provider, self._sense_mem)

        return _audit(analyzed, "unsupported_intent")


class _NullProvider:
    """Stub used when no provider is supplied — returns empty lists."""

    def items_for_sense(self, term: str, sense: str):
        return []

    def items_for_item_type(self, item_type: str):
        return []

    def provider_trace_markers(self) -> list[str]:
        return []

    @property
    def stats(self) -> dict:
        return {}
