"""Microworld QA API — FastAPI wrapper over assistant_surface logic.

Single-process server. Overlay loaded once at startup. ConversationContext
lives per session_id in memory. No auth, no DB, no external network calls.

Usage:
    python3 -m worldpgt.api.server --overlay pump-dry-run --port 8000
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import uuid
from pathlib import Path
from typing import Iterable, Optional

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel

from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator
from worldpgt.assistant_surface.context_selector import resolve_overlay
from worldpgt.assistant_surface.perf_timing import step as _timed_step
from worldpgt.assistant_surface.think_aloud import (
    build_think_aloud,
    select_inferred_facts,
    select_profile_inferred_facts,
)
from worldpgt.cognition.inference_engine import run_inference, InferenceWorkspace
from worldpgt.cognition.phrase_graph import default_phrase_graph
from worldpgt.entity_qa.synthesis_engine import synthesize, warm_definitions_index
from worldpgt.assistant_surface.types import (
    AssistantAnswer,
    OVERLAY_MODE_CUSTOM_PATH,
    OVERLAY_MODE_PUMP_DRY_RUN,
    OVERLAY_MODE_SNAPSHOT_DRY_RUN,
)
from worldpgt.dialogue.conversation_context import ConversationContext, ConversationTurn
from worldpgt.dialogue.coreference_resolver import resolve_coreferences
from worldpgt.dialogue.followup_rewriter import rewrite_followup
from worldpgt.dialogue.resolver import ResolvedQuestion, resolve_question
from worldpgt.dialogue.serving import (
    OverlayGraphReader,
    build_turn_record,
    dialogue_mode,
    serialize_bindings,
    unresolved_answer_text,
)
from worldpgt.dialogue.state import DialogueState
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.multihop_qa.assistant_adapter import try_answer_multihop
from worldpgt.reasoning.answer_behavior import (
    build_answer_plan,
    plan_is_expansion,
    prepare_persistent_evidence_graph,
)
from worldpgt.reasoning.answer_plan_renderer import render_answer_plan
from worldpgt.reasoning.pattern_discovery import PatternIndex, build_pattern_index
from worldpgt.reasoning.pattern_store import load_patterns
from worldpgt.reasoning.reasoning_adapter import try_answer_reasoning
from worldpgt.reasoning.types import GraphPattern
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex
from worldpgt.reasoning.graph_input import GraphInputLayer
from worldpgt.reasoning.relation_input_graph import default_relation_input_graph
from worldpgt.web_search.live_cache import LiveSearchCache
from worldpgt.knowledge_pump.audit_event_logger import log_audit_event
from worldpgt.assistant_surface.community_context import (
    FileCognitivePatternProvider,
    FileCommunityContextProvider,
)

_EXPERIMENTS = _ROOT / "worldpgt" / "experiments"
_ACCEPTED_OVERLAY_PATH = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY_PATH = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_SNAPSHOT_OVERLAY_PATH = (
    _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
)
_LIVE_SEARCH_CACHE_PATH = _EXPERIMENTS / "web_search_v1" / "live_cache.json"
_DEFAULT_COMMUNITY_CONTEXT_PATH = (
    _EXPERIMENTS / "community_context_v1" / "reddit_community_context.json"
)
_DEFAULT_COGNITIVE_PATTERNS_PATH = (
    _EXPERIMENTS / "community_context_v1" / "cognitive_pattern_events.json"
)
_EXPERIMENTAL_WEB_CAMPAIGN_ROOT = _EXPERIMENTS / "open_web_pump_v1"
_EVIDENCE_GROUNDED_GRAPH_FILENAME = "open_web_campaign_evidence_grounded_graph_overlay.json"
_MAIN_UI_COMPOSED_OVERLAY_PATH = (
    _EXPERIMENTS / "open_web_pump_v1" / "campaign_long_v2" / "main_ui_overlay.json"
)

_STATIC_DIR = Path(__file__).parent / "static"

# Module-level state — set once in _startup().
_orchestrator: AnswerOrchestrator | None = None
_surface_index: EntitySurfaceIndex | None = None
_overlay_items: list[dict] = []
_overlay_mode: str = "pump-dry-run"
_fact_count: int = 0
_inference_workspace: InferenceWorkspace | None = None
_graph_patterns: list[GraphPattern] = []
_pattern_index: PatternIndex | None = None
_sessions: dict[str, ConversationContext] = {}
# Dialogue v2 (explicit DialogueState) — runs per MICROWORLD_DIALOGUE_V2:
#   off    → v1 only;
#   shadow → v1 drives responses, v2 resolves + commits in parallel (default);
#   on     → v2 drives resolution; v1 context still recorded for rollback.
_sessions_v2: dict[str, DialogueState] = {}
_dialogue_traces: dict[str, dict] = {}  # session_id → last turn's v2 trace
_graph_reader: OverlayGraphReader | None = None
_community_context_path: str | None = None
_community_context_count: int = 0
_community_context_provider: FileCommunityContextProvider | None = None
_cognitive_patterns_path: str | None = None
_cognitive_patterns_count: int = 0
_cognitive_pattern_provider: FileCognitivePatternProvider | None = None

# Remembered for cache-freshness checks (see _ensure_cache_fresh).
_resolved_overlay_path: str | None = None
_startup_overlay_mode: str = "pump-dry-run"
_startup_overlay_path: str | None = None
_overlay_mtime: float | None = None
_experimental_web_graph: dict[str, object] = {"enabled": False, "item_count": 0}
_experimental_web_graph_signature: tuple[tuple[str, int, int], ...] = ()
_startup_include_experimental_web_graph: bool = False
# Persistent answer-behavior graph over the experimental slice; reset on every
# _startup() so its disk index mirrors the loaded overlay.
_experimental_edges_cache = None

app = FastAPI(title="Microworld QA API", docs_url="/docs")


# --------------------------------------------------------------------------- #
# Request / Response models
# --------------------------------------------------------------------------- #

class AskRequest(BaseModel):
    question: str
    overlay: Optional[str] = None  # informational only; server uses startup overlay
    enable_multihop: bool = False
    enable_reasoning: bool = True
    web_search: bool = False
    community_context: bool = True
    cognitive_patterns: bool = True
    think_aloud: bool = False
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    decision: str
    answer: str
    support: str
    resolved_references: list[str]
    session_id: str
    thinking: Optional[str] = None
    # Full dialogue-v2 resolution trace (slots, candidates, scores, margins).
    # Populated only when MICROWORLD_DIALOGUE_V2=on.
    dialogue: Optional[dict] = None
    # Inspectable answer-behavior plan (blocks, per-block evidence/sources,
    # score breakdowns, rejected candidates, stop reason). Populated only when
    # reasoning is enabled and the local evidence graph produced a valid plan;
    # the answer text itself is replaced only by a multi-block plan.
    answer_plan: Optional[dict] = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    _ensure_community_context_available()
    _ensure_cognitive_patterns_available()
    community_count = _community_context_provider.count() if _community_context_provider else 0
    cognitive_count = _cognitive_pattern_provider.count() if _cognitive_pattern_provider else 0
    return {
        "status": "ok",
        "overlay": _overlay_mode,
        "fact_count": _fact_count,
        "overlay_item_count": _fact_count,
        "overlay_counts": _overlay_counts(_overlay_items),
        "pump_summary": _pump_summary_counts(),
        "experimental_web_graph": _experimental_web_graph,
        "community_context": {
            "available": _community_context_path is not None,
            "path": _community_context_path,
            "item_count": community_count,
        },
        "cognitive_patterns": {
            "available": _cognitive_patterns_path is not None,
            "path": _cognitive_patterns_path,
            "event_count": cognitive_count,
            "factual_support_allowed": False,
        },
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    assert _orchestrator is not None and _surface_index is not None
    with _timed_step("server.ensure_cache_fresh"):
        _ensure_cache_fresh()
        _ensure_community_context_available()
        _ensure_cognitive_patterns_available()

    session_id = req.session_id or str(uuid.uuid4())
    context = _sessions.setdefault(session_id, ConversationContext())
    mode = dialogue_mode()

    # Dialogue v2: explicit-state resolution (pure function of the question
    # and the committed DialogueState; state mutates only in the commit step).
    state_v2: DialogueState | None = None
    resolved_v2: ResolvedQuestion | None = None
    if mode != "off":
        state_v2 = _sessions_v2.setdefault(session_id, DialogueState())
        with _timed_step("server.dialogue_resolve"):
            resolved_v2 = resolve_question(req.question, state_v2, _surface_index, _graph_reader)

    if mode == "on":
        assert resolved_v2 is not None
        effective_question = (
            req.question
            if resolved_v2.outcome == "unresolved" or resolved_v2.directives.selective_set
            else serialize_bindings(req.question, resolved_v2)
        )
        answer_style = resolved_v2.directives.answer_style
        resolved_refs = resolved_v2.resolved_references
    else:
        # v1 path drives responses in off and shadow modes.
        with _timed_step("server.coref_resolve"):
            resolution = resolve_coreferences(req.question, context, _surface_index)
        with _timed_step("server.followup_rewrite"):
            followup = rewrite_followup(resolution.resolved_question, context, _surface_index)
        effective_question = followup.resolved_question
        answer_style = followup.answer_style
        resolved_refs = [
            f"[{item.reference} → {item.display_target}]"
            for item in resolution.replacements
        ]
        if mode == "shadow" and resolved_v2 is not None:
            _log_shadow_divergence(session_id, req.question, effective_question, resolved_v2)

    with _timed_step("server.semantic_parse"):
        semantic_query = parse_semantic_query(effective_question, _surface_index)

    # Audit immediately if a dialogue reference cannot be resolved.
    if mode == "on" and resolved_v2 is not None and resolved_v2.outcome == "unresolved":
        answer = _unresolved_dialogue_v2_answer(req.question, resolved_v2)
    elif mode == "on" and resolved_v2 is not None and resolved_v2.directives.selective_set:
        # Selective references ("which one …?") need planner support to run
        # as a filter over the candidate set; until then audit honestly with
        # the candidates listed rather than mis-parse the question.
        answer = _selective_reference_answer(req.question, resolved_v2)
    elif mode != "on" and (
        resolution.unresolved_reference is not None
        and semantic_query.entity_a is None
        and semantic_query.entity_b is None
    ):
        answer = _unresolved_reference_answer(
            req.question, _overlay_mode, resolution.unresolved_reference
        )
    else:
        with _timed_step("server.orchestrator_answer"):
            answer = _orchestrator.answer(
                effective_question,
                answer_style=answer_style,
                web_search_enabled=req.web_search,
                community_context_enabled=req.community_context,
                cognitive_patterns_enabled=req.cognitive_patterns,
            )

    # Optional reasoning pass (explanatory chains / counterfactual traces).
    # Tried before multihop: reasoning question forms ("Why does X develop
    # Y?", "What if X had not founded Y?") are a disjoint pattern space, so
    # when it recognizes the question it fully answers it and multihop is
    # skipped; an unsupported form falls through unchanged.
    answer_text = answer.answer_text
    support = answer.support_kind
    reasoning_result = None

    if req.enable_reasoning:
        # Reasoning is an optional enhancement.  A malformed/discovered graph
        # pattern must never turn an otherwise evidence-backed entity answer
        # into an HTTP 500 after a large proposal campaign lands.
        with _timed_step("server.reasoning"):
            reasoning_result = _run_optional_reasoning(
                req.question,
                _overlay_items,
                _graph_patterns,
                _inference_workspace,
                _pattern_index,
            )
        if reasoning_result is not None and reasoning_result.kind != "unsupported":
            answer_text = reasoning_result.answer_text
            support = reasoning_result.support_kind
            answer = _patch_decision(answer, reasoning_result.decision)

    # Optional multi-hop pass.
    multihop = None

    should_try_multihop = (
        reasoning_result is None or reasoning_result.kind == "unsupported"
    ) and (req.enable_multihop or req.think_aloud) and (
        answer.decision == "audit"
        or answer.support_kind == "explicit_connection_path"
    )
    if should_try_multihop:
        with _timed_step("server.multihop"):
            multihop = try_answer_multihop(req.question, _overlay_items)
        if multihop.decision == "answer" or answer.support_kind == "explicit_connection_path":
            answer_text = multihop.answer_text
            support = multihop.support_kind
            # Patch decision to match multihop result when it improved the audit.
            if multihop.decision == "answer" and answer.decision == "audit":
                answer = _patch_decision(answer, "answer")

    # Answer-behavior pattern layer: when reasoning is enabled and no other
    # layer replaced the base text, try to expand a supported answer into an
    # inspectable multi-block plan over the local evidence graph.  A plan with
    # a single reliable link keeps the original short answer (only its trace
    # is exposed); no plan means nothing changes at all.
    #
    # A graph-input target may be present only in evidence relations, not in a
    # declared entity/definition record.  In that narrow case the base route
    # returns ``missing_knowledge`` even though the planner can prove a
    # multi-edge answer.  Let that plan replace *only* this ordinary knowledge
    # audit. Hard safety/privacy/current audits retain their fail-closed path.
    answer_plan_payload = None
    graph_plan_may_recover_audit = (
        answer.decision == "audit"
        and answer.support_kind == "missing_knowledge"
        and not answer.risk_flags
    )
    if (
        req.enable_reasoning
        and (answer.decision == "answer" or graph_plan_may_recover_audit)
        and (reasoning_result is None or reasoning_result.kind == "unsupported")
        and answer_text == answer.answer_text
    ):
        with _timed_step("server.answer_behavior"):
            behavior_plan = _build_optional_answer_plan(effective_question, semantic_query)
        if behavior_plan is not None:
            answer_plan_payload = behavior_plan.to_dict()
            if plan_is_expansion(behavior_plan):
                rendered_plan = _render_optional_answer_plan(behavior_plan)
                if rendered_plan:
                    answer_text = rendered_plan
                    support = "evidence_backed_answer_plan"
                    if graph_plan_may_recover_audit:
                        answer = _patch_decision(answer, "answer")

    # Optional think-aloud surface (presentation only — no behavior change).
    thinking = None
    if req.think_aloud:
        with _timed_step("server.think_aloud"):
            thinking, answer_text = _think_aloud(req.question, answer, multihop)

    # An audit is not a fact.  It is a structured acquisition signal: retain
    # the parser's resolved entity/relation so a later proposal-only campaign
    # can prioritise a real missing link without trying to infer it from prose.
    if answer.decision == "audit":
        log_audit_event(
            answer,
            entity=semantic_query.entity_a or semantic_query.entity_b,
            relation_hint=semantic_query.relation_intent,
            source="api_feedback",
        )

    # Record turn for future coreference resolution (v1 context is kept
    # up-to-date in every mode so flipping the flag never loses a session).
    with _timed_step("server.record_turn"):
        _record_turn(context, question=req.question, semantic_query=semantic_query, answer=answer)

    if state_v2 is not None and resolved_v2 is not None:
        with _timed_step("server.dialogue_commit"):
            _commit_dialogue_v2(session_id, state_v2, req.question, resolved_v2, semantic_query, answer)

    return AskResponse(
        decision=answer.decision,
        answer=answer_text,
        support=support,
        resolved_references=resolved_refs,
        session_id=session_id,
        thinking=thinking,
        dialogue=resolved_v2.to_dict() if (mode == "on" and resolved_v2 is not None) else None,
        answer_plan=answer_plan_payload,
    )


@app.get("/session/{session_id}/state")
def session_state(session_id: str) -> dict:
    """Full dialogue-v2 state for a session — the 'everything inspectable'
    requirement made concrete. Safe: state holds only canonical entity names
    and turn indices, never facts."""

    state = _sessions_v2.get(session_id)
    return {
        "session_id": session_id,
        "dialogue_mode": dialogue_mode(),
        "state": state.to_dict() if state is not None else None,
        "last_trace": _dialogue_traces.get(session_id),
    }


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

_HOW_WORKS_TOKENS = ("how do", "how does", "how is", "how are")
_HOW_WORKS_RE = re.compile(r"^\s*how\s+(?:do|does)\s+.+?\bwork\b", re.IGNORECASE)


def _is_profile_question(question: str, answer: AssistantAnswer) -> bool:
    """A profile ask ("tell me about X", "what is X") — not a mechanism question."""
    if answer.route != "entity_definition" or answer.decision == "audit":
        return False
    low = (question or "").lower()
    if any(low.startswith(t) for t in _HOW_WORKS_TOKENS):
        return False
    return True


def _needs_mechanism_gap_surface(question: str, answer: AssistantAnswer) -> bool:
    return (
        answer.route == "entity_definition"
        and answer.decision != "audit"
        and bool(_HOW_WORKS_RE.search(question or ""))
    )


def _think_aloud(question: str, answer: AssistantAnswer, multihop) -> tuple[str, str]:
    """Build the (thinking, answer) think-aloud surface for one answer."""
    cs = (answer.trace.context_summary if answer.trace else None) or {}
    matched = cs.get("matched_entities") or []
    subject = matched[0] if matched else (answer.question or question)

    synthesis = None
    profile_inferred = []
    if _is_profile_question(question, answer) and _orchestrator is not None:
        synthesis = synthesize(
            _orchestrator._provider,
            subject,
            question,
            _surface_index,
            workspace=_inference_workspace,
        )
        profile_inferred = select_profile_inferred_facts(_inference_workspace, subject)

    inferred = (
        select_inferred_facts(_inference_workspace, subject, question)
        if answer.decision == "audit"
        else profile_inferred
    )
    ta = build_think_aloud(
        answer,
        question=question,
        subject=subject,
        multihop_result=multihop,
        inferred_facts=inferred,
        synthesis=synthesis,
    )
    if _needs_mechanism_gap_surface(question, answer):
        return (
            ta.thinking,
            (
                f"{ta.answer}\n\n"
                f"I know what {subject} refers to, but I don't have verified "
                "information about how it works mechanically."
            ),
        )
    return ta.thinking, ta.answer


def _unresolved_reference_answer(question: str, overlay_mode: str, reference: str) -> AssistantAnswer:
    return AssistantAnswer(
        question=question,
        decision="audit",
        route="unknown_or_unsupported",
        answer_text=f"unresolved_reference: could not determine what {reference!r} refers to",
        overlay_mode=overlay_mode,
        supported_by_context=False,
        support_kind="missing_knowledge",
        source_system="dialogue_context",
        safe_for_general_runtime=False,
    )


def _unresolved_dialogue_v2_answer(question: str, resolved: ResolvedQuestion) -> AssistantAnswer:
    return AssistantAnswer(
        question=question,
        decision="audit",
        route="unknown_or_unsupported",
        answer_text=unresolved_answer_text(resolved),
        overlay_mode=_overlay_mode,
        supported_by_context=False,
        support_kind="missing_knowledge",
        source_system="dialogue_context_v2",
        safe_for_general_runtime=False,
    )


def _selective_reference_answer(question: str, resolved: ResolvedQuestion) -> AssistantAnswer:
    candidates = ", ".join(resolved.directives.selective_set)
    return AssistantAnswer(
        question=question,
        decision="audit",
        route="unknown_or_unsupported",
        answer_text=(
            "selective_dialogue_reference: this question selects among active "
            f"dialogue entities ({candidates}); ask about one of them directly"
        ),
        overlay_mode=_overlay_mode,
        supported_by_context=False,
        support_kind="missing_knowledge",
        source_system="dialogue_context_v2",
        safe_for_general_runtime=False,
    )


def _log_shadow_divergence(
    session_id: str,
    question: str,
    v1_effective: str,
    resolved_v2: ResolvedQuestion,
) -> None:
    """Shadow mode: log where v2 would diverge from the serving v1 path."""

    if resolved_v2.outcome == "no_slots" and v1_effective == question:
        return
    v2_effective = (
        question if resolved_v2.outcome == "unresolved"
        else serialize_bindings(question, resolved_v2)
    )
    marker = "match" if v2_effective == v1_effective else "DIVERGENCE"
    print(
        f"[dialogue_v2.shadow] {marker} session={session_id} q={question!r} "
        f"v1={v1_effective!r} v2={v2_effective!r} outcome={resolved_v2.outcome}",
        flush=True,
    )


def _commit_dialogue_v2(
    session_id: str,
    state_v2: DialogueState,
    question: str,
    resolved_v2: ResolvedQuestion,
    semantic_query,
    answer: AssistantAnswer,
) -> None:
    answer_text_entities: list[str] = []
    hints: dict[str, str] = {}
    if answer.decision != "audit":
        answer_text_entities = _dedupe([
            canonical
            for _s, canonical, _st, _e in _surface_index.find_in_text(answer.answer_text)
        ])
        if answer.support_kind == "web_search_result":
            hint = _web_search_entity_hint(answer.answer_text)
            if hint:
                name, entity_type = hint
                hints[name] = entity_type
                answer_text_entities = _dedupe([name] + answer_text_entities)
    record = build_turn_record(
        question=question,
        resolved=resolved_v2,
        semantic_query=semantic_query,
        answer=answer,
        surface_index=_surface_index,
        answer_text_entities=answer_text_entities,
        entity_type_hints=hints,
    )
    state_v2.commit(record)
    _dialogue_traces[session_id] = resolved_v2.to_dict()


def _patch_decision(answer: AssistantAnswer, decision: str) -> AssistantAnswer:
    """Return a shallow copy of answer with decision replaced."""
    from dataclasses import replace
    return replace(answer, decision=decision)


def _dedupe(seq: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


# Matches the "Source: <title> — <url>" line render_web_answer always emits
# for its top-ranked result.
_WEB_SEARCH_SOURCE_RE = re.compile(r"^Source: (.+?) — https?://\S+", re.MULTILINE)
_WEB_SEARCH_TITLE_SUFFIX_RE = re.compile(
    r"\s*[-–]\s*Wikipedia.*$|,?\s*the free encyclopedia.*$", re.IGNORECASE
)
_PERSON_PRONOUN_RE = re.compile(r"\b(?:he|she|his|her)\b", re.IGNORECASE)


def _web_search_entity_hint(answer_text: str) -> tuple[str, str] | None:
    """Best-effort (name, type) hint for a live web-search answer's subject.

    The overlay's static entity index has no way to type a person who only
    ever appeared in a live search result, so without this hint follow-up
    pronouns ("Where was she born?") can never resolve. Scoped to
    support_kind == "web_search_result" answers only — never touches the
    overlay or its index.
    """

    match = _WEB_SEARCH_SOURCE_RE.search(answer_text)
    if not match:
        return None
    title = _WEB_SEARCH_TITLE_SUFFIX_RE.sub("", match.group(1)).strip()
    if not title:
        return None
    entity_type = "person" if _PERSON_PRONOUN_RE.search(answer_text) else "topic"
    return title, entity_type


def _record_turn(
    context: ConversationContext,
    *,
    question: str,
    semantic_query,
    answer: AssistantAnswer,
) -> None:
    if answer.decision == "audit":
        mentioned: list[str] = []
        primary = None
        entity_types: dict[str, str] = {}
    else:
        # Scan answer text for entity surface forms so that entities mentioned
        # in the answer (e.g. "Elon Musk" in "SpaceX was founded by Elon Musk.")
        # become available for coreference resolution in follow-up turns.
        text_entities = [
            canonical
            for _surface, canonical, _start, _end in _surface_index.find_in_text(answer.answer_text)
        ]
        ctx_entities = (
            answer.trace.context_summary.get("matched_entities", [])
            if answer.trace and answer.trace.context_summary
            else []
        )
        mentioned = _dedupe(
            text_entities
            + ctx_entities
            + [semantic_query.entity_a or "", semantic_query.entity_b or ""]
        )
        primary = _primary_from_answer(semantic_query.entity_a, text_entities) or (
            mentioned[0] if mentioned else None
        )

        entity_types = {}
        if answer.support_kind == "web_search_result":
            hint = _web_search_entity_hint(answer.answer_text)
            if hint:
                name, entity_type = hint
                entity_types[name] = entity_type
                mentioned = _dedupe([name] + mentioned)
                primary = name

    if answer.decision == "no" and primary:
        mentioned = _dedupe([primary] + mentioned)

    turn = ConversationTurn(
        question=question,
        semantic_query=semantic_query,
        decision=answer.decision,
        primary_entity=primary,
        mentioned_entities=mentioned,
        relation_type=semantic_query.relation_intent,
        entity_types=entity_types,
    )
    context.append_turn(turn)


def _primary_from_answer(entity_a: str | None, text_entities: list[str]) -> str | None:
    """Choose the best primary entity for a confirmed answer turn.

    When entity_a is an organization but the answer names exactly one person
    (e.g. "SpaceX was founded by Elon Musk."), the person is the salient
    referent and should be primary so follow-up pronouns like "he" resolve.
    """
    if not entity_a:
        return text_entities[0] if text_entities else None
    if _surface_index is None:
        return entity_a
    ea_type = _surface_index.entity_type(entity_a)
    if ea_type not in ("organization", "vehicle", "program"):
        return entity_a
    persons = [e for e in text_entities if _surface_index.entity_type(e) == "person"]
    if len(persons) == 1:
        return persons[0]
    return entity_a


def _load_overlay_items(overlay_path: str) -> list[dict]:
    rows = json.loads(Path(overlay_path).read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        return []
    return [r for r in rows if isinstance(r, dict)]


def _available_experimental_web_graph_paths() -> tuple[Path, ...]:
    """Discover all completed evidence-grounded campaigns in deterministic order.

    A campaign writes this artifact only after its proposal-only consolidation
    completes.  Discovery removes the need to edit server code for every new
    campaign and keeps title-led/raw artifacts out of the user-facing graph.
    """
    if not _EXPERIMENTAL_WEB_CAMPAIGN_ROOT.is_dir():
        return ()
    return tuple(sorted(
        path for path in _EXPERIMENTAL_WEB_CAMPAIGN_ROOT.glob(
            f"campaign_*/{_EVIDENCE_GROUNDED_GRAPH_FILENAME}"
        )
        if path.is_file()
    ))


def _experimental_web_graph_fingerprint(
    paths: Iterable[Path] | None = None,
) -> tuple[tuple[str, int, int], ...]:
    """Cheap change token for discovery and completed-campaign updates."""
    selected = tuple(paths) if paths is not None else _available_experimental_web_graph_paths()
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in selected
        if path.is_file()
    )


def _experimental_graph_key(text: object) -> str:
    return " ".join(str(text or "").casefold().split())


def _merge_experimental_graph_items(items: Iterable[dict]) -> list[dict]:
    """Coalesce only evidence-grounded graph duplicates across campaigns.

    This is a serving-only operation.  It does not turn proposal data into
    accepted memory; it merely makes independent provenance visible on a
    single semantic edge instead of leaving duplicate rows to compete in QA.
    """
    rows = [dict(item) for item in items]
    entity_groups: dict[str, list[dict]] = {}
    relation_groups: dict[tuple[str, str, str], list[dict]] = {}
    for item in rows:
        tier = str(item.get("experimental_tier") or "")
        if item.get("overlay_type") == "overlay_entity" and tier.startswith("evidence_grounded_"):
            key = _experimental_graph_key(item.get("label"))
            if key:
                entity_groups.setdefault(key, []).append(item)
        elif item.get("overlay_type") == "overlay_relation" and tier == "evidence_grounded_abstract_relation_v1":
            key = (
                _experimental_graph_key(item.get("subject")),
                _experimental_graph_key(item.get("predicate")),
                _experimental_graph_key(item.get("object")),
            )
            if all(key):
                relation_groups.setdefault(key, []).append(item)

    merged_entities: dict[str, dict] = {}
    for key, group in entity_groups.items():
        merged = dict(group[0])
        merged["aliases"] = sorted({
            str(alias).strip()
            for item in group
            for alias in (item.get("aliases") or [])
            if str(alias).strip() and _experimental_graph_key(alias) != key
        }, key=lambda value: (len(value), value.casefold(), value))
        merged_entities[key] = merged

    merged_relations: dict[tuple[str, str, str], dict] = {}
    for key, group in relation_groups.items():
        merged = dict(group[0])
        source_urls = sorted({
            str(url).strip()
            for item in group
            for url in [item.get("source_url"), *(item.get("supporting_sources") or [])]
            if str(url or "").strip()
        })
        evidence: list[str] = []
        for item in group:
            for text in [item.get("evidence_text"), *(item.get("supporting_evidence") or [])]:
                compact = " ".join(str(text or "").split())
                if compact and compact not in evidence:
                    evidence.append(compact)
        source_count = len(source_urls)
        quality = dict(merged.get("evidence_quality") or {})
        quality["corroboration"] = "independent_sources" if source_count > 1 else "single_source"
        merged.update({
            "support_count": sum(max(1, int(item.get("support_count") or 0)) for item in group),
            "supporting_source_count": source_count,
            "supporting_sources": source_urls,
            "supporting_evidence": evidence[:6],
            "evidence_quality": quality,
        })
        merged_relations[key] = merged

    result: list[dict] = []
    emitted_entities: set[str] = set()
    emitted_relations: set[tuple[str, str, str]] = set()
    for item in rows:
        tier = str(item.get("experimental_tier") or "")
        if item.get("overlay_type") == "overlay_entity" and tier.startswith("evidence_grounded_"):
            key = _experimental_graph_key(item.get("label"))
            if key in emitted_entities:
                continue
            emitted_entities.add(key)
            result.append(merged_entities[key])
            continue
        if item.get("overlay_type") == "overlay_relation" and tier == "evidence_grounded_abstract_relation_v1":
            key = (
                _experimental_graph_key(item.get("subject")),
                _experimental_graph_key(item.get("predicate")),
                _experimental_graph_key(item.get("object")),
            )
            if key in emitted_relations:
                continue
            emitted_relations.add(key)
            result.append(merged_relations[key])
            continue
        result.append(item)
    return result


def _compose_main_ui_overlay(
    base_overlay_path: str | Path,
    graph_overlay_paths: str | Path | Iterable[str | Path],
) -> Path:
    """Compose normal UI knowledge with the explicit experimental graph.

    The result is a serving-only artifact, never accepted or promoted memory.
    It preserves the normal overlay while making the user-authorized
    experimental graph available to the main UI.
    """
    base_items = _load_overlay_items(str(base_overlay_path))
    if isinstance(graph_overlay_paths, (str, Path)):
        graph_paths = (Path(graph_overlay_paths),)
    else:
        graph_paths = tuple(Path(path) for path in graph_overlay_paths)
    graph_items = _merge_experimental_graph_items([
        item
        for path in graph_paths
        if path.is_file()
        for item in _load_overlay_items(str(path))
    ])
    seen: set[str] = set()
    merged: list[dict] = []
    for item in [*base_items, *graph_items]:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            merged.append(item)
    payload = json.dumps(merged, indent=2, ensure_ascii=False) + "\n"
    _MAIN_UI_COMPOSED_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not _MAIN_UI_COMPOSED_OVERLAY_PATH.is_file() or _MAIN_UI_COMPOSED_OVERLAY_PATH.read_text(encoding="utf-8") != payload:
        temporary = _MAIN_UI_COMPOSED_OVERLAY_PATH.with_suffix(".json.tmp")
        temporary.write_text(payload, encoding="utf-8")
        temporary.replace(_MAIN_UI_COMPOSED_OVERLAY_PATH)
    return _MAIN_UI_COMPOSED_OVERLAY_PATH


def _run_optional_reasoning(
    question: str,
    items: list[dict],
    patterns: list[GraphPattern],
    workspace: object | None,
    pattern_index: PatternIndex | None,
):
    """Return an optional reasoning result without sacrificing base QA.

    Experimental proposal graphs may expose edge cases in derived pattern
    discovery.  The evidence-backed answer is still usable when that optional
    enhancement declines or fails, so its failure is intentionally isolated.
    """
    try:
        return try_answer_reasoning(
            question,
            items,
            patterns=patterns,
            workspace=workspace,
            pattern_index=pattern_index,
        )
    except Exception:
        return None


def _build_optional_answer_plan(question: str, semantic_query):
    """Build the answer-behavior plan without ever risking the base answer.

    Targets come only from already-resolved entities (semantic parser first,
    surface index second) — the layer never guesses a target from question
    shape.  Any failure inside the optional layer degrades to None, which the
    caller treats as "keep the existing answer unchanged".
    """
    try:
        targets = [
            t for t in (semantic_query.entity_a, semantic_query.entity_b) if t
        ]
        if not targets and _surface_index is not None:
            targets = list(dict.fromkeys(
                canonical
                for _surface, canonical, _start, _end in _surface_index.find_in_text(question)
            ))
        if not targets:
            return None
        from worldpgt.relation_extraction_v2.relation_policy import relation_intents_from_text
        explicit_intents = relation_intents_from_text(question)
        # The input graph is the source of paraphrase-to-predicate semantics.
        # Preserve every denoted edge for coordinated questions; the legacy
        # policy extractor remains as a complementary source for its existing
        # controlled forms.
        graph_intents = frozenset(default_relation_input_graph().resolve_all(
            question,
            entity_spans=(
                (start, end)
                for _surface, _canonical, start, end in _surface_index.find_in_text(question)
            ) if _surface_index is not None else (),
        ))
        explicit_intents = explicit_intents | graph_intents
        predicate_filter = (
            explicit_intents if len(explicit_intents) > 1
            else semantic_query.relation_intent
        )
        return build_answer_plan(
            question,
            [],
            targets=targets,
            predicate_filter=predicate_filter,
            prepared_edges=_experimental_evidence_edges(),
        )
    except Exception:
        return None


def _experimental_relation_items() -> list[dict]:
    """The proposal/experimental-only slice of the composed overlay.

    Hard boundary: the answer-behavior layer must never expand an answer
    using accepted or promoted memory facts, even though those facts sit in
    the same in-memory ``_overlay_items`` list once composed for serving.
    ``experimental_tier`` is the same provenance marker
    ``_merge_experimental_graph_items`` already uses to recognize open-web
    proposal relations, so this reuses that exact criterion instead of
    introducing a second one.  Definitions remain outside this first
    relation-only behavior layer.
    """
    return [
        item
        for item in _overlay_items
        if item.get("overlay_type") == "overlay_relation"
        and str(item.get("experimental_tier") or "").startswith("evidence_grounded_")
    ]


def _experimental_evidence_edges():
    """Persistent answer-behavior graph, opened once per loaded overlay.

    ``_startup`` resets the cache holder whenever the overlay is (re)loaded,
    so its fingerprinted disk index follows the existing overlay lifecycle
    instead of adding a second invalidation scheme.
    """
    global _experimental_edges_cache
    if _experimental_edges_cache is None:
        resolved = Path(_resolved_overlay_path or _MAIN_UI_COMPOSED_OVERLAY_PATH)
        stat = resolved.stat()
        relation_items = _experimental_relation_items()
        _experimental_edges_cache = prepare_persistent_evidence_graph(
            relation_items,
            resolved.with_suffix(".answer_behavior.sqlite"),
            source_fingerprint=(
                f"{resolved.resolve()}:{stat.st_mtime_ns}:{stat.st_size}:"
                f"experimental-relations-v1:{len(relation_items)}"
            ),
        )
    return _experimental_edges_cache


def _render_optional_answer_plan(plan) -> str:
    try:
        return render_answer_plan(plan)
    except Exception:
        return ""


def _overlay_counts(items: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in items:
        key = str(item.get("overlay_type") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _pump_summary_counts() -> dict:
    summary_path = _EXPERIMENTS / "knowledge_pump_v1" / "pump_summary.json"
    if not summary_path.is_file():
        return {}
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(summary, dict):
        return {}
    keys = (
        "pump_answerable_fact_delta_count",
        "pump_definition_delta_count",
        "pump_relation_delta_count",
        "frontier_titles_total",
        "dynamic_frontier_total",
        "fetch_success_count_total",
        "ready_for_ingestion_count_total",
        "all_critical_passed",
    )
    return {key: summary[key] for key in keys if key in summary}


# --------------------------------------------------------------------------- #
# Startup
# --------------------------------------------------------------------------- #

def _startup(
    overlay_mode: str,
    overlay_path: str | None = None,
    *,
    community_context_path: str | None = None,
    cognitive_patterns_path: str | None = None,
    include_experimental_web_graph: bool = False,
    experimental_graph_paths: Iterable[str | Path] | None = None,
    warm_phrase_graph_on_startup: bool = True,
) -> None:
    global _orchestrator, _surface_index, _overlay_items, _overlay_mode, _fact_count
    global _inference_workspace, _graph_patterns, _pattern_index, _graph_reader
    global _resolved_overlay_path, _startup_overlay_mode, _startup_overlay_path, _overlay_mtime
    global _community_context_path, _community_context_count, _community_context_provider
    global _cognitive_patterns_path, _cognitive_patterns_count, _cognitive_pattern_provider
    global _experimental_web_graph, _experimental_web_graph_signature, _startup_include_experimental_web_graph
    global _experimental_edges_cache
    if _experimental_edges_cache is not None:
        close = getattr(_experimental_edges_cache, "close", None)
        if close is not None:
            close()
    _experimental_edges_cache = None

    _overlay_mode = OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode
    _startup_overlay_mode = overlay_mode
    _startup_overlay_path = overlay_path

    base_resolved_path = overlay_path or resolve_overlay(overlay_mode)[0]
    resolved_path = base_resolved_path
    _experimental_web_graph = {"enabled": False, "item_count": 0}
    _experimental_web_graph_signature = ()
    _startup_include_experimental_web_graph = include_experimental_web_graph
    available_graph_paths = (
        tuple(Path(path) for path in experimental_graph_paths)
        if experimental_graph_paths is not None
        else _available_experimental_web_graph_paths()
    )
    if (include_experimental_web_graph or experimental_graph_paths is not None) and overlay_path is None and available_graph_paths:
        graph_items = _merge_experimental_graph_items([
            item
            for path in available_graph_paths
            for item in _load_overlay_items(str(path))
        ])
        resolved_path = str(_compose_main_ui_overlay(base_resolved_path, available_graph_paths))
        _overlay_mode = f"{overlay_mode}+experimental-web-graph"
        _experimental_web_graph = {
            "enabled": True,
            "item_count": len(graph_items),
            "paths": [str(path) for path in available_graph_paths],
            "trust": "proposal_open_web_exploratory",
        }
        _experimental_web_graph_signature = _experimental_web_graph_fingerprint(available_graph_paths)
    _resolved_overlay_path = resolved_path
    _overlay_mtime = Path(resolved_path).stat().st_mtime

    _overlay_items = _load_overlay_items(resolved_path)
    _fact_count = len(_overlay_items)
    # Read-only role-holder lookup for dialogue-v2 role descriptors
    # ("the founder"); built once from the loaded overlay.
    _graph_reader = OverlayGraphReader(_overlay_items)
    # Inference workspace and pattern index are the expensive (O(overlay
    # size)) structures the reasoning layer and synthesis used to rebuild on
    # every single request — compute them once here and hand the same
    # instances to everything downstream (orchestrator, reasoning adapter).
    _inference_workspace = run_inference(_overlay_items)
    _pattern_index = build_pattern_index(_overlay_items)

    community_provider = None
    _community_context_path = community_context_path
    _community_context_count = 0
    _community_context_provider = None
    if community_context_path:
        community_provider = FileCommunityContextProvider(community_context_path)
        _community_context_provider = community_provider
        _community_context_count = community_provider.count()

    cognitive_provider = None
    _cognitive_patterns_path = cognitive_patterns_path
    _cognitive_patterns_count = 0
    _cognitive_pattern_provider = None
    if cognitive_patterns_path:
        cognitive_provider = FileCognitivePatternProvider(cognitive_patterns_path)
        _cognitive_pattern_provider = cognitive_provider
        _cognitive_patterns_count = cognitive_provider.count()

    _orchestrator = AnswerOrchestrator(
        overlay_mode,
        overlay_path=resolved_path if resolved_path != base_resolved_path else overlay_path,
        inference_workspace=_inference_workspace,
        live_cache=LiveSearchCache(_LIVE_SEARCH_CACHE_PATH),
        community_context_provider=community_provider,
        community_context_enabled=community_provider is not None,
        cognitive_pattern_provider=cognitive_provider,
        cognitive_patterns_enabled=cognitive_provider is not None,
    )
    _surface_index = EntitySurfaceIndex(
        accepted_overlay_path=_ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=Path(resolved_path),
        snapshot_overlay_path=_SNAPSHOT_OVERLAY_PATH,
        graph_input=GraphInputLayer.from_overlay_items(_overlay_items),
    )
    # Discovered graph patterns are optional context for the reasoning layer —
    # loaded from the nightly artifact if present, empty (never fabricated) if
    # discovery has not run yet for this overlay.
    _graph_patterns = load_patterns()
    # Warm the synthesis engine's per-provider definitions-by-subject index so
    # the first "tell me about X" request doesn't pay its one-time O(overlay
    # size) build cost.
    warm_definitions_index(_orchestrator._provider)
    # Warm the trained phrase graph (fragments, transitions, word_types) so
    # neither the first open-synthesis speech-first render nor the first
    # dialogue-v2 typed-demonstrative/role-descriptor question pays its
    # one-time spaCy-parsing training cost (~40-60s, dominated by community
    # context sentence parsing) inline on a live request.
    if warm_phrase_graph_on_startup:
        default_phrase_graph()

    print(
        f"[microworld-api] overlay={_overlay_mode}  overlay_items={_fact_count}  "
        f"patterns={len(_graph_patterns)}  community_context={_community_context_count}  "
        f"cognitive_patterns={_cognitive_patterns_count}",
        flush=True,
    )


def _ensure_cache_fresh() -> None:
    """Refresh only when the base overlay or a completed campaign changed.

    The check is a cheap stat/fingerprint pass per request.  It detects a new
    evidence-grounded campaign even though that campaign did not yet exist
    when the server first composed its UI overlay.
    """
    if _resolved_overlay_path is None:
        return
    if _startup_include_experimental_web_graph and _startup_overlay_path is None:
        if _experimental_web_graph_fingerprint() != _experimental_web_graph_signature:
            _startup(
                _startup_overlay_mode,
                overlay_path=_startup_overlay_path,
                community_context_path=_community_context_path,
                cognitive_patterns_path=_cognitive_patterns_path,
                include_experimental_web_graph=_startup_include_experimental_web_graph,
            )
            return
    current_mtime = Path(_resolved_overlay_path).stat().st_mtime
    if current_mtime == _overlay_mtime:
        return
    _startup(
        _startup_overlay_mode,
        overlay_path=_startup_overlay_path,
        community_context_path=_community_context_path,
        cognitive_patterns_path=_cognitive_patterns_path,
        include_experimental_web_graph=_startup_include_experimental_web_graph,
    )


def _ensure_community_context_available() -> None:
    """Attach the default community-context artifact if the pump created it
    after the server had already started.
    """
    global _community_context_path, _community_context_count, _community_context_provider
    if _orchestrator is None:
        return
    candidate = Path(_community_context_path) if _community_context_path else _DEFAULT_COMMUNITY_CONTEXT_PATH
    if not candidate.is_file():
        return
    if _community_context_path == str(candidate) and _community_context_provider is not None:
        _community_context_count = _community_context_provider.count()
        return
    provider = FileCommunityContextProvider(candidate)
    _community_context_path = str(candidate)
    _community_context_provider = provider
    _community_context_count = provider.count()
    _orchestrator.set_community_context_provider(provider, enabled=True)


def _ensure_cognitive_patterns_available() -> None:
    """Attach the default cognitive-pattern artifact if the pump created it
    after the server had already started.
    """
    global _cognitive_patterns_path, _cognitive_patterns_count, _cognitive_pattern_provider
    if _orchestrator is None:
        return
    candidate = (
        Path(_cognitive_patterns_path)
        if _cognitive_patterns_path
        else _DEFAULT_COGNITIVE_PATTERNS_PATH
    )
    if not candidate.is_file():
        return
    if _cognitive_patterns_path == str(candidate) and _cognitive_pattern_provider is not None:
        _cognitive_patterns_count = _cognitive_pattern_provider.count()
        return
    provider = FileCognitivePatternProvider(candidate)
    _cognitive_patterns_path = str(candidate)
    _cognitive_pattern_provider = provider
    _cognitive_patterns_count = provider.count()
    _orchestrator.set_cognitive_pattern_provider(provider, enabled=True)


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> None:
    import os
    import uvicorn

    _BUILTIN_OVERLAYS = ("accepted", "promoted", "snapshot-dry-run", "pump-dry-run")

    parser = argparse.ArgumentParser(description="Microworld QA API Server")
    parser.add_argument(
        "--overlay",
        default="pump-dry-run",
        help="Built-in overlay mode (%s) or a domain overlay as "
             "'domain:<path-to-overlay.json>' produced by run_domain_bootstrap_v1."
             % ", ".join(_BUILTIN_OVERLAYS),
    )
    parser.add_argument("--overlay-path", default=None)
    parser.add_argument(
        "--no-experimental-web-graph",
        action="store_true",
        help="Do not compose the available experimental open-web graph into the main UI overlay.",
    )
    parser.add_argument(
        "--community-context",
        default=None,
        help="Path to a low-trust community-context artifact. If omitted, the "
             "default Reddit context artifact is used when it exists.",
    )
    parser.add_argument(
        "--no-community-context",
        action="store_true",
        help="Disable automatic community-context loading.",
    )
    parser.add_argument(
        "--cognitive-patterns",
        default=None,
        help="Path to a cognitive_pattern_events.json artifact. If omitted, "
             "the default community-context pattern artifact is used when it exists.",
    )
    parser.add_argument(
        "--no-cognitive-patterns",
        action="store_true",
        help="Disable automatic cognitive-pattern surface planning.",
    )
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args(argv)

    overlay_path = args.overlay_path
    overlay_mode = args.overlay

    # Domain overlay: '--overlay domain:<path>' serves a cold-start domain
    # overlay produced by the universal bootstrap. The server treats it as a
    # custom overlay path (it does not know the difference).
    if overlay_mode.startswith("domain:"):
        overlay_path = overlay_mode.split("domain:", 1)[1]
        overlay_mode = "pump-dry-run"  # pack semantics; path overrides anyway
    elif overlay_mode not in _BUILTIN_OVERLAYS:
        parser.error(
            f"--overlay must be one of {_BUILTIN_OVERLAYS} or 'domain:<path>'"
        )

    if overlay_path is not None:
        p = Path(overlay_path)
        if not p.is_file():
            parser.error(f"overlay path does not exist: {p}")

    community_context_path = None
    if not args.no_community_context:
        candidate = Path(args.community_context) if args.community_context else _DEFAULT_COMMUNITY_CONTEXT_PATH
        if candidate.is_file():
            community_context_path = str(candidate)
        elif args.community_context:
            parser.error(f"community context path does not exist: {candidate}")

    cognitive_patterns_path = None
    if not args.no_cognitive_patterns:
        candidate = Path(args.cognitive_patterns) if args.cognitive_patterns else _DEFAULT_COGNITIVE_PATTERNS_PATH
        if candidate.is_file():
            cognitive_patterns_path = str(candidate)
        elif args.cognitive_patterns:
            parser.error(f"cognitive patterns path does not exist: {candidate}")

    _startup(
        overlay_mode,
        overlay_path=overlay_path,
        community_context_path=community_context_path,
        cognitive_patterns_path=cognitive_patterns_path,
        include_experimental_web_graph=not args.no_experimental_web_graph,
    )

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
