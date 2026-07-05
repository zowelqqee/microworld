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
from typing import Optional

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
from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
from worldpgt.multihop_qa.assistant_adapter import try_answer_multihop
from worldpgt.reasoning.pattern_discovery import PatternIndex, build_pattern_index
from worldpgt.reasoning.pattern_store import load_patterns
from worldpgt.reasoning.reasoning_adapter import try_answer_reasoning
from worldpgt.reasoning.types import GraphPattern
from worldpgt.relation_extraction_v2.entity_surface_index import EntitySurfaceIndex

_EXPERIMENTS = _ROOT / "worldpgt" / "experiments"
_ACCEPTED_OVERLAY_PATH = _EXPERIMENTS / "accepted_wiki_memory_overlay_v1.json"
_PROMOTED_OVERLAY_PATH = (
    _EXPERIMENTS / "self_ingestion_v1" / "promotion" / "promoted_wiki_memory_overlay_v1.json"
)
_SNAPSHOT_OVERLAY_PATH = (
    _EXPERIMENTS / "wiki_snapshot_ingestion_v1" / "snapshot_dry_run_overlay.json"
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

# Remembered for cache-freshness checks (see _ensure_cache_fresh).
_resolved_overlay_path: str | None = None
_startup_overlay_mode: str = "pump-dry-run"
_startup_overlay_path: str | None = None
_overlay_mtime: float | None = None

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
    think_aloud: bool = False
    session_id: Optional[str] = None


class AskResponse(BaseModel):
    decision: str
    answer: str
    support: str
    resolved_references: list[str]
    session_id: str
    thinking: Optional[str] = None


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

@app.get("/")
def root() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "overlay": _overlay_mode,
        "fact_count": _fact_count,
        "overlay_item_count": _fact_count,
        "overlay_counts": _overlay_counts(_overlay_items),
        "pump_summary": _pump_summary_counts(),
    }


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    assert _orchestrator is not None and _surface_index is not None
    with _timed_step("server.ensure_cache_fresh"):
        _ensure_cache_fresh()

    session_id = req.session_id or str(uuid.uuid4())
    context = _sessions.setdefault(session_id, ConversationContext())

    # Coreference resolution against conversation history.
    with _timed_step("server.coref_resolve"):
        resolution = resolve_coreferences(req.question, context, _surface_index)
    with _timed_step("server.followup_rewrite"):
        followup = rewrite_followup(resolution.resolved_question, context, _surface_index)
    effective_question = followup.resolved_question
    with _timed_step("server.semantic_parse"):
        semantic_query = parse_semantic_query(effective_question, _surface_index)

    resolved_refs = [
        f"[{item.reference} → {item.display_target}]"
        for item in resolution.replacements
    ]

    # Audit immediately if reference cannot be resolved.
    if (
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
                answer_style=followup.answer_style,
                web_search_enabled=req.web_search,
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
        with _timed_step("server.reasoning"):
            reasoning_result = try_answer_reasoning(
                req.question,
                _overlay_items,
                patterns=_graph_patterns,
                workspace=_inference_workspace,
                pattern_index=_pattern_index,
            )
        if reasoning_result.kind != "unsupported":
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

    # Optional think-aloud surface (presentation only — no behavior change).
    thinking = None
    if req.think_aloud:
        with _timed_step("server.think_aloud"):
            thinking, answer_text = _think_aloud(req.question, answer, multihop)

    # Record turn for future coreference resolution.
    with _timed_step("server.record_turn"):
        _record_turn(context, question=req.question, semantic_query=semantic_query, answer=answer)

    return AskResponse(
        decision=answer.decision,
        answer=answer_text,
        support=support,
        resolved_references=resolved_refs,
        session_id=session_id,
        thinking=thinking,
    )


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

    if answer.decision == "no" and primary:
        mentioned = _dedupe([primary] + mentioned)

    turn = ConversationTurn(
        question=question,
        semantic_query=semantic_query,
        decision=answer.decision,
        primary_entity=primary,
        mentioned_entities=mentioned,
        relation_type=semantic_query.relation_intent,
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

def _startup(overlay_mode: str, overlay_path: str | None = None) -> None:
    global _orchestrator, _surface_index, _overlay_items, _overlay_mode, _fact_count
    global _inference_workspace, _graph_patterns, _pattern_index
    global _resolved_overlay_path, _startup_overlay_mode, _startup_overlay_path, _overlay_mtime

    _overlay_mode = OVERLAY_MODE_CUSTOM_PATH if overlay_path is not None else overlay_mode
    _startup_overlay_mode = overlay_mode
    _startup_overlay_path = overlay_path

    resolved_path = overlay_path or resolve_overlay(overlay_mode)[0]
    _resolved_overlay_path = resolved_path
    _overlay_mtime = Path(resolved_path).stat().st_mtime

    _overlay_items = _load_overlay_items(resolved_path)
    _fact_count = len(_overlay_items)
    # Inference workspace and pattern index are the expensive (O(overlay
    # size)) structures the reasoning layer and synthesis used to rebuild on
    # every single request — compute them once here and hand the same
    # instances to everything downstream (orchestrator, reasoning adapter).
    _inference_workspace = run_inference(_overlay_items)
    _pattern_index = build_pattern_index(_overlay_items)

    _orchestrator = AnswerOrchestrator(
        overlay_mode,
        overlay_path=overlay_path,
        inference_workspace=_inference_workspace,
    )
    _surface_index = EntitySurfaceIndex(
        accepted_overlay_path=_ACCEPTED_OVERLAY_PATH,
        promoted_overlay_path=Path(resolved_path),
        snapshot_overlay_path=_SNAPSHOT_OVERLAY_PATH,
    )
    # Discovered graph patterns are optional context for the reasoning layer —
    # loaded from the nightly artifact if present, empty (never fabricated) if
    # discovery has not run yet for this overlay.
    _graph_patterns = load_patterns()
    # Warm the synthesis engine's per-provider definitions-by-subject index so
    # the first "tell me about X" request doesn't pay its one-time O(overlay
    # size) build cost.
    warm_definitions_index(_orchestrator._provider)

    print(
        f"[microworld-api] overlay={_overlay_mode}  overlay_items={_fact_count}  "
        f"patterns={len(_graph_patterns)}",
        flush=True,
    )


def _ensure_cache_fresh() -> None:
    """Recompute the inference/pattern caches only if the overlay file on
    disk changed since we last cached it (cheap ``stat()`` check per request;
    a full ``_startup()`` re-init only on an actual mtime change, e.g. the
    overlay was regenerated by a nightly job while the server kept running).
    """
    if _resolved_overlay_path is None:
        return
    current_mtime = Path(_resolved_overlay_path).stat().st_mtime
    if current_mtime == _overlay_mtime:
        return
    _startup(_startup_overlay_mode, overlay_path=_startup_overlay_path)


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

    _startup(overlay_mode, overlay_path=overlay_path)

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
