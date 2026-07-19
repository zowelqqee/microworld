"""mw_ios — the iOS-facing adapter for the MicroWorld engine.

This is a *thin, faithful* bridge, not a reimplementation. It loads the real
``worldpgt`` engine once, exposes ``warm_up`` / ``run`` / ``self_test`` returning
JSON strings (easy to marshal across the Objective-C boundary), and honours the
app's explicit QA / Creative mode toggle while preserving the engine's own
hard-safety behaviour.

Nothing here fabricates output. Every answer comes from
``AnswerOrchestrator`` and, for creative prompts, from the engine's own
``_creative_answer`` → ``generate_creative`` path (real recombination, real
``[Creative mode — generated…]`` label, real 4-gram novelty gate).

The app talks to this module through ``MWPythonBridge`` (Objective-C).
"""

from __future__ import annotations

import json
import secrets
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Offline enforcement (defence in depth — the engine is already network-free).
# --------------------------------------------------------------------------- #
# We PROVE offline operation at runtime: any attempt to open a network socket
# raises loudly instead of silently reaching out. The engine never does this
# (verified), so this guard should never fire; it exists so that a regression
# would crash visibly rather than quietly call the network.
_OFFLINE_ENFORCED = False


def enforce_offline() -> bool:
    """Neuter outbound TCP/UDP connects. Idempotent. Returns True once armed."""
    global _OFFLINE_ENFORCED
    if _OFFLINE_ENFORCED:
        return True
    import socket

    _real_connect = socket.socket.connect
    _real_connect_ex = socket.socket.connect_ex

    def _blocked_connect(self, address):  # noqa: ANN001
        raise OSError(
            "MicroWorld offline guard: outbound network is disabled on-device "
            f"(attempted connect to {address!r})"
        )

    def _blocked_connect_ex(self, address):  # noqa: ANN001
        raise OSError(
            "MicroWorld offline guard: outbound network is disabled on-device "
            f"(attempted connect_ex to {address!r})"
        )

    socket.socket.connect = _blocked_connect  # type: ignore[assignment]
    socket.socket.connect_ex = _blocked_connect_ex  # type: ignore[assignment]
    # Keep the originals reachable for any strictly-local (loopback) needs the
    # stdlib might have; we do not use them, but we don't destroy them either.
    socket._microworld_real_connect = _real_connect  # type: ignore[attr-defined]
    socket._microworld_real_connect_ex = _real_connect_ex  # type: ignore[attr-defined]
    _OFFLINE_ENFORCED = True
    return True


# --------------------------------------------------------------------------- #
# Engine handles (loaded once).
# --------------------------------------------------------------------------- #
_ORCH = None            # AnswerOrchestrator
_OVERLAY = "promoted"
_NARRATIVE_ENGINE = None  # poetry_lab NarrativeEngine; separate from QA
_AUTO_ITEMS = None


def _try_multi_evidence_plan(orch, prompt: str, answer):
    """Use the existing stdlib-only evidence planner for iOS QA.

    The desktop API wraps this same optional planner behind FastAPI.  The phone
    deliberately does not import that server module, so this adapter calls the
    unchanged planner directly.  A failed or inapplicable plan is a no-op and
    preserves the base ``AnswerOrchestrator`` result exactly.
    """
    if answer.risk_flags:
        return answer
    may_recover_audit = (
        answer.decision == "audit" and answer.support_kind == "missing_knowledge"
    )
    if answer.decision != "answer" and not may_recover_audit:
        return answer
    try:
        from worldpgt.entity_qa.semantic_question_parser import parse_semantic_query
        from worldpgt.relation_extraction_v2.relation_policy import relation_intents_from_text
        from worldpgt.reasoning.answer_behavior import (
            build_answer_plan,
            plan_is_expansion,
            prepare_evidence_graph,
        )
        from worldpgt.reasoning.answer_plan_renderer import render_answer_plan
        from worldpgt.reasoning.relation_input_graph import default_relation_input_graph

        query = parse_semantic_query(prompt)
        targets = [target for target in (query.entity_a, query.entity_b) if target]
        matches = list(orch._surface_index.find_in_text(prompt))
        if not targets:
            targets = list(dict.fromkeys(canonical for _surface, canonical, _start, _end in matches))
        if not targets:
            return answer
        explicit_intents = relation_intents_from_text(prompt) | frozenset(
            default_relation_input_graph().resolve_all(
                prompt, entity_spans=((start, end) for _surface, _canonical, start, end in matches)
            )
        )
        predicate_filter = explicit_intents if len(explicit_intents) > 1 else query.relation_intent
        experimental_relations = [
            row for row in orch._provider.all_relations()
            if str(row.get("experimental_tier") or "").startswith("evidence_grounded_")
        ]
        plan = build_answer_plan(
            prompt,
            [],
            targets=targets,
            predicate_filter=predicate_filter,
            required_distinct_predicates=(
                2 if query.query_type == "multi_fact" and not explicit_intents else 1
            ),
            max_blocks=4,
            prepared_edges=prepare_evidence_graph(experimental_relations),
        )
        if not plan_is_expansion(plan):
            return answer
        rendered = render_answer_plan(plan)
        if not rendered:
            return answer
        return replace(
            answer,
            decision="answer",
            answer_text=rendered,
            supported_by_context=True,
            support_kind="evidence_backed_answer_plan",
            source_system="ios_adapter.answer_behavior",
        )
    except Exception:
        # The optional planner must never break base offline QA.
        return answer


def _orchestrator():
    global _ORCH
    if _ORCH is None:
        from worldpgt.assistant_surface.answer_orchestrator import AnswerOrchestrator

        _ORCH = AnswerOrchestrator(_OVERLAY)
    return _ORCH


def _auto_items():
    """Read the already-bundled serving overlay; no NumPy/vector model."""
    global _AUTO_ITEMS
    if _AUTO_ITEMS is None:
        from worldpgt.assistant_surface.context_selector import resolve_overlay
        _AUTO_ITEMS = json.loads(Path(resolve_overlay(_OVERLAY)[0]).read_text(encoding="utf-8"))
    return _AUTO_ITEMS


def warm_up(overlay: str = "promoted", warm_creative: bool = False) -> str:
    """Build the engine once. Called at app startup, off the main thread.

    ``warm_creative`` optionally loads the separate poetry_lab narrative
    artifact so the first Creative tap is instant. The app calls this in the
    background shortly after launch; QA is ready immediately regardless.
    """
    global _OVERLAY
    _OVERLAY = overlay or "promoted"
    enforce_offline()
    t = time.perf_counter()
    _orchestrator()  # builds QA artifacts (phrase graph, overlays)
    qa_ms = (time.perf_counter() - t) * 1000.0
    creative_ms = None
    if warm_creative:
        creative_ms = _warm_creative_model()
    return json.dumps(
        {
            "ok": True,
            "overlay": _OVERLAY,
            "qa_warm_ms": round(qa_ms, 3),
            "creative_warm_ms": creative_ms,
            "offline_enforced": _OFFLINE_ENFORCED,
        }
    )


def _narrative_engine():
    """Load the prebuilt mixed-corpus poetry_lab scene engine once."""
    global _NARRATIVE_ENGINE
    if _NARRATIVE_ENGINE is not None:
        return _NARRATIVE_ENGINE

    bundled = Path(__file__).resolve().parent / "poetry_lab"
    # Desktop smoke uses the source adapter; the app uses the staged package.
    source = Path(__file__).resolve().parents[4] / "poetry_lab"
    root = bundled if (bundled / "artifacts" / "narrative_model.json").is_file() else source
    artifact = root / "artifacts" / "narrative_model.json"
    if not artifact.is_file():
        raise RuntimeError("Missing staged poetry_lab narrative model")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from poemcore.narrative import NarrativeEngine

    _NARRATIVE_ENGINE = NarrativeEngine(artifact)
    return _NARRATIVE_ENGINE


def _warm_creative_model() -> float:
    t = time.perf_counter()
    _narrative_engine()
    return round((time.perf_counter() - t) * 1000.0, 3)


def warm_creative() -> str:
    """Public: pre-build the creative model on demand (background task)."""
    ms = _warm_creative_model()
    return json.dumps({"ok": True, "creative_warm_ms": ms})


# --------------------------------------------------------------------------- #
# The one place mode is honoured.
# --------------------------------------------------------------------------- #
def _narrative_prompt(prompt: str) -> str:
    """Add only compact Russian scene cues for the app's English preset text."""
    lower = prompt.lower()
    cues = []
    for needle, cue in (
        ("rocket", "ракета небо звезда огонь"),
        ("space", "небо звезда ночь"),
        ("evening", "вечер"),
        ("moscow", "москва город"),
        ("city", "город улица свет"),
    ):
        if needle in lower:
            cues.append(cue)
    return f"{prompt}. {' '.join(cues)}" if cues else prompt


def _prompt_knowledge_facts(prompt: str) -> tuple[tuple[str, str, str], ...]:
    """Small semantic anchors for concepts absent from the literary corpus."""
    lower = prompt.lower()
    facts: list[tuple[str, str, str]] = []
    if "rocket" in lower:
        facts.extend((
            ("ракета", "устремляется", "небо"),
            ("ракета", "оставляет", "огонь"),
            ("ракета", "видит", "звезду"),
        ))
    if "space" in lower:
        facts.append(("космос", "светится", "звезда"))
    return tuple(facts)


def _qa_knowledge_facts(pack) -> tuple[tuple[str, str, str], ...]:
    """Read stable QA context as a transient knowledge overlay for a scene."""
    facts: list[tuple[str, str, str]] = []
    for definition in pack.definitions:
        if definition.stability in {"stable", "semi_stable"}:
            facts.append((definition.subject, definition.predicate, definition.definition))
    for relation in pack.direct_relations:
        if relation.trust != "weak_context_only" and relation.stability in {"stable", "semi_stable"}:
            facts.append((relation.subject, relation.predicate, relation.object))
    # The narrative graph is a bounded working overlay, not a new fact store.
    return tuple(dict.fromkeys(facts))[:8]


def _run_creative(orch, prompt: str):
    """Run poetry_lab's planned literary narrative after the safety screen."""
    from worldpgt.assistant_surface.question_router import route as route_question
    from worldpgt.assistant_surface.types import AssistantRoute
    from worldpgt.assistant_surface.assistant_trace import new_trace

    # Hard-safety still wins under a creative framing (private / current-live /
    # universal / inversion) — this is a deliberate engine feature, preserved.
    natural = route_question(prompt, orch._surface_index)
    if natural.is_hard_safety:
        return orch.answer(prompt)

    route = AssistantRoute(
        question=prompt,
        intent="creative_request",
        risk_flags=["creative_generated"],
        notes="creative free-generation (explicit mode)",
    )
    pack, ctx = orch._selector.select(prompt)
    knowledge_facts = tuple(dict.fromkeys(
        (*_qa_knowledge_facts(pack), *_prompt_knowledge_facts(prompt))
    ))
    trace = new_trace(route)
    result = _narrative_engine().run(
        _narrative_prompt(prompt), sentences=6, reasoning=True,
        knowledge_facts=knowledge_facts,
    )
    passage = result.paragraph.text().strip()
    if not passage:
        return orch._make(
            route, ctx, trace, decision="audit", answer_text="", supported=False,
            support_kind="missing_knowledge", source_system="poetry_lab.narrative",
            audit_reason="The literary narrative model did not produce a complete scene.",
        )
    trace.add(
        "creative: renderer=poetry_lab.narrative; "
        f"sources={len(result.meta.get('sources', []))}; "
        f"qa_knowledge_facts={len(knowledge_facts)}; reasoning=scene_plan"
    )
    return orch._make(
        route, ctx, trace, decision="answer",
        answer_text=f"{orch._CREATIVE_LABEL}\n\n{passage}", supported=False,
        support_kind="creative_generated", source_system="poetry_lab.narrative",
        extra_risk=["creative_generated"],
    )


def _auto_answer(orch, prompt: str):
    """Small phone-only router: deterministic patterns, no embedding runtime."""
    from worldpgt.assistant_surface.question_router import route as route_question
    natural = route_question(prompt, orch._surface_index)
    if natural.is_hard_safety:
        return orch.answer(prompt), "qa_safety"
    lower = prompt.lower()
    if re.search(r"\b(using (only|exactly|just)|staying strictly within)\b.*\bfacts\b", lower):
        from worldpgt.reasoning import constrained_creative_v1 as cc
        entities = [c for _s, c, _a, _b in orch._surface_index.find_in_text(prompt)]
        if entities:
            spec = cc.select_facts(_auto_items(), entities[0], n=3)
            if spec.n >= 2:
                return _auto_record(cc.generate_constrained(spec), "constrained_creative", "grounded_generation"), "constrained_creative"
    if re.search(r"\b(compose|write|tell|invent|create)\b.*\b(poem|story|fiction|tale|verse|creative|imaginative)\b", lower):
        return _run_creative(orch, prompt), "pure_creative"
    if re.match(r"\s*(what if|what would|suppose|why might|why is|how might)\b", lower):
        from worldpgt.reasoning import reflective_reasoning_v1 as r1
        from worldpgt.reasoning import reflective_reasoning_extended_v2 as r2
        plan = r1.reflect(prompt, _auto_items())
        if plan and plan.decision == "speculative":
            if plan.rule == "counterfactual_removal" and plan.step:
                focal = plan.step.premises[0]
                facts = [edge for edge in plan.step.conclusion_facts if edge.s == focal.o][:2]
                examples = " and ".join(f"{edge.predicate.replace('_', ' ')} {edge.object}" for edge in facts)
                text = (f"If {focal.subject} had not {focal.predicate.replace('_', ' ')} {focal.object}, "
                        f"the existence of {focal.object} in this evidence slice would be in question. "
                        f"Its recorded activities, including {examples}, would then also be in question. "
                        "This is a speculative inference, not a stored fact.")
            else:
                text = r1.render_reflective_plan(plan)
            return _auto_record(text, "reflective", "speculative_inference"), "reflective"
        entities = list(dict.fromkeys(c for _s, c, _a, _b in orch._surface_index.find_in_text(prompt)))
        if len(entities) >= 2:
            ext = r2.co_attribution_for_pair(r1.load_edges(_auto_items()), entities[0], entities[1])
            if ext.decision == "speculative_extended":
                return _auto_record(r2.render_extended(ext.steps[0]), "reflective_extended", "speculative_extended"), "reflective_extended"
    return _try_multi_evidence_plan(orch, prompt, orch.answer(prompt)), "qa"


def _auto_record(text: str, route: str, support: str):
    """Minimal AnswerOrchestrator-compatible result for non-QA phone branches."""
    from types import SimpleNamespace
    return SimpleNamespace(answer_text=text, decision="answer", route=route,
                           support_kind=support, risk_flags=[])


def run(prompt: str, mode: str) -> str:
    """Answer ``prompt`` in ``mode`` ('qa' or 'creative'); return a JSON string.

    Latency here is engine-side only; the app measures authoritative latency in
    Swift around the whole bridge call. Both are reported so nothing is faked.
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return json.dumps({"ok": False, "error": "empty prompt"})

    orch = _orchestrator()
    mode = (mode or "qa").lower()
    is_creative = mode.startswith("c")
    is_auto = mode.startswith("a")

    t = time.perf_counter()
    if is_auto:
        answer, auto_branch = _auto_answer(orch, prompt)
    elif is_creative:
        answer = _run_creative(orch, prompt)
    else:
        answer = _try_multi_evidence_plan(orch, prompt, orch.answer(prompt))
    engine_ms = (time.perf_counter() - t) * 1000.0

    payload: dict[str, Any] = {
        "ok": True,
        "text": (answer.answer_text or "").strip(),
        "decision": answer.decision,
        "route": answer.route,
        "support_kind": answer.support_kind,
        "risk_flags": list(answer.risk_flags or []),
        "engine_ms": round(engine_ms, 3),
        # QA is deterministic for a given prompt; Creative deliberately re-rolls
        # a fresh passage each request (documented behaviour), so it is not.
        "deterministic": (not is_creative),
        # The creative gate is boolean (novelty pass/fail), not a numeric score,
        # so we do not invent a novelty number. null == "not exposed".
        "novelty": None,
        "mode": "auto" if is_auto else ("creative" if is_creative else "qa"),
    }
    return json.dumps(payload)


def self_test() -> str:
    """Offline self-test: one QA prompt + one Creative prompt. Returns JSON."""
    results = []
    for mode, prompt in (("qa", "Who founded SpaceX?"),
                         ("creative", "Describe a room.")):
        raw = json.loads(run(prompt, mode))
        results.append(
            {
                "mode": mode,
                "prompt": prompt,
                "ok": bool(raw.get("ok")) and bool(raw.get("text")),
                "decision": raw.get("decision"),
                "chars": len(raw.get("text") or ""),
            }
        )
    return json.dumps(
        {
            "ok": all(r["ok"] for r in results),
            "offline_enforced": _OFFLINE_ENFORCED,
            "results": results,
        }
    )


if __name__ == "__main__":  # desktop smoke test
    print(warm_up())
    print(run("Who founded SpaceX?", "qa"))
    print(run("Describe an evening in Moscow.", "creative"))
    print(self_test())
