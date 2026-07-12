"""Deterministic dialogue-context benchmark.

Runs scripted multi-turn sessions from a JSON fixture against the real
resolver + DialogueState, with a tiny fixture-defined world standing in for
the overlay (so the benchmark is independent of on-disk artifacts). Expected
values are asserted against *traces* — bindings, outcomes, margins, exact
integer scores, topic — not just answers.

Hard gates, all of which fail the run:
  1. false-resolution rate must be 0 (any wrong binding fails; a justified
     audit never does);
  2. determinism: every session runs twice and traces must be identical;
  3. replay: ``DialogueState.replay(records)`` must equal the live end state.

Usage:  python3 -m worldpgt.benchmarks.dialogue_benchmark [fixture.json]
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from worldpgt.cognition.phrase_graph import default_phrase_graph
from worldpgt.dialogue.bound_index import BoundSurfaceIndex
from worldpgt.dialogue.reference_grammar import _relation_intent
from worldpgt.dialogue.resolver import ResolvedQuestion, resolve_question
from worldpgt.dialogue.state import AnswerEntity, DialogueState, TurnRecord

DEFAULT_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "dialogue_sessions_v1.json"

_DEFINITION_RE = re.compile(
    r"^\s*(?:tell\s+me\s+(?:more\s+)?about|what\s+is|who\s+is|расскажи\s+(?:про|о|об))\s+(.+?)\s*[?.!]*\s*$",
    re.IGNORECASE,
)


# ── Fixture world ────────────────────────────────────────────────────────────


class WorldIndex:
    """EntitySurfaceIndex read interface over a fixture-defined world."""

    def __init__(self, entities: dict[str, str | None], surfaces: dict[str, str]) -> None:
        self._types = dict(entities)
        self._surface_to_canonical: dict[str, str] = {}
        for canonical in entities:
            self._surface_to_canonical[canonical.lower()] = canonical
        for surface, canonical in surfaces.items():
            self._surface_to_canonical.setdefault(surface.lower(), canonical)
        ordered = sorted(self._surface_to_canonical, key=len, reverse=True)
        self._combined_re = (
            re.compile(
                r"(?<!\w)(?:" + "|".join(re.escape(s) for s in ordered) + r")(?!\w)",
                re.IGNORECASE,
            )
            if ordered
            else None
        )

    def resolve(self, surface: str) -> str | None:
        normalized = re.sub(r"\s+", " ", (surface or "").strip().lower())
        return self._surface_to_canonical.get(normalized)

    def entity_type(self, canonical: str) -> str | None:
        return self._types.get(canonical)

    def find_in_text(self, text: str) -> list[tuple[str, str, int, int]]:
        if self._combined_re is None:
            return []
        return [
            (m.group(0), self._surface_to_canonical[m.group(0).lower()], m.start(), m.end())
            for m in self._combined_re.finditer(text)
        ]

    def has_surface(self, text: str) -> bool:
        return bool(self.find_in_text(text))


class WorldGraph:
    def __init__(self, facts: list[tuple[str, str, str]]) -> None:
        self.facts = [tuple(f) for f in facts]

    def objects_for(self, subject: str, relation: str) -> tuple[str, ...]:
        return tuple(o for s, r, o in self.facts if s == subject and r == relation)

    def subjects_for(self, obj: str, relation: str) -> tuple[str, ...]:
        return tuple(s for s, r, o in self.facts if o == obj and r == relation)

    def role_holders(self, anchor: str, relation: str) -> tuple[str, ...]:
        return self.objects_for(anchor, relation) + self.subjects_for(anchor, relation)


# ── Stub planner ─────────────────────────────────────────────────────────────
#
# The benchmark exercises the resolver and state lifecycle, not the real
# planner: answers are computed by a transparent fact lookup over the fixture
# world, just enough to drive commits the way the serving path will.


def _stub_turn_record(
    question: str,
    resolved: ResolvedQuestion,
    index: WorldIndex,
    graph: WorldGraph,
    hints: dict[str, str],
) -> TurnRecord:
    hints_tuple = tuple(sorted(hints.items()))
    user_named = tuple(dict.fromkeys(
        canonical for _s, canonical, _st, _e in index.find_in_text(question)
    ))
    resolved_referents = tuple(
        (res.slot.ref_class, res.entities[0])
        for res in resolved.slots
        if res.outcome == "resolved" and len(res.entities) == 1 and res.slot.ref_class != "topic_shift"
    )

    if resolved.outcome == "unresolved":
        return TurnRecord(
            question=question, user_named=user_named,
            answer_decision="audit", entity_type_hints=hints_tuple,
        )

    effective_question = resolved.directives.reformulated_question or question
    # Mirror the serving-path transport: bound spans make the resolver's
    # decisions visible to downstream lookup, exactly as the orchestrator will
    # see them through BoundSurfaceIndex.
    lookup = (
        BoundSurfaceIndex(index, resolved.bindings) if resolved.bindings else index
    )

    definition = _DEFINITION_RE.match(effective_question)
    if definition:
        subject = lookup.resolve(definition.group(1))
        if subject is None:
            return TurnRecord(
                question=question, user_named=user_named,
                answer_decision="audit", entity_type_hints=hints_tuple,
            )
        return TurnRecord(
            question=question,
            user_named=user_named,
            answer_entities=(AnswerEntity(subject, "answer_subject"),),
            topic_op=("set", subject),
            question_subject=subject,
            resolved_referents=resolved_referents,
            entity_type_hints=hints_tuple,
        )

    relation = _relation_intent(effective_question)

    if resolved.directives.selective_set and relation:
        text = effective_question.lower()
        passing = tuple(
            e for e in resolved.directives.selective_set
            if any(s == e and r == relation and o.lower() in text for s, r, o in graph.facts)
        )
        if passing:
            return TurnRecord(
                question=question,
                user_named=user_named,
                answer_entities=tuple(AnswerEntity(e, "answer_subject") for e in passing),
                relation_intent=relation,
                resolved_referents=resolved_referents,
                entity_type_hints=hints_tuple,
                topic_op=resolved.directives.topic_op,
            )
        return TurnRecord(
            question=question, user_named=user_named,
            answer_decision="audit", relation_intent=relation,
            entity_type_hints=hints_tuple,
        )

    bound = [c for span in resolved.bindings for c in span.canonicals]
    subject = bound[0] if bound else (user_named[0] if user_named else None)

    if relation is None or subject is None:
        return TurnRecord(
            question=question, user_named=user_named,
            answer_decision="audit", relation_intent=relation,
            resolved_referents=resolved_referents, entity_type_hints=hints_tuple,
        )

    objects = list(graph.objects_for(subject, relation))
    inverse = False
    if not objects:
        objects = list(graph.subjects_for(subject, relation))
        inverse = True

    excluded = set(resolved.directives.exclude_objects)
    remaining = [o for o in objects if o not in excluded]

    if not objects:
        return TurnRecord(
            question=question, user_named=user_named,
            answer_decision="audit", question_subject=subject,
            relation_intent=relation, resolved_referents=resolved_referents,
            entity_type_hints=hints_tuple,
        )
    if not remaining:
        # Exhausted by exclusion — the honest "nothing further in memory".
        return TurnRecord(
            question=question, user_named=user_named,
            answer_decision="no", question_subject=subject,
            relation_intent=relation, resolved_referents=resolved_referents,
            entity_type_hints=hints_tuple,
        )

    answer_entities = [AnswerEntity(subject, "answer_subject")]
    surfaced = []
    for obj in remaining:
        answer_entities.append(AnswerEntity(obj, "answer_object", relation, subject))
        surfaced.append((subject, relation, obj) if not inverse else (obj, relation, subject))

    return TurnRecord(
        question=question,
        user_named=user_named,
        answer_entities=tuple(answer_entities),
        surfaced_relations=tuple(surfaced),
        question_subject=subject,
        relation_intent=relation,
        resolved_referents=resolved_referents,
        entity_type_hints=hints_tuple,
        topic_op=resolved.directives.topic_op,
    )


# ── Assertions ───────────────────────────────────────────────────────────────


def _check_turn(
    session_id: str,
    turn_no: int,
    expect: dict,
    resolved: ResolvedQuestion,
    record: TurnRecord,
    state: DialogueState,
) -> list[str]:
    failures: list[str] = []

    def fail(message: str) -> None:
        failures.append(f"{session_id}#t{turn_no}: {message}")

    if "outcome" in expect and resolved.outcome != expect["outcome"]:
        fail(f"outcome {resolved.outcome!r} != expected {expect['outcome']!r}")

    if "bindings" in expect:
        actual = {
            res.slot.surface: list(res.entities)
            for res in resolved.slots
            if res.outcome in ("resolved", "resolved_set")
        }
        if actual != expect["bindings"]:
            fail(f"bindings {actual!r} != expected {expect['bindings']!r}")

    if "unresolved_slot" in expect:
        want = expect["unresolved_slot"]
        matching = [
            res for res in resolved.slots
            if res.slot.surface == want.get("surface", res.slot.surface)
            and res.outcome in ("ambiguous", "no_candidate")
        ]
        if not matching:
            fail(f"expected unresolved slot {want!r}, slots: "
                 f"{[(r.slot.surface, r.outcome) for r in resolved.slots]!r}")
        elif "slot_outcome" in want and matching[0].outcome != want["slot_outcome"]:
            fail(f"unresolved slot outcome {matching[0].outcome!r} != {want['slot_outcome']!r}")

    if "margin_min" in expect:
        margins = [res.margin for res in resolved.slots if res.margin is not None]
        if margins and margins[0] < expect["margin_min"]:
            fail(f"margin {margins[0]} < required {expect['margin_min']}")

    if "top_scores" in expect:
        scores: dict[str, int] = {}
        for res in resolved.slots:
            for cand in res.candidates:
                scores.setdefault(cand.canonical, cand.total)
        for canonical, want_score in expect["top_scores"].items():
            if scores.get(canonical) != want_score:
                fail(f"score[{canonical}] {scores.get(canonical)!r} != expected {want_score}")

    if "exclude" in expect:
        actual_excluded = sorted(resolved.directives.exclude_objects)
        if actual_excluded != sorted(expect["exclude"]):
            fail(f"exclude {actual_excluded!r} != expected {sorted(expect['exclude'])!r}")

    if "selective_set" in expect:
        if sorted(resolved.directives.selective_set) != sorted(expect["selective_set"]):
            fail(f"selective_set {sorted(resolved.directives.selective_set)!r} "
                 f"!= expected {sorted(expect['selective_set'])!r}")

    if "strategy" in expect:
        strategies = [res.strategy for res in resolved.slots]
        if expect["strategy"] not in strategies:
            fail(f"strategy {expect['strategy']!r} not in {strategies!r}")

    if "decision" in expect and record.answer_decision != expect["decision"]:
        fail(f"decision {record.answer_decision!r} != expected {expect['decision']!r}")

    if "topic_after" in expect and state.active_topic != expect["topic_after"]:
        fail(f"topic {state.active_topic!r} != expected {expect['topic_after']!r}")

    if "answer_style" in expect and resolved.directives.answer_style != expect["answer_style"]:
        fail(f"answer_style {resolved.directives.answer_style!r} "
             f"!= expected {expect['answer_style']!r}")

    return failures


# ── Runner ───────────────────────────────────────────────────────────────────


@dataclass
class SessionResult:
    session_id: str
    turns: int
    failures: list[str] = field(default_factory=list)
    traces: list[dict] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass
class BenchmarkReport:
    sessions: list[SessionResult]
    resolver_calls: int = 0
    resolver_total_ns: int = 0

    @property
    def passed(self) -> bool:
        return all(s.passed for s in self.sessions)

    @property
    def failures(self) -> list[str]:
        return [f for s in self.sessions for f in s.failures]

    def summary(self) -> str:
        total = len(self.sessions)
        ok = sum(1 for s in self.sessions if s.passed)
        mean_us = (
            self.resolver_total_ns / self.resolver_calls / 1000
            if self.resolver_calls
            else 0.0
        )
        lines = [
            f"dialogue benchmark: {ok}/{total} sessions passed, "
            f"{self.resolver_calls} resolver calls, mean {mean_us:.1f}µs/call",
        ]
        lines.extend(self.failures)
        return "\n".join(lines)


def _run_session_once(session: dict) -> tuple[SessionResult, list[TurnRecord], DialogueState, int]:
    world = session.get("world", {})
    index = WorldIndex(world.get("entities", {}), world.get("surfaces", {}))
    graph = WorldGraph([tuple(f) for f in world.get("facts", [])])

    state = DialogueState()
    records: list[TurnRecord] = []
    result = SessionResult(session_id=session["id"], turns=len(session["turns"]))
    resolver_ns = 0

    for turn_no, turn in enumerate(session["turns"], start=1):
        question = turn["q"]
        started = time.perf_counter_ns()
        resolved = resolve_question(question, state, index, graph)
        resolver_ns += time.perf_counter_ns() - started

        record = _stub_turn_record(question, resolved, index, graph, turn.get("hints", {}))
        state.commit(record)
        records.append(record)

        result.traces.append(resolved.to_dict())
        result.failures.extend(
            _check_turn(session["id"], turn_no, turn.get("expect", {}), resolved, record, state)
        )

    return result, records, state, resolver_ns


def run_session(session: dict) -> tuple[SessionResult, int, int]:
    first, records, live_state, ns_first = _run_session_once(session)
    second, _records2, _state2, ns_second = _run_session_once(session)

    # Gate 2: byte-identical determinism across runs.
    if json.dumps(first.traces, sort_keys=True) != json.dumps(second.traces, sort_keys=True):
        first.failures.append(f"{session['id']}: NON-DETERMINISTIC traces across identical runs")

    # Gate 3: fold/replay property.
    replayed = DialogueState.replay(records)
    if replayed.to_dict() != live_state.to_dict():
        first.failures.append(f"{session['id']}: replay(records) != live state")

    calls = 2 * len(session["turns"])
    return first, calls, ns_first + ns_second


def run_benchmark(fixture_path: Path | str = DEFAULT_FIXTURE_PATH) -> BenchmarkReport:
    # Warm the trained phrase graph before any timed resolver call. Its one-
    # time training cost (spaCy parsing community-context sentences) is a
    # cold-start build, the same category as the spaCy-model-load or index-
    # warm-up costs excluded elsewhere in this codebase -- not part of the
    # per-call latency budget being measured here.
    default_phrase_graph()
    data = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    report = BenchmarkReport(sessions=[])
    for session in data["sessions"]:
        result, calls, ns = run_session(session)
        report.sessions.append(result)
        report.resolver_calls += calls
        report.resolver_total_ns += ns
    return report


def main() -> int:
    import sys

    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_FIXTURE_PATH
    report = run_benchmark(path)
    print(report.summary())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
