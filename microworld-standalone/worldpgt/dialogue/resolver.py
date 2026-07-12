"""Planner-owned dialogue reference resolution.

``resolve_question`` is the single entry point: a pure function of
(question, DialogueState, entity index) → resolved bindings + an itemized
trace. It runs *before* semantic parsing; its decisions travel to the
untouched parser via :class:`~worldpgt.dialogue.bound_index.BoundSurfaceIndex`.

Guarantees:
  * candidates come only from the session registry (plus entities named
    earlier in the same question) — the resolver cannot introduce an entity
    the dialogue hasn't legitimately touched;
  * type gates are hard filters, applied before scoring;
  * a score margin below ``RESOLVE_MARGIN`` audits — never argmax-and-hope;
  * if *any* slot is unresolved the whole question audits (partial resolution
    is a guess by another name);
  * the optional graph reader is read-only and used in exactly one branch
    (role descriptors), trace-marked ``graph_verified`` when it decides.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace as dc_replace
from typing import Protocol

from worldpgt.dialogue import constants as C
from worldpgt.dialogue import reference_grammar as G
from worldpgt.dialogue.salience import base_salience, slot_salience, type_gate_passes
from worldpgt.dialogue.state import DialogueState, EntityActivation


class GraphReader(Protocol):
    """Narrow read-only view of trusted memory used by role descriptors.

    ``role_holders(anchor, relation)`` returns entities related to *anchor*
    through *relation* in either direction — overlay relations are stored
    both ways ((SpaceX, founded_by, Musk) but (Musk, leader_of, Tesla)), and
    a role descriptor asks "who holds this role for the anchor", not "what
    is the forward object".
    """

    def role_holders(self, anchor: str, relation: str) -> tuple[str, ...]: ...


@dataclass(frozen=True)
class CandidateScore:
    canonical: str
    total: int
    breakdown: tuple[tuple[str, int], ...]

    def to_dict(self) -> dict:
        return {
            "canonical": self.canonical,
            "total": self.total,
            "breakdown": [list(p) for p in self.breakdown],
        }


@dataclass(frozen=True)
class SlotResolution:
    slot: G.ReferenceSlot
    outcome: str  # "resolved" | "resolved_set" | "ambiguous" | "no_candidate"
    entities: tuple[str, ...]
    candidates: tuple[CandidateScore, ...]
    margin: int | None
    strategy: str

    def to_dict(self) -> dict:
        return {
            "slot": self.slot.to_dict(),
            "outcome": self.outcome,
            "entities": list(self.entities),
            "candidates": [c.to_dict() for c in self.candidates],
            "margin": self.margin,
            "strategy": self.strategy,
        }


@dataclass(frozen=True)
class BoundSpan:
    start: int
    end: int
    surface: str
    canonicals: tuple[str, ...]
    possessive: bool = False


@dataclass(frozen=True)
class DialogueDirectives:
    answer_style: str = "normal"  # "normal" | "followup"
    exclude_objects: tuple[str, ...] = ()
    exclusion_subject: str | None = None
    exclusion_relation: str | None = None
    selective_set: tuple[str, ...] = ()
    topic_op: tuple[str, ...] = ("keep",)
    reformulated_question: str | None = None  # topic-shift transport only

    def to_dict(self) -> dict:
        return {
            "answer_style": self.answer_style,
            "exclude_objects": list(self.exclude_objects),
            "exclusion_subject": self.exclusion_subject,
            "exclusion_relation": self.exclusion_relation,
            "selective_set": list(self.selective_set),
            "topic_op": list(self.topic_op),
            "reformulated_question": self.reformulated_question,
        }


@dataclass(frozen=True)
class ResolvedQuestion:
    raw_question: str
    outcome: str  # "no_slots" | "resolved" | "unresolved"
    slots: tuple[SlotResolution, ...] = ()
    bindings: tuple[BoundSpan, ...] = ()
    directives: DialogueDirectives = field(default_factory=DialogueDirectives)

    @property
    def resolved_references(self) -> list[str]:
        """Formatted for the API's existing resolved_references field."""
        out = []
        for res in self.slots:
            if res.outcome in ("resolved", "resolved_set") and res.slot.surface:
                out.append(f"[{res.slot.surface} → {', '.join(res.entities)}]")
        return out

    def to_dict(self) -> dict:
        return {
            "raw_question": self.raw_question,
            "outcome": self.outcome,
            "slots": [s.to_dict() for s in self.slots],
            "bindings": [
                {
                    "start": b.start,
                    "end": b.end,
                    "surface": b.surface,
                    "canonicals": list(b.canonicals),
                    "possessive": b.possessive,
                }
                for b in self.bindings
            ],
            "directives": self.directives.to_dict(),
        }


# ────────────────────────────────────────────────────────────────────────────


def resolve_question(
    question: str,
    state: DialogueState,
    index,
    graph_reader: GraphReader | None = None,
) -> ResolvedQuestion:
    parse = G.detect_slots(question, index)

    if parse.topic_shift_surface is not None:
        canonical = index.resolve(parse.topic_shift_surface)
        return ResolvedQuestion(
            raw_question=question,
            outcome="resolved",
            slots=(
                SlotResolution(
                    slot=G.ReferenceSlot(
                        surface=parse.topic_shift_surface, start=0, end=0,
                        ref_class="topic_shift", type_gate=None,
                    ),
                    outcome="resolved",
                    entities=(canonical,),
                    candidates=(CandidateScore(canonical, 0, (("topic_shift", 0),)),),
                    margin=None,
                    strategy="topic_shift",
                ),
            ),
            directives=DialogueDirectives(
                answer_style="followup",
                topic_op=("set", canonical),
                reformulated_question=f"Tell me about {canonical}.",
            ),
        )

    if not parse.slots:
        return ResolvedQuestion(raw_question=question, outcome="no_slots")

    same_question = _same_question_activations(question, state, index)
    resolutions: list[SlotResolution] = []
    bindings: list[BoundSpan] = []
    selective_set: tuple[str, ...] = ()
    topic_op: tuple[str, ...] = ("keep",)
    answer_style = "normal"

    for slot in parse.slots:
        if slot.ref_class == G.ELLIPTICAL:
            res = _resolve_elliptical(slot, state)
            answer_style = "followup"
        elif slot.ref_class == G.DEMONSTRATIVE_BARE:
            res = _resolve_bare_demonstrative(slot, state)
        elif slot.ref_class == G.ROLE_DESCRIPTOR:
            res = _resolve_role_descriptor(slot, state, index, graph_reader, same_question)
        elif slot.ref_class == G.PRONOUN_PLURAL:
            res = _resolve_plural(slot, state, index, same_question)
        elif slot.ref_class == G.CONTRASTIVE:
            res = _resolve_contrastive(slot, state, index)
        elif slot.ref_class == G.SELECTIVE:
            res = _resolve_selective(slot, state, index, same_question)
        else:  # singular pronouns and typed demonstratives
            res = _resolve_singular(slot, state, index, same_question)

        resolutions.append(res)

        if res.outcome == "resolved":
            if slot.ref_class == G.ELLIPTICAL:
                # Virtual mention appended past the end of the text; the
                # parser picks it up as the sole entity of the question.
                bindings.append(BoundSpan(len(question), len(question), "", res.entities))
            elif slot.ref_class == G.CONTRASTIVE:
                bindings.append(BoundSpan(
                    slot.start, slot.end, slot.surface, res.entities, slot.possessive))
                topic_op = ("set", res.entities[0])  # contrastive pivot
            else:
                bindings.append(BoundSpan(
                    slot.start, slot.end, slot.surface, res.entities, slot.possessive))
        elif res.outcome == "resolved_set":
            if slot.ref_class == G.SELECTIVE:
                selective_set = res.entities
            else:  # plural pronoun — bind both at the same span
                bindings.append(BoundSpan(
                    slot.start, slot.end, slot.surface, res.entities, slot.possessive))

    unresolved = any(r.outcome in ("ambiguous", "no_candidate") for r in resolutions)
    outcome = "unresolved" if unresolved else "resolved"
    if unresolved:
        bindings = []
        selective_set = ()
        topic_op = ("keep",)

    directives = DialogueDirectives(
        answer_style=answer_style,
        selective_set=selective_set,
        topic_op=topic_op,
    )
    if outcome == "resolved" and parse.has_exclusion:
        directives = _with_exclusions(directives, parse, resolutions, state)

    return ResolvedQuestion(
        raw_question=question,
        outcome=outcome,
        slots=tuple(resolutions),
        bindings=tuple(bindings),
        directives=directives,
    )


# ── Strategies ───────────────────────────────────────────────────────────────


def _resolve_singular(
    slot: G.ReferenceSlot,
    state: DialogueState,
    index,
    same_question: dict[str, EntityActivation],
) -> SlotResolution:
    scored = _score_gated(slot, state, index, same_question, slot.type_gate)
    return _pick_by_margin(slot, scored, strategy="salience")


def _resolve_role_descriptor(
    slot: G.ReferenceSlot,
    state: DialogueState,
    index,
    graph_reader: GraphReader | None,
    same_question: dict[str, EntityActivation],
) -> SlotResolution:
    # Pass 1: entity that entered the dialogue as the answer to
    # role_relation(anchor) for a currently-active anchor.
    active_anchors = frozenset(
        a.canonical
        for a in state.entities.values()
        if base_salience(a, state)[0] > C.ACTIVATION_THRESHOLD
    ) | ({state.active_topic} if state.active_topic else frozenset())

    role_matched = [
        a for a in _candidate_pool(state, same_question).values()
        if type_gate_passes(_etype(a, state, index), slot.type_gate)
        and any(rel == slot.role_relation and anchor in active_anchors
                for rel, anchor, _t in a.dialogue_roles)
    ]
    if role_matched:
        scored = _score_activations(slot, state, same_question, role_matched)
        return _pick_by_margin(slot, scored, strategy="dialogue_role")

    # Pass 2: graph-verified — a read of trusted memory used only to *select*
    # among existing entities; it can never assert anything.
    if graph_reader is not None and state.active_topic and slot.role_relation:
        holders = graph_reader.role_holders(state.active_topic, slot.role_relation)
        known = [o for o in holders if index.resolve(o) is not None]
        if len(known) == 1:
            cand = CandidateScore(known[0], 0, (("graph_verified", 0),))
            return SlotResolution(
                slot=slot, outcome="resolved", entities=(known[0],),
                candidates=(cand,), margin=None, strategy="graph_verified",
            )
        if len(known) > 1:
            cands = tuple(
                CandidateScore(o, 0, (("graph_verified", 0),)) for o in sorted(known)
            )
            return SlotResolution(
                slot=slot, outcome="ambiguous", entities=(),
                candidates=cands, margin=0, strategy="graph_verified",
            )

    return SlotResolution(
        slot=slot, outcome="no_candidate", entities=(), candidates=(),
        margin=None, strategy="dialogue_role",
    )


def _resolve_plural(
    slot: G.ReferenceSlot,
    state: DialogueState,
    index,
    same_question: dict[str, EntityActivation],
) -> SlotResolution:
    scored = [
        c for c in _score_gated(slot, state, index, same_question, slot.type_gate)
        if c.total > C.ACTIVATION_THRESHOLD
    ]
    if len(scored) == C.PLURAL_SIZE:
        return SlotResolution(
            slot=slot, outcome="resolved_set",
            entities=tuple(c.canonical for c in scored),
            candidates=tuple(scored), margin=None, strategy="plural_active",
        )
    outcome = "no_candidate" if not scored else "ambiguous"
    return SlotResolution(
        slot=slot, outcome=outcome, entities=(),
        candidates=tuple(scored), margin=None, strategy="plural_active",
    )


def _resolve_contrastive(
    slot: G.ReferenceSlot, state: DialogueState, index
) -> SlotResolution:
    """"The other one": contrasts entities that have been *under discussion*
    (current or former topics), excluding the current focus. Inherits the
    focus's type when the surface names no noun."""

    focus = _focus(state)
    gate = slot.type_gate
    if gate is None and focus is not None:
        focus_act = state.entities.get(focus)
        if focus_act is not None:
            focus_type = _etype(focus_act, state, index)
            if focus_type is not None:
                gate = frozenset({focus_type})

    pool = [
        a for a in state.entities.values()
        if (a.canonical == state.active_topic or a.was_topic)
        and a.canonical != focus
        and (gate is None or type_gate_passes(_etype(a, state, index), gate))
        and base_salience(a, state)[0] > C.ACTIVATION_THRESHOLD
    ]
    scored = _score_activations(slot, state, {}, pool)
    if len(scored) == 1:
        return SlotResolution(
            slot=slot, outcome="resolved", entities=(scored[0].canonical,),
            candidates=tuple(scored), margin=None, strategy="contrastive",
        )
    outcome = "no_candidate" if not scored else "ambiguous"
    return SlotResolution(
        slot=slot, outcome=outcome, entities=(),
        candidates=tuple(scored), margin=None, strategy="contrastive",
    )


def _resolve_selective(
    slot: G.ReferenceSlot,
    state: DialogueState,
    index,
    same_question: dict[str, EntityActivation],
) -> SlotResolution:
    scored = [
        c for c in _score_gated(slot, state, index, same_question, slot.type_gate)
        if c.total > C.ACTIVATION_THRESHOLD
    ]
    if 2 <= len(scored) <= C.SELECTIVE_MAX:
        return SlotResolution(
            slot=slot, outcome="resolved_set",
            entities=tuple(c.canonical for c in scored),
            candidates=tuple(scored), margin=None, strategy="selective_active",
        )
    outcome = "no_candidate" if not scored else "ambiguous"
    return SlotResolution(
        slot=slot, outcome=outcome, entities=(),
        candidates=tuple(scored), margin=None, strategy="selective_active",
    )


def _resolve_elliptical(slot: G.ReferenceSlot, state: DialogueState) -> SlotResolution:
    topic = state.active_topic
    if topic is None:
        return SlotResolution(
            slot=slot, outcome="no_candidate", entities=(), candidates=(),
            margin=None, strategy="elliptical_topic",
        )
    return SlotResolution(
        slot=slot, outcome="resolved", entities=(topic,),
        candidates=(CandidateScore(topic, C.ACTIVE_TOPIC, (("active_topic", C.ACTIVE_TOPIC),)),),
        margin=None, strategy="elliptical_topic",
    )


def _resolve_bare_demonstrative(slot: G.ReferenceSlot, state: DialogueState) -> SlotResolution:
    focus = _focus(state)
    if focus is None:
        return SlotResolution(
            slot=slot, outcome="no_candidate", entities=(), candidates=(),
            margin=None, strategy="focus",
        )
    return SlotResolution(
        slot=slot, outcome="resolved", entities=(focus,),
        candidates=(CandidateScore(focus, 0, (("focus", 0),)),),
        margin=None, strategy="focus",
    )


# ── Shared helpers ───────────────────────────────────────────────────────────


def _focus(state: DialogueState) -> str | None:
    """The single entity most recently in focus: the last answer's primary
    (its first answer-object — the new information — else its subject),
    falling back to the active topic."""

    if state.last_answer is not None and state.last_answer.primary is not None:
        return state.last_answer.primary
    return state.active_topic


def _candidate_pool(
    state: DialogueState,
    same_question: dict[str, EntityActivation],
) -> dict[str, EntityActivation]:
    pool = dict(state.entities)
    for canonical, act in same_question.items():
        pool.setdefault(canonical, act)
    return pool


def _same_question_activations(
    question: str, state: DialogueState, index
) -> dict[str, EntityActivation]:
    """Ephemeral activations for entities named earlier in this same question
    (intra-turn anaphora: "What does SpaceX build and who founded it?")."""

    out: dict[str, EntityActivation] = {}
    for _surface, canonical, _start, _end in index.find_in_text(question):
        if canonical in out:
            continue
        existing = state.entities.get(canonical)
        if existing is not None:
            out[canonical] = existing
            continue
        out[canonical] = EntityActivation(
            canonical=canonical,
            entity_type=index.entity_type(canonical) or state.entity_type_hints.get(canonical),
            introduced_turn=state.turn_counter + 1,
            last_mention_confirmed=state.confirmed_counter,
            mention_count=1,
            introduction_source="user_named",
        )
    return out


def _etype(activation: EntityActivation, state: DialogueState, index) -> str | None:
    """Entity type for gating: session hint first (covers e.g. web-search
    entities the static index can't type), then the trusted index. Both are
    deterministic reads; nothing is stored back."""

    if activation.entity_type is not None:
        return activation.entity_type
    hint = state.entity_type_hints.get(activation.canonical)
    if hint is not None:
        return hint
    return index.entity_type(activation.canonical)


def _score_gated(
    slot: G.ReferenceSlot,
    state: DialogueState,
    index,
    same_question: dict[str, EntityActivation],
    gate: frozenset[str] | None,
) -> list[CandidateScore]:
    pool = [
        a for a in _candidate_pool(state, same_question).values()
        if type_gate_passes(_etype(a, state, index), gate)
    ]
    return _score_activations(slot, state, same_question, pool)


def _score_activations(
    slot: G.ReferenceSlot,
    state: DialogueState,
    same_question: dict[str, EntityActivation],
    pool: list[EntityActivation],
) -> list[CandidateScore]:
    same_named = frozenset(same_question)
    active_anchors = frozenset(state.entities) | same_named
    scored = []
    for act in pool:
        total, breakdown = slot_salience(
            act, state,
            ref_class=slot.ref_class,
            role_relation=slot.role_relation,
            role_anchors_active=active_anchors,
            same_question_named=same_named,
        )
        scored.append(CandidateScore(act.canonical, total, breakdown))
    # Score descending; canonical ascending purely to make the order total —
    # a name never decides a resolution, only the margin rule does.
    scored.sort(key=lambda c: (-c.total, c.canonical))
    return scored


def _pick_by_margin(
    slot: G.ReferenceSlot,
    scored: list[CandidateScore],
    *,
    strategy: str,
) -> SlotResolution:
    if not scored:
        return SlotResolution(
            slot=slot, outcome="no_candidate", entities=(), candidates=(),
            margin=None, strategy=strategy,
        )
    if len(scored) == 1:
        return SlotResolution(
            slot=slot, outcome="resolved", entities=(scored[0].canonical,),
            candidates=tuple(scored), margin=None, strategy=strategy,
        )
    margin = scored[0].total - scored[1].total
    if margin >= C.RESOLVE_MARGIN:
        return SlotResolution(
            slot=slot, outcome="resolved", entities=(scored[0].canonical,),
            candidates=tuple(scored), margin=margin, strategy=strategy,
        )
    return SlotResolution(
        slot=slot, outcome="ambiguous", entities=(),
        candidates=tuple(scored), margin=margin, strategy=strategy,
    )


def _with_exclusions(
    directives: DialogueDirectives,
    parse: G.GrammarParse,
    resolutions: list[SlotResolution],
    state: DialogueState,
) -> DialogueDirectives:
    """"What else did he found?" — subtract triples already surfaced this
    session. Directionless: for (s, p, o) with p == relation, if the bound
    subject is s the exclusion is o, and vice versa."""

    relation = parse.relation_intent
    if relation is None:
        return directives

    subject: str | None = None
    for res in resolutions:
        if res.outcome == "resolved" and len(res.entities) == 1:
            subject = res.entities[0]
            break
    if subject is None:
        subject = _focus(state)
    if subject is None:
        return directives

    excluded: list[str] = []
    for s, p, o, _turn in state.mentioned_relations:
        if p != relation:
            continue
        if s == subject and o not in excluded:
            excluded.append(o)
        elif o == subject and s not in excluded:
            excluded.append(s)

    return dc_replace(
        directives,
        exclude_objects=tuple(excluded),
        exclusion_subject=subject,
        exclusion_relation=relation,
    )
