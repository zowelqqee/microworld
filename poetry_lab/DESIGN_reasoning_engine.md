# Design — a domain-independent reasoning engine (world-state layer)

**Status:** architecture proposal. Nothing here is implemented.

**Research frame.** The language bottleneck is closed: the same core produces
QA answers, verse, and prose. The remaining bottleneck is reasoning — the
system makes *locally* good decisions (activate, rank, pick next) but does not
model *causal chains*: nothing stops a character from acting in two places at
once, appearing without introduction, or vanishing mid-scene. The requested
evolution:

```text
current:  Goal → Activation → Ranking → Language
desired:  Goal → Activation → Reasoning Graph → Hypothesis Graph → Plan → Language
```

Constraints honoured throughout: no transformers, no neural networks, no
probabilistic decoding, the graph architecture stays.

---

## 0. The central finding of the code inspection

**Every mechanism the desired pipeline needs already exists in production
MicroWorld — as six separate, unconnected components.** The proposal invents
no new algorithm; it composes and generalizes existing ones along two axes the
QA domain never needed: **time** (QA answers are timeless; narrative states
change) and **rule domain** (QA rules describe ownership; narrative rules
describe scenes).

| Needed capability | Already exists as | File |
|---|---|---|
| Hypothesis lifecycle (form → test → accept/reject, bounded, stop reasons) | `run_thought_loop`: explicit `ThoughtHypothesis(status, required_roles, reason)`, ≤8 steps, `ThoughtLoopDecision` | `worldpgt/cognition/thought_loop.py` |
| State-transition rules **as data** | Datalog-style unification: `if_pattern` / `unless_pattern` / `then_pattern` / `type_guards`, rules in JSON, *adding a rule requires no code change* | `worldpgt/cognition/rule_interpreter.py` + `worldpgt/knowledge/base_inference_rules.json` |
| Derived facts with provenance | `InferredFact(subject, predicate, object, rule, chain, confidence)` — every inference carries its proof chain | `worldpgt/cognition/inference_engine.py` |
| Contradiction / dependency analysis | counterfactual = remove a fact, re-run inference, diff: `lost_inferences`, `dependent_entities` | `worldpgt/reasoning/counterfactual.py` |
| Entity persistence across turns | `TaskMemory` (active/parked tasks, events); `EntityActivation` (recency, mention counts) — already ported as `DiscourseState` | `worldpgt/cognition/working_memory.py`, `worldpgt/dialogue/state.py` |
| Search for best continuation + audit fallback | enumerate paths → `validate_path` gates → pick best → honest audit when none survive | `worldpgt/multihop_qa/path_planner.py` |
| Scoring with named breakdown (proof of the score) | integer parts, every point named | `worldpgt/dialogue/salience.py` (ported: `poemcore/discourse.py`) |

The desired pipeline maps onto them 1:1:

```text
Goal            → existing (SceneGoal / PoemGoal / question intent)
Activation      → existing (concept_graph.activate, ported)
Reasoning Graph → WorldState + transition rules   (inference_engine ⊕ rule_interpreter, + time)
Hypothesis Graph→ candidate next-states           (thought_loop, generalized test predicate)
Plan            → accepted hypothesis chain       (path_planner shape, beam instead of 2-hop)
Language        → existing (SentencePlan / LineDecision realization)
```

**QA becomes the degenerate case** of the same engine: timeless facts, plan
depth 1, transition rules = `base_inference_rules.json`. That is the concrete
sense in which the engine is reusable across QA, poetry, prose, and future
domains — not "shared philosophy" but literally one engine, different rule
files and depth parameters.

---

## 1. Architecture proposal

One new module family in `poemcore/` (production untouched):

```text
poemcore/world_state.py    WorldState, StateFact, EntityRecord, persistence
poemcore/transitions.py    TransitionRule loading + unification (port of rule_interpreter)
poemcore/hypothesis.py     Hypothesis, generate/score/test (generalized thought_loop)
poemcore/plan_search.py    beam search over hypothesis chains → ReasoningPlan
rules/narrative_rules.json domain rules for prose (data, not code)
rules/poetry_rules.json    domain rules for verse (weaker: image development)
```

Flow per generation step (one sentence / one line):

```text
                       ┌─ activation field (existing) ─┐
Goal ──► WorldState(t) ┤                               ├─► candidate Hypotheses
                       └─ transition rules (data)     ─┘        (bounded K)
                                                                    │
                     score: causal / temporal / persistence /       ▼
                     state-transition / contradiction        beam search (B,D)
                                                                    │
                                                                    ▼
                                            ReasoningPlan = accepted chain
                                                                    │
                     language realizes plan (existing SentencePlan) ▼
                     realization feedback: only *realized* facts commit to WorldState(t+1)
```

The last line is load-bearing: the trigram surface may fail to realize a
decision. The state must therefore be advanced from what was **actually
rendered** (`SentenceRealization.realized` — this feedback channel already
exists in `poemcore/narrative.py`), never from what was merely planned.
Otherwise the world model diverges from the text, which is precisely the bug
class this layer exists to kill.

### Design rules carried over from the transfers so far

- **Advisory scoring before hard gates.** A contradiction is a large penalty,
  not an exception — a slightly-inconsistent continuation still beats an empty
  one (lesson from the morphology layer: score, don't refuse, when a perfect
  candidate may not exist).
- **`unknown` constrains nothing.** Entities without type/gender/state impose
  no penalty (lesson from the entity-type layer).
- **Every score is a named breakdown** (`[("entity_persistence", -4), ...]`) —
  the trace is the proof, as in `salience.py`.
- **Rules are data.** New domain = new JSON file, zero engine changes — exactly
  the property `rule_interpreter.py` already guarantees in production.
- **Audit fallback.** When no hypothesis survives scoring above threshold, the
  planner degrades to the current (stateless) behaviour and *records that it
  did* — the `path_planner` audit shape, inverted for creative mode.

---

## 2. Data structures

```python
# world_state.py ----------------------------------------------------------

@dataclass(frozen=True)
class StateFact:
    """Generalizes InferredFact with a time index.

    predicate examples (narrative): at, present, speaking_to, holds, alive,
    introduced. Domain-specific predicates live in rules, not in code.
    """
    subject: str
    predicate: str
    object: str
    t: int                                  # discrete step the fact holds at
    source: Literal["stated", "inferred", "assumed"]
    rule: str = ""                          # provenance, as in InferredFact
    chain: tuple[tuple[str, str, str], ...] = ()

@dataclass
class EntityRecord:
    """Generalizes EntityActivation/TaskMemory for narrative persistence."""
    name: str
    entity_type: str          # from the ported knowledge layer (person/place/…)
    gender: str | None        # from the morphology layer
    introduced_at: int | None # None = not yet introduced → introduction required
    last_action_at: int
    location: str | None      # current `at` object, denormalized for O(1) checks

@dataclass
class WorldState:
    t: int
    facts: frozenset[StateFact]             # facts holding at t
    entities: dict[str, EntityRecord]
    log: tuple[StateFact, ...]              # full history (for temporal checks)

    def advance(self, delta: tuple[StateFact, ...],
                retracts: frozenset[tuple[str, str]]) -> "WorldState":
        """Frame axiom as code: every fact persists to t+1 unless its
        (subject, predicate) key is retracted by the applied delta."""

# hypothesis.py ------------------------------------------------------------

@dataclass(frozen=True)
class Hypothesis:
    """One candidate continuation of the world. Generalizes ThoughtHypothesis:
    `required_roles` becomes `delta` (facts it asserts) + `retracts`."""
    name: str                               # e.g. "pilat_continues_interrogation"
    delta: tuple[StateFact, ...]
    retracts: frozenset[tuple[str, str]]    # (subject, predicate) keys ended
    focus_entity: str
    action: str
    status: Literal["pending", "accepted", "rejected"] = "pending"
    score: float = 0.0
    breakdown: tuple[tuple[str, float], ...] = ()
    reason: str = ""

@dataclass(frozen=True)
class ReasoningPlan:
    """The inspectable artifact language receives. Chain of accepted
    hypotheses = the causal skeleton of the paragraph/stanza."""
    goal: SceneGoal | PoemGoal
    steps: tuple[Hypothesis, ...]
    final_state: WorldState
    audits: tuple[str, ...]                 # every fallback taken, visible
```

```jsonc
// rules/narrative_rules.json — same schema as base_inference_rules.json,
// plus "retracts" and "temporal": true. Examples for the Pilat case:
[
  { "rule_id": "co_location_v1",
    "if_pattern":  [{"subject": "?A", "predicate": "speaking_to", "object": "?B"}],
    "then_pattern": {"subject": "?A", "predicate": "at", "object": "@loc(?B)"},
    "description": "interlocutors share a location" },

  { "rule_id": "introduction_required_v1",
    "if_pattern":  [{"subject": "?A", "predicate": "acts"}],
    "unless_pattern": [{"subject": "?A", "predicate": "introduced"}],
    "then_pattern": {"violation": "unintroduced_entity", "weight": -6} },

  { "rule_id": "exit_requires_reason_v1",
    "if_pattern":  [{"subject": "?A", "predicate": "present", "t": "t-1"},
                    {"subject": "?A", "predicate": "absent",  "t": "t"}],
    "unless_pattern": [{"subject": "?A", "predicate": "departed"}],
    "then_pattern": {"violation": "unexplained_disappearance", "weight": -8} },

  { "rule_id": "single_location_v1",
    "if_pattern":  [{"subject": "?A", "predicate": "at", "object": "?L1"},
                    {"subject": "?A", "predicate": "at", "object": "?L2"}],
    "constraint": "?L1 != ?L2",
    "then_pattern": {"violation": "bilocation", "weight": -10} }
]
```

Note the two rule kinds sharing one schema: **derivation rules** (produce
StateFacts — the Reasoning Graph) and **violation rules** (produce named
penalties — the contradiction/consistency scorers). Production's
`unless_pattern` and `type_guards` transfer unchanged; `type_guards` plugs
directly into the already-built `entity_types.py` output.

---

## 3. Algorithms

**A. State inference (Reasoning Graph).** After every committed step, run the
ported unification engine over `WorldState.facts` with the derivation rules:
closure of what else must be true (`speaking_to → co-located`, etc.). This *is*
production `run_inference`, scoped to the activation field instead of the whole
overlay, with `t` carried through. Non-recursive single pass, as in production.

**B. Hypothesis generation.** For step t, candidates come from three bounded
sources (cap K ≈ 20–30 total):
  1. *Continuations*: for each active entity, actions reachable in the concept
     graph from its current state facts (activation-gated, existing mechanism);
  2. *Rule completions*: derivation rules with a partially-matched
     `if_pattern` propose the missing fact as a delta ("interrogation ongoing"
     proposes "Pilat asks next question");
  3. *Goal steps*: unrealized goal commitments (planned beats — existing
     `_scene_beats`).

**C. Hypothesis scoring** — the five requested dimensions, each a named part:

| Part | Computation | Reuses |
|---|---|---|
| `causal_consistency` | +w if the delta is derivable by some rule from state (has a proof chain); −w if no rule connects it | rule_interpreter unification |
| `temporal_consistency` | violation rules with `t`-indexed patterns (exit-without-reason, action-before-introduction) | rule engine + `WorldState.log` |
| `entity_persistence` | `EntityRecord.introduced_at`/`last_action_at` checks: unintroduced actor −w, long-dormant entity re-entering without mention −w/2 | TaskMemory/DiscourseState shape |
| `state_transition` | +w if delta actually changes state (not a tautology of current facts) — rewards the story *moving* | set diff vs `facts` |
| `contradiction_risk` | tentatively apply delta, run violation rules + functional-predicate clash check (same subject+predicate, different object, e.g. two `at`); each hit −w. This is `counterfactual.py`'s diff idea run *forward* (add, don't remove) | counterfactual + inference diff |

**D. Plan search.** Beam search over hypothesis chains: width B, depth D
(D = sentences in the paragraph / lines in the stanza). Score of a chain =
Σ step scores + continuity bonus (consecutive steps sharing focus entity).
Deterministic: candidates sorted, ties broken by `seeded_weighted_pick` with
the session seed. Greedy (B=1) is the ablation baseline; the whole point of
the beam is that a locally-worse sentence can enable a globally-consistent
scene. **This replaces "select the next sentence independently" with "search
for the best continuation of the current world state."**

**E. Commit loop (the generalized thought loop).** Per step:
`form (generate K) → test (score, reject contradictions) → accept best →
realize (language) → commit only realized facts → advance state`. Stop
reasons as in production: goal satisfied, depth reached, or
`blocked_no_consistent_continuation` → audit fallback to stateless behaviour.

---

## 4. Expected complexity

Let F = |facts in state| (≈ 10–40 for a scene), R = rules (≈ 10–20), K =
candidates/step (≤ 30), B = beam width (≈ 4), D = depth (≈ 10 sentences).

- State inference: rule unification is a join over F facts per pattern slot;
  patterns are ≤ 2–3 slots → O(R · F²) worst case ≈ 20 · 1600 = trivial.
- Scoring one hypothesis: apply delta (O(F)) + violation pass (O(R · F)) ≈
  sub-millisecond.
- Beam: O(B · K · D) scorings ≈ 4 · 30 · 10 = 1200 scorings per paragraph.

Everything stays deterministic and comfortably under a second per paragraph on
the current corpus — the same order as the existing candidate loop. No
combinatorial risk: unification never runs over the full 23k-node concept
graph, only over the activation-gated state (the same scoping production uses
by running inference over one QA context, not all of Wikipedia).

---

## 5. Implementation roadmap

Smallest-working-version first; each phase independently testable and
A/B-able against the current generator.

| Phase | Deliverable | Est. size | Proves |
|---|---|---|---|
| 1 | `world_state.py`: StateFact/EntityRecord/WorldState + frame-axiom `advance` + unit tests | ~150 lines | state can track the Pilat example by hand |
| 2 | `transitions.py`: port `rule_interpreter` unification (strip worldpgt imports, add `t`), + `narrative_rules.json` (~8 rules incl. the four above) | ~250 lines + data | rules derive co-location; violations fire on bilocation |
| 3 | `hypothesis.py`: generation (3 sources) + 5-part scoring with breakdowns | ~200 lines | scored hypotheses visible in `--show-plan` trace |
| 4 | `plan_search.py`: beam + commit loop + realization feedback; bridge `ReasoningPlan → SentencePlan` (existing) | ~150 lines | full pipeline runs end-to-end on prose |
| 5 | `eval_world_consistency.py` + A/B vs current seeded generator | ~150 lines | the objective: consistency ↑, everything else flat |
| 6 | Generality proof: run the *same engine* with `base_inference_rules.json`-style QA rules at D=1 and diff its derivations against production `run_inference` on a fixture overlay; then a thin poetry rule file | small | one engine, three domains |

Phases 1–4 ≈ 750 lines total. No generator rewrite: the language layer keeps
receiving `SentencePlan`s; only their *source* changes from local heuristics to
the plan.

## 6. Reuse map (existing files)

**Ported (shape preserved, inputs re-domained):**
- `worldpgt/cognition/rule_interpreter.py` → `poemcore/transitions.py` — the
  single largest reuse; the engine is already data-driven, needs only the `t`
  axis and our predicate classes.
- `worldpgt/cognition/thought_loop.py` → `poemcore/hypothesis.py` commit loop
  (form/test/accept/reject/stop_reason; the test predicate generalizes from
  "required_roles ⊆ workspace" to "score > θ and no hard contradiction").
- `worldpgt/cognition/inference_engine.py::InferredFact` → `StateFact`
  (add `t`; keep `rule`/`chain` provenance verbatim).
- `worldpgt/reasoning/counterfactual.py` → forward contradiction check.
- `worldpgt/cognition/working_memory.py::TaskMemory` +
  `worldpgt/dialogue/state.py::EntityActivation` → `EntityRecord`.
- `worldpgt/multihop_qa/path_planner.py` → plan/validate/audit shape of
  `plan_search.py`.

**Reused in place (already in poemcore, unchanged):**
- `concept_graph.py` (activation — candidate source), `discourse.py`
  (salience — tie-breaking within equal consistency), `entity_types.py`
  (fills `type_guards`), `morphology.py` (gender in EntityRecord),
  `reasoning.py::_choose_surface_roles` (realizability bridge to language),
  `narrative.py::SentenceRealization` (the commit-what-was-realized feedback),
  `novelty.py`, `phrase_model.py` (untouched language layer).

**Deliberately not reused:** `support_guard.py` (QA gate polarity — already
established as the domain-defining component), question parsers, `api/`.

## 7. Evaluation plan

New `eval/eval_world_consistency.py`, A/B: current seeded+ranked generator vs
reasoning-planned generator, same prompts, same corpus, same everything else.

**Primary metrics (the objective — must move):**
- *entity-persistence violations* / paragraph: an entity acts without having
  been introduced or after disappearing (countable from WorldState vs text);
- *bilocation / temporal contradictions*: violation-rule hits on the final
  realized text;
- *subject continuity*: share of adjacent sentences whose focus entities are
  connected by a stated relation (vs. mere lexical repetition);
- *causal groundedness*: share of realized sentences whose plan step carries a
  non-empty proof chain (was derivable, not arbitrary);
- *unexplained entity jumps* (existing metric, expected to drop further).

**Guard metrics (must stay flat — regressions here mean the layer leaked):**
novelty (1.00), gender agreement (0), local grammaticality, determinism,
generation time (< a few seconds/paragraph).

**Ablations:** rules off (score→salience only) / persistence off / beam B=1
(greedy) — each isolates one claimed contribution.

**Generality check (the actual research question):** the phase-6 QA regression
— identical engine, QA rule file, depth 1 — must reproduce production
inference results on a fixture. If it does, "reasoning layer reusable across
QA / poetry / prose / future domains" is demonstrated, not asserted.

**Honest failure modes to expect and report:** hand-written rules will be
incomplete (missed violations are silent); SVO noise in fragment roles will
inject wrong StateFacts (mitigated by committing only realized facts, but not
eliminated); beam may prefer static consistency over narrative movement
(the `state_transition` reward exists precisely to counter this — its weight
is the main tuning risk).

## Implementation note (2026-07-10)

The design's broad module boundary remains valid. One concrete mismatch was
confirmed during implementation: `SentenceRealization.realized` records role
labels, not semantic fact triples. The implementation therefore adds a minimal
`realized_facts` channel and accepts a fact only when a finite verb is actually
realized. Fragmentary trigrams may still be rendered through an explicit,
non-committing stateless fallback; they never advance `WorldState`.
