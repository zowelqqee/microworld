# Reflective reasoning core v1 — design

**Status: plan only. No code, no generation.**
A reasoning-core-level "reflective mode" — generation-time labelling of speculative inference,
**not** post-hoc classification of finished text. This is an isolated research path parallel
to the QA answer/audit path, the constrained-creative mode, and the (closed) informed-reflection
mode. It modifies nothing yet.

> **Recommendation up front (details in §6):** the *labelling* mechanism is architecturally
> tractable and low-risk — that is precisely why this is a different task from the failed
> informed-reflection branch. But there is a **distinct keystone risk here — inference-rule
> soundness** — and it deserves the same cheap pilot gate the last two branches used. Do a
> ~1–2 day inference-rule pilot **before** the full build.

---

## 0. Why this is a different problem, not the failed one renamed

The informed-reflection pilot (`artifacts/informed_reflection_v1/pilot_report.md`) failed for
one structural reason: it tried to **classify already-generated, unlabelled text** by surface
markers, and the same hedge word (`perhaps`, `it seems`, `if`) legitimately scopes both a fact
and a speculation — you cannot recover scope from the surface without semantic parsing. It was
*reverse-engineering* origin after the fact.

This task inverts the information flow. The reasoning core **decides, at each plan-construction
step, what it is doing** — (a) pulling a verified edge from the graph, or (b) constructing a
speculative inference on top of graph facts. The `grounded`/`speculative` label is **emitted as
part of building the step**, never recovered from text afterward. There is nothing to guess:
the planner has the provenance in hand at the moment of construction.

This is not a new idea in the codebase — it is an extension of a distinction the plan layer
already makes. `answer_behavior.build_answer_plan` returns evidence-backed `ContentBlock`s or
`None`; `answer_session.py` turns that into `decision="answer"` (grounded) vs `decision="audit"`
with `support_kind="missing_knowledge"` (no support). **This task adds a third, intermediate
category** — construction-time-labelled speculative inference — into a slot that already
carries per-block `kind` and full provenance.

---

## 1. Research question

> Can the reasoning core produce a **third class of answer plan** — beyond *answer* (fully
> grounded) and *audit* (no support) — in which it explicitly constructs and labels a
> **speculative inference chain** (grounded premises → explicit, inspectable reasoning step →
> speculative conclusion), with the label attached **at construction time**, not recovered
> after generation?

Success is measured on two separable axes (see §4), and it is important they stay separate:
grounded parts must stay at the proven QA zero-hallucination standard, **and** every
speculative step must be fully traceable premises→rule→conclusion. The open empirical question
is not "can we label it" (we can — §2) but "are the labelled conclusions defensible reasoning
or relabelled graph adjacency" (§6).

---

## 2. Architectural entry point

Real integration points, from source:

- `worldpgt/reasoning/answer_behavior.py:340` — `AnswerPlan` holds
  `blocks: tuple[ContentBlock, ...]`. Each `ContentBlock` (:298) carries `kind`, a `GraphStep`
  with its `EvidenceEdge`, `cautions`, and `alternatives`. `AnswerPlan.trace()` (:369) already
  emits per-block `kind`, `claim`, `evidence_id`, `sources`, `score` — an inspectable trace.
- `answer_behavior.build_answer_plan` (:1120) — the grounded planner. Returns `None` when no
  valid grounded plan exists → the caller audits.
- `answer_session.py` — maps plan/None to `decision` + `support_kind`
  (`"missing_knowledge"`, `"safe_policy_answer"`, …). This is where the third `support_kind`
  is registered.

### The three step types

- **`grounded_step`** — the existing mechanism, **unchanged**. Cites a specific `EvidenceEdge`
  with `evidence_id` + sources. This path is not touched; its proven metrics must be preserved.
- **`speculative_step`** — NEW. Constructed only when the planner recognizes a narrow
  what-if/why-might/what-would-happen-if question pattern **and** the grounded premises it needs
  are present. It:
  1. selects grounded premise edges from the graph (each with its own `evidence_id`),
  2. applies **one explicit, named, non-neural inference rule** (not an implicit heuristic),
  3. emits a conclusion tagged `speculative` with the rule name and the premise ids attached.
- **`audit`** — unchanged. If neither a grounded plan nor a *valid* speculative construction is
  available, the system still audits rather than inventing.

### The one inference rule for v1 — "counterfactual removal"

Deliberately a single, inspectable rule, not a rule engine:

> Given a grounded fact `F(subject, predicate, object)`, a "what if `subject` had not
> `predicate` `object`" question constructs a speculative_step by identifying **other explicit,
> graph-connected facts that share `subject` or `object` as a node**, and reasoning that those
> specific connected facts *would be the ones affected* — **restricted to graph-connected
> nodes only, never open-ended speculation.** The conclusion names exactly which connected
> facts, with their evidence ids, and is labelled speculative because the *causal dependency
> between them is inferred, not stored.*

Example: `F = (Elon Musk, founded, SpaceX)`. "What if Musk had not founded SpaceX?" →
premises: that edge + the graph-connected `(SpaceX, develops, rockets)`, `(SpaceX, employs, …)`
→ rule → speculative conclusion: "the facts that SpaceX develops rockets and … would be the
ones in question" — with all edge ids, explicitly framed as inference, not assertion.

### Scope limit for v1 (explicit)

Not "the system can freely philosophize." v1 proves **one** concrete, explicit speculative
reasoning type — **counterfactual/what-if over graph-connected facts** — a narrow, testable
first step, not a general reflection capability. The reuse of the CHAIN/`HopEdge` traversal
primitive from `compositional_grammar_v1` (object→subject joins over graph-connected nodes) is
the natural implementation substrate for "graph-connected facts."

---

## 3. Renderer

The speech-plan renderer marks output by source **from plan structure**, not from text:

- Grounded blocks render exactly as today (`answer_plan_renderer._grounded_clause`, quoting the
  evidence span for traceability).
- Speculative blocks render with explicit inference framing — e.g. *"Based on what is known …,
  one might reason that …"* — generated from the block's `kind` + rule name.

This is **not new machinery**. `render_answer_plan` already selects framing per `block.kind`,
and the `uncertainty_note` kind already renders structure-derived framing ("the evidence
diverges on X … neither reading is treated as settled") — built from plan structure, never
guessed from text. `speculative_step` is one more `kind` in the same `_CONNECTIVES`/`_sentence_for`
dispatch, and one more `support_kind` value known from the moment the plan is built — exactly
as the `creative_generated` label already works.

---

## 4. Metrics (design only)

- **Grounded-step accuracy** — the existing QA standard: **0% hallucination on grounded parts.**
  This must stay at the already-proven level; adding speculative steps must not regress it. (Test:
  the existing answer-behavior benchmark must pass unchanged when speculative construction is off,
  and grounded blocks inside a mixed plan must still cite real `evidence_id`s.)
- **Speculative-step traceability** — for every `speculative_step`, can we display the full
  chain: premises (grounded facts, by evidence_id) → named inference rule → conclusion? A
  complete, inspectable trace, no black-box reasoning. This is a *structural* property the plan
  either has or doesn't — cheaply auditable via `AnswerPlan.trace()`.
- **Scope coverage** — honestly, how many question patterns (what-if, why-might, what-would-happen-if)
  are actually backed by explicit inference rules in v1. v1 targets **one** rule
  (counterfactual removal); report coverage as "1 pattern family," not inflated.

No composite score. Grounded accuracy and speculative traceability are reported separately, and
speculative *plausibility* (§6) is reported as a manual judgment, never folded into an accuracy
number.

---

## 5. Why this is tractable where informed-reflection was not

The failure mode that killed informed-reflection **cannot occur here**: there is no
"recover the label from surface text" step. The planner constructs the step and knows its
origin with certainty — a grounded edge has an `evidence_id`; a speculative conclusion is
produced by a named rule over named premises. Labelling is correct *by construction*, the same
way `decision="audit"`/`support_kind` is correct by construction today. That is a real,
categorical difference, not optimism.

---

## 6. Honest assessment — the keystone risk is different, and cheaply checkable

The labelling is not the risk. **The keystone risk is inference-rule soundness: how widely can
inference rules go before they stop being "explicit and inspectable" and become implicit
guessing dressed up as reasoning?**

Concretely for counterfactual removal: the graph stores *relational* edges, not *causal* ones.
"SpaceX develops rockets" is connected to "Elon Musk founded SpaceX," but the graph does not
encode that the first *depends on* the second. So a rule that enumerates graph-connected facts
and asserts they "would be affected" risks producing **non-sequiturs labelled as reasoning** —
e.g. "if Musk hadn't founded SpaceX, then [some unrelated but graph-adjacent fact]." A
speculative label on a non-sequitur is still a defect: the trace is honest about *origin* but
the *conclusion* is absurd, and "traceable" is not the same as "defensible."

This is the exact analogue of the keystone risks the last two pilots isolated, and it is
**cheaply checkable before committing ~8–10 days**:

### Proposed pilot gate (~1–2 days, do this first)

1. Take **10–15 what-if questions** over the existing accepted/promoted graph overlay.
2. **By hand**, construct what the counterfactual-removal rule *would* output for each:
   premises (real edge ids) → rule → speculative conclusion. (No code needed, or a ~50-line
   throwaway enumerator over the overlay — same discipline as `pilot_classifier.py`.)
3. **Manually judge each conclusion on two axes:** (a) is the premises→rule→conclusion trace
   complete and inspectable? (expected: yes, by construction), and (b) is the speculative
   conclusion **defensible / non-absurd**, or a graph-adjacency non-sequitur?
4. **Gate:** if a clear majority of conclusions are defensible (say ≥ ~70% non-absurd, with the
   absurd ones filterable by a stated structural criterion such as "only traverse edges whose
   predicate is plausibly dependency-bearing"), **proceed to the full build.** If most
   conclusions are non-sequiturs and no simple structural filter rescues them, that is an
   honest negative result — the rule is relabelled adjacency, not reasoning — and we stop or
   re-scope, exactly as with informed-reflection.

### Effort, if the pilot passes

| Component | Est. |
|---|---|
| **Inference-rule pilot gate (above)** | **1–2 days — do FIRST** |
| `speculative_step` block kind + plan construction path (reuse HopEdge traversal) | 2 days |
| what-if / why-might question-pattern admission gate (conservative, explicit) | 1 day |
| `support_kind="speculative_inference"` wiring in `answer_session` | 0.5 day |
| Renderer framing for the new kind (extend `_CONNECTIVES`/`_sentence_for`) | 0.5 day |
| Metrics + benchmark (grounded-accuracy-unchanged regression + traceability check) | 1.5 days |
| Tests | 1 day |

**Total past the gate: ~6.5–7.5 days.** The pilot's 1–2 days decide the rest.

---

## 7. Recommendation

**Proceed — but to the inference-rule pilot first, not straight to the full build.**

Reasoning:
- The architectural premise is sound and genuinely distinct from the failed branch: labelling
  at construction time removes the exact failure that killed informed-reflection (§5). I am
  confident the labelling and traceability axes will hold.
- The remaining risk — whether the one inference rule produces *defensible* speculation rather
  than relabelled adjacency (§6) — is real, is the keystone, and is checkable for ~1–2 days.
  Spending that before ~6.5–7.5 days of build is the same cheap-gate discipline that has now
  paid off twice (grammar-extraction saved ~8–9.5 days; informed-reflection saved ~8–10).
- Unlike informed-reflection, my prior here is **optimistic**: the pilot is likely to pass,
  possibly with a small structural filter on which predicates are dependency-bearing. But the
  gate should still run, because "likely" is not "verified," and a cheap check that confirms a
  green light is as valuable as one that catches a red.

**Concrete next step for review:** approve the §6 pilot (10–15 what-if cases, hand-constructed
counterfactual-removal traces, manual defensibility judgment) as a standalone deliverable.
Escalate to the full build only if speculative conclusions are defensibly non-absurd.

---

## Out of scope (v1)

Multiple inference rules / a rule engine, non-counterfactual speculation (analogy, abduction,
prediction), open-ended (non-graph-connected) speculation, multi-hop speculative chains beyond
the counterfactual's immediate connected set, causal-edge modelling in the graph itself, and
production routing. v1 proves exactly one explicit, inspectable, construction-time-labelled
speculative reasoning type over graph-connected facts.
