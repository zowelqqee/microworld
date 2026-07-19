# Reflective reasoning core v1 — build notes

Both inference rules cleared their gate pilots, so this is the scoped v1 build. Isolated
research path: **no production grounded route was modified** (verified — only new files added).

## What was built

- `worldpgt/reasoning/reflective_reasoning_v1.py` — the module. Two construction-time-labelled
  speculative rules, each shipping with the pilot-validated structural filter:
  - `counterfactual_removal(edges, subject, predicate, object)` — fires only on existence-conferring
    predicates over entity objects; conclusion = facts referencing the object node; else `audit`.
  - `abduction_explanation(edges, subject, object)` — fires on a 2-hop bridge S→M→O; defers to
    grounded on a direct edge; declines on no-bridge / spurious 3-hop.
  - `reflect(question, overlay)` — conservative admission gate (regex for the two supported
    patterns; returns `None` otherwise — an admission gate, not NLU).
  - `render_reflective_plan(plan)` — framing built from plan structure; every speculative output
    is explicitly tagged "(This is a speculative inference, not a stored fact.)".
- `worldpgt/tests/test_reflective_reasoning_v1.py` — 16 tests, all passing, pinned to pilot findings.
- `worldpgt/experiments/run_reflective_reasoning_v1.py` — runs both rules over the real 276-edge
  overlay; writes `build_v1_results.json`. Result: 7 speculative, 5 audit, 1 grounded-deferral —
  matching the pilots.

## Five build conditions — status

1. **Ship filtered rules, not naive** — done; naive rule is not exposed.
2. **Abduction checks direct edge first** — done (`grounded_deferral`).
3. **Decline spurious 3-hop** — done (v1 default; see open question below).
4. **Prefer declining to speculating** — done; every non-defensible case audits.
5. **Grounded accuracy untouched** — done; the grounded planner is not imported or modified.

Counterfactual-specific condition 3 from the first pilot ("gate on single-founder facts to keep
the closed-world reading valid") is **NOT yet enforced** — see open questions.

## Plan structure carries the label at construction time

`ReflectivePlan.to_dict()` emits `support_kind` ∈ {`speculative_inference`, `grounded`,
`missing_knowledge`} and, for speculative steps, `premise_evidence_ids` + `conclusion_evidence_ids`.
This is the whole point: the label exists in the plan, never recovered from text.

## OPEN QUESTIONS for review (not decided arbitrarily)

1. **2-hop vs 3-hop abduction** (from `pilot_abduction_report.md` §5). v1 declines 3-hop-through-
   shared-entity ("Tesla ~ rockets via Musk"). These are weak-but-not-absurd associations. Adopting
   the conservative 2-hop-only default was principled (zero-absurd rule), but whether to later add
   graded-strength 3-hop associations is a semantics decision about what the mode claims. **Your call.**
2. **Single-founder gating for counterfactuals.** The founding-counterfactual is defensible because
   the graph records one founder. Tesla has two (Eberhard, Tarpenning); for such facts the "entity
   would not exist" reading weakens. v1 does **not** yet special-case multi-founder objects. Options:
   (a) decline when the object has other founders, or (b) weaken the conclusion ("some of the
   following might be affected"). Not decided — flagging for review.
3. **Integration surface.** This v1 is a standalone module for A/B evaluation. Wiring it into
   `answer_session` as a real third `support_kind` (vs keeping it experiment-only) is a separate
   decision with routing/safety implications; not attempted here.

## Known v1 limitations (not blocking)

- Renderer predicate lexicalization is a small hand map (`_PREDICATE_PHRASE`); unknown predicates
  fall back to underscore-stripping. Readability proxy only; the trace keeps raw predicates.
- Counterfactual conclusion lists can be long (SpaceX has 9 connected facts); no ranking/truncation
  yet. Fine for a trace; a production surface would cap it.
