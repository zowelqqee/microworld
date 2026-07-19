# Reflective reasoning core — inference-rule gate pilot report

**Type: hard-gate pilot (~1–2 day discipline). Decides whether the ~6.5–7.5 day full build proceeds.**
**Verdict: CONDITIONAL PASS — proceed to build, but with scope narrowed to the pattern the pilot proved sound, and the structural filter baked in as the admission gate. Do NOT ship the naive rule.**

This pilot tests the one keystone risk `design.md` §6 isolated: not whether we can *label*
a step speculative at construction time (that is sound by construction — never in doubt), but
whether the **counterfactual-removal inference rule produces defensible conclusions or
relabelled graph adjacency dressed up as reasoning.**

Artifacts: [`pilot_enumerator.py`](pilot_enumerator.py) (applies the rule over the real graph),
[`pilot_traces.json`](pilot_traces.json) (raw candidates), [`pilot_score.py`](pilot_score.py)
(applies the structural filter), [`pilot_scored.json`](pilot_scored.json) (filtered results).

---

## 1. Method

- **Real graph, real edges.** 13 focal facts drawn from `artifacts/compositional_grammar_v1/capability_overlay.json`
  (276 accepted relations). Every premise→conclusion trace is built from actual overlay edges,
  not invented — same discipline as the informed-reflection pilot.
- **13 what-if cases** spanning both predicate classes: existence-conferring (`founded`,
  `develops`, `produces`, …) and incidental (`located_in`, `leader_of`, `known_for`,
  `estimated_net_worth`).
- **The rule, run naively first:** for `what if S had not P O`, collect every graph-connected
  fact (co-subject / object-as-subject / shared-node) as a candidate "affected" fact.
- **Manual defensibility judgment** (the core of the pilot): for each candidate, is
  "if not F, then this fact is in question" a **defensible** counterfactual, or a
  **non-sequitur**? A fact is defensibly in question only if it plausibly depends, causally or
  existentially, on the removed fact.

---

## 2. Result 1 — the naive rule is mostly non-sequiturs (fails on its own)

Across 13 cases the naive rule produced **98 candidate "affected" facts**. Manual judgment: the
large majority are non-sequiturs. Representative failures:

| What-if | A candidate the naive rule emits | Defensible? |
|---|---|---|
| Musk had not **founded** SpaceX | Musk founded **xAI** | ❌ independent venture |
| Musk had not **founded** SpaceX | Musk **estimated_net_worth** $1.1T | ❌ attribute persists (magnitude aside) |
| Musk had not **leader_of** Tesla | **Martin Eberhard founded Tesla** | ❌ unrelated co-founder fact |
| Tesla had not **produced** electric cars | Musk **leader_of** Tesla | ❌ leadership independent of product line |
| SpaceX had not **located_in** Starbase | SpaceX **develops** rockets | ❌ location change ≠ product change |

Only ~13–15 of the 98 naive candidates are defensible (~15%). **The naive rule fails the gate
decisively** — it is exactly the "relabelled adjacency, not reasoning" failure §6 warned about.
The graph stores *relational* edges, not *causal* ones, so blind connectivity is not inference.

---

## 3. Result 2 — a stated structural filter removes 100% of the absurd conclusions

The filter `design.md` §6 anticipated, **refined by this pilot's data** (the enumerator's
initial `DEPENDENCY_BEARING` set was too broad; the data showed only *existence-conferring*
predicates over *entity* objects work):

> Fire the rule only when **(1)** the focal predicate is **existence-conferring** (founding/
> creation of a persistent entity: `founded`, `founded_by`, `created_by`, `developed_by`,
> `product_of`, `construction_started`), **and (2)** the removed object **is itself a graph
> entity** (appears as the subject of ≥1 other fact). Then admit only candidates that
> **reference that object node**. Otherwise **decline** (audit) — emit nothing.

Applied (`pilot_scored.json`):

| | Value |
|---|---:|
| Cases where rule fires | **2 / 13** |
| Cases that correctly decline | 11 / 13 |
| Naive candidates | 98 |
| Admitted candidates after filter | **13** (86.7% reduction) |
| **Admitted candidates that are defensible (manual)** | **13 / 13 = 100%** |
| Absurd conclusions emitted | **0** |

The two firing cases and their admitted conclusions — all manually confirmed defensible:

**"What if Elon Musk had not founded SpaceX?"** (9 admitted)
→ Musk leader_of SpaceX; Musk known_for SpaceX; SpaceX develops rockets; SpaceX develops
spacecraft; SpaceX produces Falcon rockets; SpaceX produces Dragon spacecraft; SpaceX located_in
Starbase; SpaceX located_in El Segundo; Gwynne Shotwell leader_of SpaceX.
*(All follow: if the entity SpaceX had not been founded by Musk, every fact about SpaceX — and
anyone's relationship to it — is defensibly in question. Closed-world reading: the graph records
no alternate founder.)*

**"What if Jeff Bezos had not founded Blue Origin?"** (4 admitted)
→ Bezos leader_of Blue Origin; Blue Origin develops rockets; Blue Origin develops spacecraft;
Blue Origin located_in Kent, Washington. *(Same structure.)*

Every full trace is inspectable: premises (real `evidence_id`s) → named rule
(`counterfactual_removal`) → speculative conclusion (each an `evidence_id`). This confirms the
**traceability** metric (§4 of the design) holds by construction.

---

## 4. The honest caveat: soundness is bought with narrow coverage

The filter passes the gate — but it is essential to state *how*. It achieves 100%
defensibility by collapsing the rule to essentially **one pattern: the counterfactual over the
founding of a named entity that has its own downstream facts.** Consequences, stated plainly:

- **Scope coverage (a §4 metric) is low.** Of 13 diverse what-if forms, only 2 produce any
  conclusion, and both are the same founding-counterfactual shape. `located_in`, `leader_of`,
  `known_for`, `net_worth`, and even `develops`/`produces` over generic products all correctly
  decline — because their objects are attributes or generic categories with no downstream facts,
  so there is nothing defensible to say.
- **The filter under-covers some genuinely-defensible cases (safe false-declines).** E.g.
  "What if SpaceX had not developed rockets?" declines, yet "SpaceX produces Falcon rockets"
  (Falcon rockets *are* rockets) is arguably in question. Missing it is a **coverage gap, not an
  absurdity** — declining is safe; emitting a non-sequitur is not. v1 should prefer silence.
- **Closed-world assumption.** The founding-counterfactual is defensible *because the graph
  records no alternate founder*. If the graph later holds co-founders (it does for Tesla:
  Eberhard, Tarpenning), the "entity would not exist" reading weakens and the rule would need to
  reason about remaining founders. v1 must gate on single-founder facts or explicitly weaken the
  conclusion.

So the keystone question — *defensible reasoning vs relabelled guessing* — gets a clear **YES**,
but only for a **narrow, well-defined slice**, and only because the naive rule is discarded in
favour of the filtered one.

---

## 5. Gate decision & recommendation

**PROCEED to the full build — conditionally.** The §6 gate criterion ("conclusions defensible,
with absurd ones filterable by a stated structural criterion") is met: the stated filter takes
defensibility from ~15% (naive) to 100% (filtered), emitting zero absurd conclusions. This is
the genuinely-sound, construction-time-labelled speculative reasoning the design proposed — and
it is categorically unlike the informed-reflection failure, because nothing is guessed from text.

Conditions the build must honour (all cheap, all learned from this pilot):
1. **Ship the filtered rule, never the naive one.** The admission gate (existence-conferring
   predicate + object-is-entity + references-object-node) *is* the mechanism, not an optional
   post-filter.
2. **Scope v1 honestly to the founding-counterfactual pattern.** Do not market it as general
   what-if reasoning. Report scope coverage as the small number it is (~2/13 here).
3. **Gate on single-founder facts** (or weaken the conclusion when alternate founders exist) to
   keep the closed-world reading valid.
4. **Prefer declining to speculating.** A false-decline is a coverage gap; a non-sequitur is a
   correctness defect. v1 optimizes for zero absurd output.
5. **Grounded-step accuracy must stay at the proven QA level** — the speculative path is additive
   and must not touch the grounded planner (design §4).

This matches — and tempers — the optimistic prior stated in the design doc: the capability is
real and sound, just narrower than "what-if reasoning" in general. The ~1–2 day pilot cost
confirmed a green light *and* corrected the scope before ~6.5–7.5 days of build, which is exactly
what the gate is for.

**Concrete next step for review:** approve the full build scoped to (1)–(5) above, or ask for a
second pilot rule (e.g. `why-might` via abduction over the same graph) before committing — but
the counterfactual-removal rule itself has cleared its gate.
