# Reflective reasoning core — why-might abduction gate pilot report

**Type: hard-gate pilot #2 (~1 day). Second inference rule for the reflective reasoning core.**
**Verdict: PASS (with the 2-hop structural filter). Better coverage than the counterfactual rule. One architectural fork recorded as an OPEN QUESTION for review (§5) — a principled-conservative default is adopted for v1 so the build can proceed.**

Tests the same keystone question as the counterfactual pilot, for a second rule: does
*why-might abduction* (inference to a plausible explanation) over the graph produce **defensible
explanatory chains**, or does it relabel graph adjacency as reasoning?

Artifacts: [`pilot_abduction_enumerator.py`](pilot_abduction_enumerator.py),
[`pilot_abduction_traces.json`](pilot_abduction_traces.json),
[`pilot_abduction_score.py`](pilot_abduction_score.py),
[`pilot_abduction_scored.json`](pilot_abduction_scored.json).

---

## 1. Method

- **Real edges** from `capability_overlay.json` (276 relations).
- **14 "why might S be associated with O?" cases**, chosen to span: pairs with a clean 2-hop
  bridge (expected defensible), pairs bridged only through a shared person (expected spurious),
  and pairs with no bridge (expected decline).
- **Naive rule first:** dump the 1-hop neighbourhood of S and O as candidate "explanations."
- **Structural filter:** require a genuine connecting **path** S → M → O; abduction offers the
  bridge node M as the explanation.
- **Manual defensibility judgment:** is "S might be associated with O because <path>" a
  defensible explanation, or a non-sequitur?

---

## 2. Result 1 — the naive rule is again mostly non-sequiturs

The naive 1-hop dump produced **113 candidate explanations** across 14 cases. Most are
non-sequiturs — e.g. for "why might Elon Musk be associated with rockets?" the naive rule offers
"Elon Musk estimated_net_worth US$1.1 trillion" and "Elon Musk founded The Boring Company" as
"explanations." Same failure as before: 1-hop adjacency is not explanation. Naive fails the gate.

---

## 3. Result 2 — the 2-hop bridge filter is defensible and covers well

Routing 14 cases by the filter (`pilot_abduction_scored.json`):

| Route | Count | Meaning |
|---|---:|---|
| **fire_speculative** (2-hop bridge exists) | **9** | emit explanation via intermediate M |
| decline_spurious (only 3-hop, via shared person) | 3 | too weak → decline |
| decline_no_bridge (no path) | 1 | genuinely unrelated → decline |
| direct_grounded (direct edge exists) | 1 | not speculative — route to grounded answer |

**All 9 firing cases produce manually-confirmed defensible explanations** (100%):

| Why might … | Explanation the rule emits (bridge) |
|---|---|
| Musk ~ rockets | Musk leads SpaceX → SpaceX develops rockets |
| Musk ~ spacecraft | Musk leads SpaceX → SpaceX develops spacecraft |
| Musk ~ electric cars | Musk leads Tesla → Tesla produces electric cars |
| Bezos ~ rockets | Bezos leads Blue Origin → Blue Origin develops rockets |
| Bezos ~ spacecraft | Bezos leads Blue Origin → Blue Origin develops spacecraft |
| Shotwell ~ rockets | Shotwell leads SpaceX → SpaceX develops rockets |
| Musk ~ Starbase, TX | Musk leads SpaceX → SpaceX located in Starbase |
| Musk ~ Falcon rockets | Musk leads SpaceX → SpaceX produces Falcon rockets |
| Shotwell ~ spacecraft | Shotwell leads SpaceX → SpaceX develops spacecraft |

The declines are also correct:
- **Musk ~ Paris** → no path. Musk is genuinely unconnected to LVMH/Paris. Correct decline.
- **Blue Origin ~ rockets** → a *direct* edge exists ("Blue Origin develops rockets"). This is
  not a why-might speculation at all — it is a grounded fact. The rule correctly does **not**
  fire; the grounded planner answers it. (Good side-finding: the abduction rule must check for a
  direct edge first and defer to grounded QA.)

Coverage is **materially better than the counterfactual rule** (9/14 fire vs 2/13): "why might S
be associated with O" is naturally a path-finding question, and 2-hop bridges through a shared
entity are genuinely explanatory. Every full trace is inspectable (premises = real edge ids →
rule `abduction_path_explanation` → speculative conclusion), so **traceability holds by
construction**, as with the counterfactual rule.

---

## 4. Why the 3-hop declines matter (the filter is doing real work)

Three cases have **no 2-hop bridge, only 3-hop paths routed through a shared person**:

- "Why might **Tesla** be associated with rockets?" → Tesla ← Musk (leads both) → SpaceX →
  rockets.
- "Why might **Neuralink** be associated with rockets?" → Neuralink ← Musk → SpaceX → rockets.
- "Why might **SpaceX** be associated with electric cars?" → SpaceX ← Musk → Tesla → cars.

These are **guilt-by-association** explanations: "these two things share an owner, and the other
one does X." At the 2-hop filter they correctly **decline**. This is the abduction analogue of
the counterfactual pilot's non-sequiturs — and it confirms the filter is separating real
explanatory bridges from spurious shared-node adjacency, not just counting hops.

---

## 5. OPEN QUESTION for review — 2-hop vs 3-hop-through-shared-entity

**This is a genuine architectural fork with legitimate arguments both ways. Per the standing
instruction, I am recording it explicitly rather than deciding it arbitrarily. v1 adopts the
conservative default below so the build can proceed, but the decision is yours.**

The 3-hop-through-shared-person explanations (§4) are **not fully absurd** the way the naive
non-sequiturs are. "Tesla and SpaceX are both Musk companies, and SpaceX makes rockets" is a
*weak but real* association a person might actually offer. So:

- **Option A (v1 default): 2-hop only.** Zero weak explanations; decline the association-through-
  person cases. Follows the established "prefer declining to speculating / zero absurd output"
  principle from the counterfactual pilot. Cost: misses the weaker-but-arguably-valid
  associations (~3/14 here).
- **Option B: allow 3-hop through a shared entity, labelled as a *weaker* association.** Broader
  coverage; matches how people loosely associate co-owned entities. Cost: the conclusions are
  looser, and "weaker" is a graded label the renderer and metrics must represent honestly —
  risks reintroducing the "is this reasoning or adjacency?" ambiguity the gate exists to prevent.

**Why I did not just pick B for coverage:** it changes the *meaning* of a speculative_step (from
"defensible explanatory bridge" to "graded-strength association"), which is a semantics decision
about what the mode claims, not an implementation detail. I adopted **A for v1** because it is
the principled continuation of the zero-absurd rule already established — not an arbitrary
coin-flip — and it keeps v1's claims clean. But if you want the mode to cover looser
associations, B is defensible and the pilot data (§4) shows exactly which cases it would add.

---

## 6. Gate decision & recommendation

**PASS.** The why-might abduction rule, with the 2-hop bridge filter, produces 100% defensible
explanations on the 9 cases it fires, correctly declines the 4 it should (1 no-bridge, 3
spurious-3-hop), and correctly defers the 1 direct-edge case to grounded QA. Naive-to-filtered
takes defensibility from ~low (1-hop dump) to 100%. This is a second genuinely-sound,
construction-time-labelled speculative rule.

Build conditions (carried from the counterfactual pilot, plus abduction-specific):
1. **Ship the 2-hop-filtered rule, not the naive dump.**
2. **Check for a direct edge first** — if S–O is a grounded fact, defer to the grounded planner;
   abduction fires only when the relationship is *not* directly stored.
3. **Decline on spurious-only (3-hop-through-shared-entity)** for v1 — pending the §5 decision.
4. **Prefer declining to speculating** (zero absurd output).
5. **Grounded-step accuracy must stay at the proven QA level** — additive path only.

Both proposed rules — counterfactual removal and why-might abduction — have now cleared their
gates. The reflective reasoning core build is justified for **two** inference rules, scoped as
above, with the §5 open question flagged for your call.
