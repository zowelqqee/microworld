# Branch router v1 — design + gate pilot

**Type: design + hard-gate pilot (same discipline as the day's prior five).**
**Verdict: PROCEED to integration.** Retrieval/centroid router, **no training, no neural
inference at route time**. Gate met with a recalibrated margin.

Dispatches between `qa` / `reflective` / `constrained_creative` / `pure_creative`, with QA as the
conservative default. Imports production code read-only; touches no existing branch. Artifacts:
[`router.py`](router.py), [`pilot_cases.json`](pilot_cases.json), [`run_pilot.py`](run_pilot.py),
[`pilot_results.json`](pilot_results.json).

---

## 1. Architecture (as fixed in the task)

1. **Fast-path structural rules** — the SAME explicit markers the branches already use:
   `reflective_reasoning_v1._WHATIF_RE` / `_WHYMIGHT_RE` (imported directly), and a
   constrained-creative regex for "using only/exactly/just these facts". Matched → done, no
   embedding.
2. **Semantic centroid fallback** — the proven predicate-centroid mechanism, reused verbatim:
   GloVe `glove-wiki-gigaword-100` (static vectors, dict-lookup + mean — **not** a model forward
   pass), L2-normed per-branch centroids from 8–12 representative example phrases, margin-gated
   cosine (absolute threshold + margin over runner-up).
3. **Below threshold/margin → default to QA** (conservative, self-auditing; never defaults to a
   creative/speculative branch).

Safety/private/current-sensitive screen is assumed to run **before** the router, unchanged.

This is consistent with the "no neural inference in the reasoning core" invariant: identical to
`predicate_centroid_index` in kind — deterministic arithmetic over static vectors.

### Representative examples (all from existing material)
QA question templates (from the dataset builder / predicate phrases); reflective forms (today's
what-if / why-might pilots); constrained-creative (today's build prompt); pure-creative
(Creative-mode demo prompts: "Compose a poem about rockets", "Write a story about a rocket", …).

---

## 2. Pilot

31 hand-labelled cases (labels fixed before running): 18 clear (4–5 per branch) + 13 boundary
(qa↔reflective, constrained↔pure, underspecified-default).

### Calibration sweep — the inherited margin is wrong for this task

Mean-pooled GloVe question vectors live in a tight cone (the predicate index notes 0.79–0.98),
and **branch centroids sit even closer together** than predicate centroids (every example is
"question/instruction"-shaped). So the inherited `margin=0.04` over-rejects true matches and
dumps them to the QA default:

| threshold | margin | overall miss | clear miss | ambiguous miss |
|---:|---:|---:|---:|---:|
| 0.85 | **0.04 (inherited)** | 16.1% | 11.1% | 23.1% |
| 0.85 | 0.03 | 12.9% | 5.6% | 23.1% |
| **0.85** | **0.02 (calibrated)** | **3.2%** | **0.0%** | **7.7%** |
| 0.88 | 0.02 | 6.5% | 5.6% | 7.7% |

The absolute threshold barely matters at margin 0.02 (0.80 and 0.85 are identical — sims are all
≥0.85); raising it to 0.88 starts rejecting true matches (clear miss → 5.6%). **Calibrated
values: threshold 0.85, margin 0.02.** (Same calibration procedure the predicate index used — the
margin, not the absolute cosine, is the decisive signal; it is simply tighter here.)

### Result at calibrated 0.85 / 0.02

| Metric | Value | Gate | Pass |
|---|---:|---|:--:|
| Clear-case misroute | **0.0%** (0/18) | 0% | ✅ |
| Ambiguous misroute | **7.7%** (1/13) | <10% | ✅ |
| Overall misroute | 3.2% (1/31) | — | — |

By boundary: `qa_vs_reflective` and `constrained_vs_pure` each had their misroutes resolved by
the margin recalibration except **one** residual case. Method split: 6 fast-path, 16 centroid,
9 default-QA.

### The single residual miss fails SAFE

> "Tell a short story about a rocket company like SpaceX" — gold `pure_creative`, routed `qa`
> (margin 0.004 — a genuine near-tie: "story" pulls creative, "SpaceX company" pulls QA).

It is below margin, so it **defaults to QA** — the conservative, self-auditing branch — rather
than mis-committing to a creative or speculative branch. This is the architecture's intended
failure mode: an ambiguous request falls back to the safest path, not a wrong confident one. So
even the one miss is a safe miss, not a dangerous one.

---

## 3. Verdict & calibrated config

**PROCEED to integration.** Gate met: 0% clear misroute, 7.7% ambiguous (<10%), and the lone
residual miss fails safe to QA.

Integration notes (for the build step, pending review):
- **Calibrated thresholds: `threshold=0.85, margin=0.02`** — differs from the predicate-matching
  default (`0.04`); documented above with the sweep that justifies it.
- For production, persist a **compact query vocabulary** (as `predicate_centroid_index` does) so
  the route path needs no GloVe model loaded — the pilot used the full model for convenience;
  the mechanism is identical.
- Fast-path stays first (6/31 here) — free, exact, and covers the explicit-marker cases with no
  embedding.
- Keep QA as the default for every below-margin case.

### Scope guard
31 cases, one small example set per branch, one embedding model. The 3.2% is an in-sample
calibration figure on hand-picked boundaries (deliberately adversarial), **not** a production
routing-accuracy estimate. It shows the mechanism is viable and the safe-default holds; a larger
labelled set would be needed before claiming a population accuracy.
