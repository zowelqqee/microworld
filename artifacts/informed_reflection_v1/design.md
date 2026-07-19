# Informed-reflection generation v1 — design

**Status: plan only. No code, no generation.**
Third and most architecturally risky generation mode, parallel to strict QA
(answer/audit) and constrained-creative (must include all given facts verbatim). This is an
isolated research path awaiting a **gating decision** — see the recommendation at the end.
It does not modify any production route, the support gate, or the existing Creative mode.

> **Read the recommendation first (§7).** Unlike the other two design docs, this one does
> **not** recommend proceeding straight to full implementation. It recommends a small,
> cheap, manual **pilot as a gate** — because the core mechanism may be an honest negative
> result, and we should find that out for ~1–2 days of work, not ~10.

---

## 1. Research question

> Can a system distinguish, **within freely generated reflective text**, between factual
> assertions (which must remain accurate) and speculative/hypothetical content (which need
> not be fact-checked) — maintaining **zero hallucination on the factual assertions** while
> allowing free-form reasoning on the speculative parts — in a way a same-scale LLM,
> prompted similarly, does or does not reliably do?

Explicit non-goal: **not** "who reasons or speculates better" (subjective, unwinnable). The
measurable claim is narrow: **who better maintains factual accuracy specifically on the
factual claims embedded within free-form reflective/hypothetical text, without
over-constraining the speculative parts.** Two failure modes are both losses, and the second
is easy to miss: (a) hallucinating inside a factual assertion, and (b) collapsing everything
into hedged speculation to *avoid* ever asserting a fact — which is not "informed reflection,"
just evasion (see the speculative-content-rate guard in §4).

How it differs from the other two modes:
- **Strict QA** — answer or audit; no free reasoning.
- **Constrained-creative** — must include *all* N given facts verbatim; hard inclusion gate.
- **Informed-reflection** — facts are *available to reference, not mandatory*; free
  reasoning is allowed; but *any explicit factual assertion that does appear* must be
  accurate. The gate moves from "did you include the facts" to "is each thing you *asserted
  as fact* actually true."

---

## 2. Core problem — sentence-level factual/speculative segmentation

This is the whole task. Everything else is downstream of it. Within generated output we must
separate:

- **(a) Factual assertion** — declarative, presenting something as true ("X founded Y", "X
  is located in Y"). **Must** be verified against the graph; wrong or unsupported → flagged
  as hallucination.
- **(b) Speculative / hypothetical / reflective** — conditional, counterfactual, or
  explicitly framed as opinion/imagination ("what if X hadn't happened", "one might argue
  that…", "it is interesting to consider…"). **Not** subject to fact-verification.

### Can this be done rule-based, before assuming an LLM classifier is required?

We commit to trying **deterministic, structural classification first**, using signals that
do not require neural inference:
- **Grammatical mood / conditionals** — `if … would/could`, `were X to …`, subjunctive.
- **Modal / epistemic verbs** — `might`, `may`, `could`, `perhaps`, `possibly`, `probably`.
- **Explicit hedging / framing phrases** — `one might argue`, `it is interesting to
  consider`, `I suspect`, `arguably`, `in my view`, `imagine that`, `suppose`.
- **Assertion defaults** — a bare present-tense declarative with a named subject and a known
  predicate cue, none of the above markers present → provisionally classed as **factual
  assertion** (the conservative default: unmarked declaratives are treated as claims, so we
  err toward verifying rather than excusing).

### What already exists (honest baseline)

The system already contains the *closest precedent* to this classification decision, and it
is important to be precise about what it does and does not cover:
- `worldpgt/cognition/support_guard.py` — `validate_conclusion_support` classifies a
  conclusion **by its declared `kind`** and enforces different rules per kind: it flags
  unsupported profile terms against an evidence workspace, blocks a `mechanism_gap`
  conclusion that nonetheless claims a mechanism (`"works by"`), and rejects forbidden
  claims. `worldpgt/cognition/reasoning_engine.py` / `mini_reasoner.py` produce the
  reasoning trace whose conclusions this guards.

The crucial gap: `support_guard` classifies **pre-declared, structured conclusions** carrying
a `kind` label the reasoner assigned. Informed-reflection needs to classify **free-generated
sentences that carry no such label** — the classification must be *recovered from surface
form after generation*, which is a strictly harder problem. So the architectural slot (a
gate that treats different assertion types by different rules) exists and is reusable in
spirit; the **post-hoc sentence classifier is the new, unproven piece.**

### The honest limit clause (stated up front, not as an afterthought)

If rule-based sentence classification proves unreliable in a small manual pilot — high
false-positive rate (speculation mis-read as asserted fact) or, worse, high false-negative
rate (a real factual claim slipping through as "speculation" and escaping verification) —
then that is **the same class of honest architectural limit** documented in the
grammar-extraction pilot (`artifacts/grammar_extraction_v1/adhoc_pilot_report.md`), where a
wider extractor still hit structural walls and the honest outcome was a **negative result
that saved the larger investment.** It must be documented the same way, not forced. A
classifier that is only ~70% reliable makes every downstream metric meaningless (§4), so an
unreliable classifier is a *stop* signal, not a "tune it later" signal.

---

## 3. Generation mechanism

Freer than constrained-creative, but grounded in factual context about the subject.

- **Base engine — reuse `creative_generator.py`.** Its `_boost_field` soft spreading
  activation is exactly the right primitive for "facts available to reference, not mandatory
  to include": the known facts about subject X seed the boost field, biasing the walk toward
  them **without a hard inclusion requirement.** `seeded_weighted_pick` keeps it replayable.
- **Difference from the existing free Creative mode.** That mode has *no grounding at all* —
  pure corpus recombination, and its 4-gram gate only prevents recitation. Informed-reflection
  **must have access to factual context about the specific subject** (from the same graph
  overlay QA reads), even though it is not obliged to use every fact. So the input is
  `{subject, available_facts[]}` used as *seed/context*, not as a constraint set.
- **The 4-gram novelty gate is retained** (recombine, not recite) but is orthogonal to the
  factual/speculative question — it governs originality, the §2 classifier governs accuracy.
- Generation and classification are **decoupled**: generate freely, then run every output
  sentence through the §2 classifier, then verify only the factual-assertion sentences
  against the graph. This keeps the risky component (classifier) isolated and independently
  measurable.

---

## 4. Metrics (design only)

- **Factual-assertion accuracy** — among sentences classified as factual (not speculative),
  the fraction that are accurate vs. hallucinated (verified against the graph overlay). Held
  to the **QA-track unsupported-claim standard: 0% if achievable.** This is the headline
  metric — but see the dependency below.
- **Speculative-content rate** — fraction of output legitimately classified as speculation.
  **Must not be ~100%.** If everything is classed as speculation, the system is dodging
  factual claims entirely, which is evasion, not informed reflection. A healthy result shows
  a *mix*: some grounded assertions, some free speculation. Report the distribution, not just
  the accuracy on the factual slice.
- **Classification accuracy (the critical, gating metric)** — under **manual review**, did
  the classifier correctly label each sentence factual vs. speculative? This is the
  bottleneck of the entire task. **If classification is unreliable, every other metric is
  meaningless** — a "0% hallucination on factual sentences" number is worthless if factual
  sentences were silently misfiled as speculation. This metric therefore *gates* the
  interpretation of the other two, and must be reported first.

Reporting rule: factual-assertion accuracy is only creditable *conditional on* a stated,
manually-verified classification accuracy. No composite score.

---

## 5. LLM comparison

- **Same model / scale** — `mlx-community/Qwen2.5-0.5B-Instruct-4bit`, same checkpointed
  harness as `open_book_qa`.
- **Same subject set**, plus a disjoint-independence subset (as in prior tracks).
- **Prompt** — "Share some thoughts about X, including what you know and what you speculate
  about." The LLM is invited to reflect exactly as our system does.
- **Same scorer** — the §2 classifier + §4 factual-accuracy check applied **identically to
  the LLM's output.** The interesting question is empirical: when the LLM explicitly asserts
  a fact inside reflective text, does it stay accurate, or does it hallucinate *even within
  reflection*? Applying our own classifier to the LLM output is itself a confound to note
  (our classifier defines what counts as "an assertion" for both sides) — this must be
  disclosed; it is a symmetric scorer, but it is *our* scorer.

---

## 6. Honest risk & effort estimate

**This is the most architecturally risky of the three branches (QA, constrained-creative,
informed-reflection).** Sentence-level factual/speculative classification is an open,
not-fully-solved problem even in mainstream NLP; doing it *rule-based, deterministically,
at 0.5B-adjacent scale* is deliberately betting on the easy tail of a hard problem.

Effort, if the pilot (below) passes:

| Component | Est. |
|---|---|
| **Manual classification pilot (gate)** — 10–15 hand-picked sentences, rule-based classifier prototype, manual scoring | **1–2 days — do this FIRST** |
| Rule-based sentence classifier (mood/modal/hedge markers + assertion default) | 2–3 days |
| Factual-assertion verifier against graph overlay (reuse overlay + support-guard patterns) | 1.5 days |
| Generator wiring (creative_generator soft-boost from subject facts) | 1 day |
| Experiment runner + Qwen comparison + report | 1.5 days |
| Tests | 1 day |

**Total if it proceeds past the gate: ~8–10 days.** But the honest structure is: **the first
1–2 days decide the other ~8.**

**Explicit stop condition:** *If rule-based classification proves unreliable in the small
manual pilot (10–15 sentences) — meaningful false-positive/false-negative rates on manual
check — recommend NOT proceeding to full implementation. That is an honest negative result,
not a failure of effort,* and it should be written up the way the grammar-extraction pilot
wrote up its structural walls. A cheap, decisive negative here *saves* the ~8-day build.

---

## 7. Recommendation

**Do not start full implementation. Run the gating pilot, then decide.**

Reasoning:
- The payoff is real and distinctive (a factual/speculative-aware generation mode is
  something the other two modes cannot express), **but it rests entirely on one unproven
  component** — the post-hoc sentence classifier — whose feasibility we do not yet know and
  whose failure invalidates everything downstream (§4).
- The precedent in `support_guard` is encouraging but does **not** de-risk this: it
  classifies *labelled* conclusions, not *unlabelled free-generated* sentences.
- Given remaining time before external deadlines, spending ~8–10 days on a build whose
  keystone might not hold is the wrong bet. Spending **1–2 days** on a manual pilot that can
  *decisively* tell us whether rule-based classification is viable on this material is the
  right bet — it either unlocks the build with evidence, or produces a clean, publishable
  negative result.

**Concrete next step for review:** approve the §6 pilot (10–15 sentences, manual scoring of
classification accuracy) as a standalone deliverable. Escalate to full implementation **only
if** the pilot shows classification is reliable enough that the §4 metrics would be
meaningful. If it does not, stop here and document the limit — that is a legitimate outcome
of this branch, on par with the grammar-extraction negative result.

---

## Out of scope (v1)

Neural/LLM-based sentence classification, multi-sentence discourse structure, mixed
factual-within-speculative sentences (a single sentence that is half assertion, half
hypothesis — the pilot should note how often these occur, as they are the hardest case),
long-form reflection, stylistic control, and any production routing.
