# Informed-reflection classification — gate pilot report

**Type: hard-gate pilot (1–2 day discipline). Decides whether the ~8–10 day full build proceeds.**
**Verdict: DO NOT PROCEED to full implementation. Honest structural negative result.**

This pilot tests the one keystone risk `design.md` isolated: can a **deterministic, rule-based
(no neural inference)** classifier reliably separate *factual assertions* (must be verified)
from *speculative/hypothetical* content (need not be) inside **free-generated, unlabelled**
sentences? `support_guard.py` does something adjacent, but only on *pre-labelled structured
conclusions*; here there is no label to lean on.

Artifacts: [`pilot_sentences.json`](pilot_sentences.json) (38 examples, gold labels judged
before running), [`pilot_classifier.py`](pilot_classifier.py) (minimal prototype using only
the markers `design.md` §2 named), [`pilot_results.json`](pilot_results.json) (raw output).

---

## 1. Method

- **38 hand-authored sentences** over the system's real subject domain (SpaceX, Tesla, LVMH,
  …). Per `design.md` §2, the set **deliberately over-samples the hard cases** — hedged-factual,
  mixed, speculative-with-definite-framing — because those are exactly what the pilot must
  stress. **Consequence, stated up front: the aggregate accuracy below is a STRESS-TEST
  figure, not a population estimate.** (Same "do not over-generalize" caveat the
  grammar-extraction pilot made.)
- **Gold labels judged by hand before the rule set was written or run.**
- **Rule:** any speculative marker present → `SPECULATIVE`; else → `FACTUAL` (conservative
  default: an unmarked declarative is treated as an asserted claim). Markers are exactly the
  families `design.md` §2 proposed (conditional `if/would/were-to`, modals
  `might/may/could/perhaps/possibly/probably`, hedges `it seems / one might argue / I suspect
  / …`), matched on word boundaries.
- **Positive class = FACTUAL** (the class that triggers verification), so:
  - **False positive** = speculative mislabelled factual → *unverified content passes as verified*.
  - **False negative** = factual mislabelled speculative → *a real claim escapes verification*.

---

## 2. Results (from `pilot_results.json`)

| Metric | Value | Design threshold | Pass? |
|---|---:|---|:--:|
| Clean-case accuracy (20 clean sentences) | **100.0%** | — | ✅ |
| Overall accuracy (32 non-mixed) | **84.4%** | > 90% | ❌ |
| False-positive rate, **clean cases only** | **0.0%** | < 5% | ✅ |
| False-positive rate, overall | 14.3% | — | — |
| False-negative rate | 16.7% | — | — |
| Mixed-sentence frequency | 15.8% (6/38) | evaluated separately | ⚠️ |

The gate criterion was **> 90% overall accuracy AND < 5% FP on clean cases**. Clean-FP passes;
**overall accuracy fails (84.4%)**. The failures are not random noise — they fall into three
structural buckets.

### The errors, side by side

**False negatives — hedged-factual (real claims that would escape verification):**

| id | sentence | markers fired | why it's wrong |
|---|---|---|---|
| 27 | "**It seems that** SpaceX was founded in 2002." | `it seems`, `seems that` | The hedge wraps a checkable claim; the founding date still needs verifying. |
| 28 | "**Perhaps** the most important fact is that SpaceX develops rockets." | `perhaps` | `perhaps` scopes the meta-judgment "most important", **not** the embedded fact "SpaceX develops rockets" — which is asserted and checkable. |
| 30 | "**If** I recall correctly, Blue Origin was founded by Jeff Bezos." | `if` | "If I recall" is a *memory* hedge, not a conditional over the world. The claim is factual. |

Note ids 29 & 31 ("As far as I know…", "Apparently…") were scored **correct** — but only
because the rule set happens not to list those particular hedges. Had it, they would have
failed identically. Their "correctness" is luck, not robustness.

**False positives — unmarked speculation (predictions passing as verifiable fact):**

| id | sentence | why it's wrong |
|---|---|---|
| 37 | "The luxury market **will** double in the next ten years." | A future prediction with no modal/hedge; routed to FACTUAL → would be sent for verification as if checkable. |
| 38 | "Reusable rockets change the economics of spaceflight **forever**." | Sweeping editorial generalization, present tense, no marker → routed to FACTUAL. |

**Mixed sentences (15.8% of the set) — no coherent handling is possible:**

The binary rule split the 6 mixed sentences 5 SPECULATIVE / 1 FACTUAL. Both outcomes are wrong
in the same way — a mixed sentence contains a factual half that must be verified **and** a
speculative half that must not, and a single label cannot express that:

- id 21 "SpaceX **was founded in 2002**, and it **might** reach Mars…" → labelled SPECULATIVE →
  the *verifiable* founding date rides along unverified.
- id 23 "Tesla **is based in Austin**, and one wonders **whether it will move again**." → no
  listed marker → labelled FACTUAL → the *speculative* half ("will move again") would be sent
  for verification and flagged as a hallucination it never was.

---

## 3. Why this is a *structural* limit, not a tuning gap

This is the crux, and it is the same shape as the grammar-extraction negative result
("more predicate types don't help — the wall is structural").

**The same lexical marker legitimately scopes both a fact and a speculation, and surface
matching cannot tell which.** `perhaps`, `it seems`, `if`, `could` appear in *hedged-factual*
sentences (where a checkable claim is still being asserted) and in *genuinely speculative*
ones. The marker's mere presence is not the signal; **what the marker takes scope over** is —
and recovering scope ("does this modal operate on the whole proposition, or on a meta-comment
about it?") is a semantic/parse-level judgment, precisely the neural/semantic understanding
this pilot was testing whether we could avoid.

Adding more markers cannot fix it — it makes it *worse*: every hedge added to catch a
speculative case also mis-fires on the hedged-factual case that shares that hedge (ids 27–31
are the live demonstration). And **mixed sentences are unfixable at the sentence level by
construction** — they need sub-sentence (clause/span) segmentation, which is a strictly harder
problem than the one the design already flagged as risky.

Critically, these are not exotic edge cases for *this* mode: **hedged and mixed
fact-plus-speculation language is what "informed reflection" IS.** The generation mode's own
premise produces exactly the sentence forms the classifier cannot handle. A tool that is
perfect on clean declaratives but unreliable on hedged/mixed ones is reliable everywhere
*except* the content this mode exists to produce.

---

## 4. Gate decision & recommendation

**STOP. Do not proceed to the full ~8–10 day informed-reflection build on a rule-based
classifier.** This is an honest negative result, not a failure of effort — and it cost ~1 day
instead of ~10, which is the entire point of the gate (same economics as the grammar-extraction
pilot that saved the 8–9.5-day declarative-engine build).

Because a rule-based classifier that is unreliable on hedged/mixed sentences makes the
downstream headline metric ("0% hallucination on factual assertions") **meaningless** — a
clean factual-accuracy number is worthless if factual sentences (ids 27, 28, 30) were silently
filed as speculation and never verified, and if mixed sentences leak unverified facts (id 21).
`design.md` §4 already made classification reliability the *gating* metric; it does not clear
the bar.

**What I am *not* claiming:** that the mode is impossible, or that the 84.4% is a population
rate. On clean, unambiguous sentences the approach is perfect (100%), and the dangerous
FP-on-clean axis is 0%. If the generator could be constrained to emit *only* clean
declaratives or clearly-marked speculation (never hedged, never mixed), rule-based
classification might suffice — but that constraint would defeat the "free reflection" premise.

**Options for review, in order of my recommendation:**
1. **Stop this branch** and record the structural limit. Invest the freed ~8–10 days in the
   two branches that passed their design review (constrained-creative → build;
   QA → already strong). *(Recommended, given deadline pressure.)*
2. **Re-scope, don't rebuild:** drop "free reflection" and require the generator to tag each
   sentence it emits as assertion-vs-speculation *at generation time* (the label exists before
   surface realization — the `support_guard` situation), turning this back into the *solved*
   pre-labelled problem. This is a different, smaller mode, not the one designed.
3. Only if a reviewer disputes the stress-set framing: rerun on a **naturalistic** sample of
   real generated reflection to get a true base rate of hedged/mixed sentences. But note the
   §3 argument predicts the base rate will be *high*, not low, for this mode specifically.

**I recommend option 1.** The negative result is clean, the reason is structural and
demonstrated (not vague), and it is the honest counterpart to the two branches that earned a
green light.
